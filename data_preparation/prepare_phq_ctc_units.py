#!/usr/bin/env python3
"""Fit the five K=10 HuBERT codebooks used by the default PHQ 5CV run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoFeatureExtractor, HubertModel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from global_unit_cache import save_packed_unit_cache  # noqa: E402
from haren_ctc_utils import feature_output_lengths  # noqa: E402


FOLDS = tuple(range(5))
K = 10
MODEL_NAME = "facebook/hubert-large-ll60k"
NAMESPACE = "phq5_balanced_ctc_posthoc_v1"
EXPECTED_UTTERANCES = 5258
EXPECTED_SUBJECTS = 189


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fit-frames-per-chunk", type=int, default=64)
    parser.add_argument("--fit-frames-per-subject", type=int, default=1024)
    parser.add_argument("--fit-batch-frames", type=int, default=4096)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identifier_set_sha256(identifiers) -> str:
    canonical = "\n".join(sorted({str(value) for value in identifiers})) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def deterministic_identifier_order(identifiers, *, seed: int) -> list[str]:
    values = [str(value) for value in identifiers]
    if len(values) != len(set(values)):
        raise ValueError("Identifier order requires unique values")
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{int(seed)}\0{value}".encode("utf-8")
        ).digest(),
    )


def fold_manifests(manifest_root: Path, fold: int) -> dict[str, Path]:
    root = manifest_root / "offset" / f"fold_{fold}"
    return {
        role: root / f"{role}_manifest.csv"
        for role in ("train_dev", "test")
    }


def read_manifest(path: Path, role: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"path", "subject", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} lacks {required - set(frame.columns)}")
    result = frame.copy()
    result["role"] = role
    result["identifier"] = result["path"].astype(str).map(
        lambda value: Path(value).stem
    )
    if (
        result.empty
        or result["path"].duplicated().any()
        or result["identifier"].duplicated().any()
    ):
        raise ValueError(f"{path} has an invalid utterance inventory")
    return result


def read_inventory(
    manifest_root: Path,
) -> tuple[
    pd.DataFrame,
    dict[int, dict[str, Path]],
    dict[int, set[str]],
    dict[int, set[str]],
    dict[int, set[str]],
]:
    manifests = {
        fold: fold_manifests(manifest_root, fold) for fold in FOLDS
    }
    inventory_frames = []
    train_ids: dict[int, set[str]] = {}
    test_ids: dict[int, set[str]] = {}
    train_subjects: dict[int, set[str]] = {}
    for fold, paths in manifests.items():
        train = read_manifest(paths["train_dev"], "train_dev")
        test = read_manifest(paths["test"], "test")
        train_ids[fold] = set(train["identifier"].astype(str))
        test_ids[fold] = set(test["identifier"].astype(str))
        train_subjects[fold] = set(train["subject"].astype(str))
        test_subjects = set(test["subject"].astype(str))
        if (
            train_ids[fold] & test_ids[fold]
            or train_subjects[fold] & test_subjects
            or len(train_subjects[fold] | test_subjects) != EXPECTED_SUBJECTS
        ):
            raise RuntimeError(f"Fold {fold} train/test leakage")
        inventory_frames.extend((train, test))
    inventory = (
        pd.concat(inventory_frames, ignore_index=True)
        .drop_duplicates("path")
        .sort_values("path")
        .reset_index(drop=True)
    )
    if (
        len(inventory) != EXPECTED_UTTERANCES
        or inventory["subject"].nunique() != EXPECTED_SUBJECTS
        or inventory["identifier"].duplicated().any()
    ):
        raise RuntimeError(
            f"Unexpected offset inventory: {len(inventory)} utterances, "
            f"{inventory['subject'].nunique()} subjects"
        )
    return inventory, manifests, train_ids, test_ids, train_subjects


def load_wave(path: Path) -> torch.Tensor:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getframerate() != 16000
            or handle.getnchannels() != 1
            or handle.getsampwidth() != 2
        ):
            raise ValueError(f"Expected mono PCM16 16 kHz: {path}")
        pcm = handle.readframes(handle.getnframes())
    values = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(values)


def audio_chunks(waveform: torch.Tensor, max_samples: int):
    for start in range(0, waveform.numel(), max_samples):
        chunk = waveform[start : start + max_samples]
        if chunk.numel() >= 400:
            yield start, chunk


def extract_chunk_features(
    chunk: torch.Tensor,
    *,
    extractor,
    model,
    device: torch.device,
) -> torch.Tensor:
    inputs = extractor(
        chunk.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    values = inputs.input_values.to(device)
    mask = torch.ones_like(values, dtype=torch.long, device=device)
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type,
        enabled=device.type == "cuda",
        dtype=torch.bfloat16,
    ):
        outputs = model(
            values,
            attention_mask=mask,
            output_hidden_states=True,
        )
    return outputs.hidden_states[12][0].float()


def nearest_units(
    features: torch.Tensor, centers: torch.Tensor
) -> np.ndarray:
    distances = torch.cdist(features.unsqueeze(0), centers.unsqueeze(0))[0]
    return distances.argmin(dim=-1).to(dtype=torch.int64).cpu().numpy()


def subject_balanced_fit_matrix(
    subject_values: dict[str, np.ndarray],
    subjects: list[str],
    *,
    cap: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    ordered = sorted(str(subject) for subject in subjects)
    if (
        not ordered
        or len(ordered) != len(set(ordered))
        or int(cap) <= 0
        or any(subject not in subject_values for subject in ordered)
    ):
        raise ValueError("Invalid subject-balanced fit inputs")
    widths = {
        int(np.asarray(subject_values[subject]).shape[1])
        for subject in ordered
        if np.asarray(subject_values[subject]).ndim == 2
    }
    if len(widths) != 1 or any(
        np.asarray(subject_values[subject]).ndim != 2
        or len(subject_values[subject]) == 0
        for subject in ordered
    ):
        raise ValueError("Subject feature matrices must be nonempty and aligned")
    balanced_frames = min(
        int(cap), *(len(subject_values[subject]) for subject in ordered)
    )
    values = np.concatenate(
        [
            np.asarray(subject_values[subject])[:balanced_frames]
            for subject in ordered
        ],
        axis=0,
    )
    generator = np.random.default_rng(int(seed))
    return values[generator.permutation(len(values))], int(balanced_frames)


def main() -> None:
    args = parse_args()
    if (
        args.fit_frames_per_chunk < K
        or args.fit_batch_frames < K
        or args.fit_frames_per_subject < args.fit_frames_per_chunk
        or args.max_seconds <= 0
    ):
        raise ValueError("Invalid frame sampling or chunk configuration")
    manifest_config_path = args.manifest_root / "manifest_config.json"
    manifest_config = json.loads(manifest_config_path.read_text())
    if (
        manifest_config.get("subjects") != EXPECTED_SUBJECTS
        or manifest_config.get("outer_folds") != len(FOLDS)
        or manifest_config.get("variant") != "offset"
    ):
        raise RuntimeError("Manifest root is not the offset PHQ 5CV protocol")
    data_root = Path(manifest_config["source_data_root"]).resolve()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing nonempty output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    inventory, manifests, train_ids, test_ids, train_subjects = read_inventory(
        args.manifest_root
    )
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = HubertModel.from_pretrained(MODEL_NAME).to(device).eval()
    model_revision = str(
        getattr(model.config, "_commit_hash", None) or "unresolved"
    )
    if model_revision == "unresolved":
        raise RuntimeError("HuBERT artifact revision could not be resolved")
    for parameter in model.parameters():
        parameter.requires_grad = False
    max_samples = int(round(args.max_seconds * 16000))

    estimators = {
        fold: MiniBatchKMeans(
            n_clusters=K,
            random_state=args.seed + 1000 * fold + K,
            batch_size=args.fit_batch_frames,
            n_init=3,
            reassignment_ratio=0.0,
        )
        for fold in FOLDS
    }
    subject_fit_buffers: dict[int, dict[str, list[np.ndarray]]] = {
        fold: {subject: [] for subject in train_subjects[fold]}
        for fold in FOLDS
    }
    subject_fit_counts = {
        fold: {subject: 0 for subject in train_subjects[fold]}
        for fold in FOLDS
    }
    fit_contributing_ids = {fold: set() for fold in FOLDS}
    index_by_identifier = {
        str(row.identifier): index
        for index, row in enumerate(inventory.itertuples(index=False))
    }
    order = [
        index_by_identifier[identifier]
        for identifier in deterministic_identifier_order(
            index_by_identifier, seed=args.seed
        )
    ]
    generators = {
        fold: torch.Generator().manual_seed(args.seed + 1000 * fold)
        for fold in FOLDS
    }
    for ordinal, row_index in enumerate(order, start=1):
        row = inventory.iloc[row_index]
        relative = str(row["path"])
        identifier = str(row["identifier"])
        subject = str(row["subject"])
        eligible_folds = [
            fold for fold in FOLDS if identifier in train_ids[fold]
        ]
        if not eligible_folds:
            continue
        waveform = load_wave(data_root / relative)
        for _start, chunk in audio_chunks(waveform, max_samples):
            features = extract_chunk_features(
                chunk, extractor=extractor, model=model, device=device
            )
            for fold in eligible_folds:
                remaining = (
                    args.fit_frames_per_subject
                    - subject_fit_counts[fold][subject]
                )
                if remaining <= 0:
                    continue
                sample_count = min(
                    args.fit_frames_per_chunk, len(features), remaining
                )
                indices = torch.randperm(
                    len(features), generator=generators[fold]
                )[:sample_count]
                sample = features[
                    indices.to(features.device)
                ].cpu().numpy()[:remaining]
                subject_fit_buffers[fold][subject].append(sample)
                subject_fit_counts[fold][subject] += len(sample)
                fit_contributing_ids[fold].add(identifier)
        if ordinal % 100 == 0:
            print(f"fit pass {ordinal}/{len(order)}", flush=True)

    fit_frames_by_fold = {}
    for fold in FOLDS:
        subjects = sorted(train_subjects[fold])
        subject_values = {
            subject: np.concatenate(parts, axis=0)
            for subject, parts in subject_fit_buffers[fold].items()
            if parts
        }
        if any(subject not in subject_values for subject in subjects):
            raise RuntimeError(f"Fold {fold} has missing fit subjects")
        fold_values, balanced_frames = subject_balanced_fit_matrix(
            subject_values,
            subjects,
            cap=args.fit_frames_per_subject,
            seed=args.seed + 1000 * fold,
        )
        if balanced_frames < K:
            raise RuntimeError(f"Fold {fold} has too few frames per subject")
        for start in range(0, len(fold_values), args.fit_batch_frames):
            estimators[fold].partial_fit(
                fold_values[start : start + args.fit_batch_frames]
            )
        fit_frames_by_fold[fold] = {
            "subjects": len(subjects),
            "frames_per_subject": int(balanced_frames),
            "total_frames": int(len(fold_values)),
            "candidate_identifiers": len(train_ids[fold]),
            "candidate_identifier_sha256": identifier_set_sha256(
                train_ids[fold]
            ),
            "contributing_identifiers": len(fit_contributing_ids[fold]),
            "contributing_identifier_sha256": identifier_set_sha256(
                fit_contributing_ids[fold]
            ),
        }
    del subject_fit_buffers

    centers = {
        fold: torch.from_numpy(estimator.cluster_centers_)
        .float()
        .to(device)
        for fold, estimator in estimators.items()
    }
    sequences = {fold: {} for fold in FOLDS}
    for ordinal, row in enumerate(inventory.itertuples(index=False), start=1):
        relative = str(row.path)
        identifier = str(row.identifier)
        eligible_folds = [
            fold for fold in FOLDS if identifier in train_ids[fold]
        ]
        if not eligible_folds:
            continue
        waveform = load_wave(data_root / relative)
        per_fold_parts = {fold: [] for fold in eligible_folds}
        chunks = list(audio_chunks(waveform, max_samples))
        for chunk_index, (_start, chunk) in enumerate(chunks):
            features = extract_chunk_features(
                chunk, extractor=extractor, model=model, device=device
            )
            for fold in eligible_folds:
                units = nearest_units(features, centers[fold])
                per_fold_parts[fold].append(units)
                if chunk_index < len(chunks) - 1:
                    per_fold_parts[fold].append(units[-1:])
        expected = int(
            feature_output_lengths(
                torch.tensor([waveform.numel()]),
                model.config.conv_kernel,
                model.config.conv_stride,
            )[0]
        )
        for fold, parts in per_fold_parts.items():
            sequence = np.concatenate(parts)
            if len(sequence) < expected:
                sequence = np.pad(
                    sequence, (0, expected - len(sequence)), mode="edge"
                )
            sequences[fold][identifier] = sequence[:expected]
        if ordinal % 100 == 0:
            print(f"assignment pass {ordinal}/{len(inventory)}", flush=True)
    for fold in FOLDS:
        if set(sequences[fold]) != train_ids[fold]:
            raise RuntimeError(f"Fold {fold} cache inventory mismatch")

    software_versions = {
        package: importlib.metadata.version(package)
        for package in ("numpy", "scikit-learn", "torch", "transformers")
    }
    products = []
    for fold, estimator in estimators.items():
        codebook_path = args.output_root / f"fold_{fold}_k{K}_codebook.npz"
        np.savez_compressed(
            codebook_path,
            centers=estimator.cluster_centers_.astype(np.float32),
        )
        cache_path = args.output_root / f"fold_{fold}_k{K}_units.npz"
        metadata = {
            "namespace": NAMESPACE,
            "source_data_root": str(data_root),
            "source_manifest_config": str(manifest_config_path.resolve()),
            "source_manifest_config_sha256": file_sha256(
                manifest_config_path
            ),
            "fold": fold,
            "k": K,
            "model": MODEL_NAME,
            "model_revision": model_revision,
            "hidden_layer": 12,
            "fit_scope": "outer_train",
            "codebook_fit_split": "exact_outer_train_dev_manifest_ids_only",
            "cached_roles": ["train_dev"],
            "excluded_roles": ["test"],
            "train_utterances": len(train_ids[fold]),
            "dev_utterances": 0,
            "cached_utterances": len(train_ids[fold]),
            "train_identifier_sha256": identifier_set_sha256(
                train_ids[fold]
            ),
            "dev_identifier_sha256": "",
            "cached_identifier_sha256": identifier_set_sha256(
                train_ids[fold]
            ),
            "codebook_sampling_policy": (
                "exact_train_id_subject_equal_chunk_uniform_frames_v2"
            ),
            "fit_frames_per_subject_cap": args.fit_frames_per_subject,
            "fit_subjects": fit_frames_by_fold[fold]["subjects"],
            "fit_frames_per_subject": fit_frames_by_fold[fold][
                "frames_per_subject"
            ],
            "fit_total_frames": fit_frames_by_fold[fold]["total_frames"],
            "fit_candidate_utterances": fit_frames_by_fold[fold][
                "candidate_identifiers"
            ],
            "fit_candidate_identifier_sha256": fit_frames_by_fold[fold][
                "candidate_identifier_sha256"
            ],
            "fit_contributing_utterances": fit_frames_by_fold[fold][
                "contributing_identifiers"
            ],
            "fit_contributing_identifier_sha256": fit_frames_by_fold[fold][
                "contributing_identifier_sha256"
            ],
            "manifest_sha256": {
                "train": file_sha256(manifests[fold]["train_dev"]),
                "train_dev": file_sha256(manifests[fold]["train_dev"]),
                "test": file_sha256(manifests[fold]["test"]),
            },
            "codebook_path": str(codebook_path.resolve()),
            "codebook_sha256": file_sha256(codebook_path),
            "frame_stride_samples": 320,
            "chunk_samples": max_samples,
            "chunk_boundary_policy": (
                "duplicate_previous_unit_for_omitted_crossing_frame"
            ),
            "seed": args.seed,
            "software_versions": software_versions,
        }
        save_packed_unit_cache(cache_path, sequences[fold], metadata=metadata)
        products.append(
            {
                **metadata,
                "cache_path": str(cache_path.resolve()),
                "cache_sha256": file_sha256(cache_path),
            }
        )
    summary = {
        "namespace": NAMESPACE,
        "fit_scope": "outer_train",
        "inventory_utterances": len(inventory),
        "inventory_subjects": int(inventory["subject"].nunique()),
        "cache_roles": ["train_dev"],
        "excluded_roles": ["test"],
        "model": MODEL_NAME,
        "model_revision": model_revision,
        "codebooks": products,
    }
    manifest_path = args.output_root / "global_unit_manifest.json"
    manifest_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
