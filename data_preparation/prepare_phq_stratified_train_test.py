#!/usr/bin/env python3
"""Build subject-level PHQ-stratified five-fold manifests for all 189 subjects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from preprocess import OFFSET_MAP, process_subject


NAMESPACE = "phq5_stratified_all189_offset_v1"
OUTER_FOLDS = 5
INNER_DEV_SUBJECTS = 31
PHQ_BIN_EDGES = (-1, 4, 9, 14, 19, 24)
PHQ_BIN_LABELS = ("0-4", "5-9", "10-14", "15-19", "20-24")
EXPECTED_BIN_COUNTS = {
    "0-4": 86,
    "5-9": 46,
    "10-14": 30,
    "15-19": 20,
    "20-24": 7,
}
EXPECTED_SOURCE_COUNTS = {
    "official_train": 107,
    "official_dev": 35,
    "official_test": 47,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", required=True, type=Path)
    parser.add_argument(
        "--source-data-root",
        type=Path,
        help="Reuse an existing all-189 audio root containing pool/.",
    )
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--trans-dir", type=Path)
    parser.add_argument("--offset", action="store_true")
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_metadata(path: Path, source_split: str) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(
        columns={
            "PHQ_Binary": "PHQ8_Binary",
            "PHQ_Score": "PHQ8_Score",
        }
    )
    required = {"Participant_ID", "PHQ8_Binary", "PHQ8_Score"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path}: missing {sorted(required - set(frame))}")
    result = frame[
        ["Participant_ID", "PHQ8_Binary", "PHQ8_Score"]
    ].rename(
        columns={
            "PHQ8_Binary": "binary",
            "PHQ8_Score": "phq_score",
        }
    )
    result["source_split"] = source_split
    return result


def load_all_subjects(label_dir: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    metadata_paths = {
        "official_train": label_dir / "train_split_Depression_AVEC2017.csv",
        "official_dev": label_dir / "dev_split_Depression_AVEC2017.csv",
        "official_test": label_dir / "full_test_split.csv",
    }
    subjects = (
        pd.concat(
            [
                canonical_metadata(path, source)
                for source, path in metadata_paths.items()
            ],
            ignore_index=True,
        )
        .sort_values("Participant_ID")
        .reset_index(drop=True)
    )
    if (
        len(subjects) != 189
        or subjects["Participant_ID"].duplicated().any()
        or subjects["source_split"].value_counts().to_dict()
        != EXPECTED_SOURCE_COUNTS
    ):
        raise RuntimeError("Expected 189 unique DAIC-WOZ subjects")
    subjects["binary"] = subjects["binary"].astype(int)
    subjects["phq_score"] = subjects["phq_score"].astype(int)
    expected_binary = (subjects["phq_score"] >= 10).astype(int)
    if not np.array_equal(subjects["binary"].to_numpy(), expected_binary):
        raise RuntimeError("PHQ binary/score mismatch")
    return subjects, metadata_paths


def prepare_audio_pool(
    subjects: pd.DataFrame,
    *,
    pool_dir: Path,
    audio_dir: Path,
    transcript_dir: Path,
    offset: bool,
) -> dict[int, int]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    available: dict[int, int] = {}
    offset_map = OFFSET_MAP if offset else {}
    for row in subjects.itertuples(index=False):
        subject = int(row.Participant_ID)
        required = 46 if int(row.binary) else 20
        count = process_subject(
            subject,
            int(row.binary),
            int(row.phq_score),
            str(pool_dir),
            str(audio_dir),
            str(transcript_dir),
            offset_map=offset_map,
            sample_mode="fixed",
            fixed_n=required,
            min_duration=1.0,
        )
        available[subject] = int(count)
    return audit_audio_pool(subjects, pool_dir, available)


def audit_audio_pool(
    subjects: pd.DataFrame,
    pool_dir: Path,
    reported_counts: dict[int, int] | None = None,
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in subjects.itertuples(index=False):
        subject = int(row.Participant_ID)
        paths = sorted(
            pool_dir.glob(f"{subject}_*.wav"),
            key=lambda path: int(path.stem.split("_")[1]),
        )
        required = 46 if int(row.binary) else 20
        indices = [int(path.stem.split("_")[1]) for path in paths]
        if (
            len(paths) < required
            or indices != list(range(1, len(paths) + 1))
            or (
                reported_counts is not None
                and reported_counts.get(subject) != len(paths)
            )
        ):
            raise RuntimeError(f"Subject {subject}: invalid audio inventory")
        for wav in paths:
            if (
                int(wav.with_suffix(".label").read_text().strip())
                != int(row.binary)
                or int(wav.with_suffix(".phq_label").read_text().strip())
                != int(row.phq_score)
            ):
                raise RuntimeError(f"Subject metadata mismatch: {wav}")
        counts[subject] = len(paths)
    inventory_subjects = {
        int(path.stem.split("_")[0]) for path in pool_dir.glob("*.wav")
    }
    if inventory_subjects != set(subjects["Participant_ID"].astype(int)):
        raise RuntimeError("Audio pool does not contain exactly 189 subjects")
    return counts


def add_available_counts(
    frame: pd.DataFrame, available_counts: dict[int, int]
) -> pd.DataFrame:
    result = frame.copy()
    result["available_clips"] = result["Participant_ID"].map(available_counts)
    if result["available_clips"].isna().any():
        raise RuntimeError("Missing pooled audio")
    return result


def write_manifest(
    subjects: pd.DataFrame,
    *,
    pool_dir: Path,
    source_data_root: Path,
    split: str,
    output_path: Path,
) -> None:
    rows = []
    for row in subjects.itertuples(index=False):
        clip_count = (
            (46 if int(row.binary) else 18) if split == "train" else 20
        )
        clip_count = min(clip_count, int(row.available_clips))
        for clip_index in range(1, clip_count + 1):
            wav = pool_dir / f"{int(row.Participant_ID)}_{clip_index}.wav"
            if not (
                wav.is_file()
                and wav.with_suffix(".label").is_file()
                and wav.with_suffix(".phq_label").is_file()
            ):
                raise FileNotFoundError(wav)
            rows.append(
                {
                    "path": os.path.relpath(wav, source_data_root),
                    "subject": int(row.Participant_ID),
                    "label": int(row.binary),
                    "phq_score": int(row.phq_score),
                    "severity": str(row.phq_bin),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def add_phq_bins(subjects: pd.DataFrame) -> pd.DataFrame:
    result = subjects.copy()
    result["phq_bin"] = pd.cut(
        result["phq_score"],
        bins=PHQ_BIN_EDGES,
        labels=PHQ_BIN_LABELS,
        include_lowest=True,
    ).astype(str)
    if not result["phq_bin"].isin(PHQ_BIN_LABELS).all():
        raise RuntimeError("PHQ score outside 0-24")
    counts = (
        result["phq_bin"]
        .value_counts()
        .reindex(PHQ_BIN_LABELS, fill_value=0)
        .astype(int)
        .to_dict()
    )
    if counts != EXPECTED_BIN_COUNTS:
        raise RuntimeError(f"Unexpected PHQ-bin inventory: {counts}")
    # Persist the requested five-bin definition in every generated manifest.
    result["severity"] = result["phq_bin"]
    return result


def assign_outer_folds(subjects: pd.DataFrame, seed: int) -> pd.DataFrame:
    assigned = subjects.copy().reset_index(drop=True)
    assigned["outer_fold"] = -1
    splitter = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=seed,
    )
    for fold, (_, test_positions) in enumerate(
        splitter.split(assigned, assigned["phq_bin"])
    ):
        assigned.loc[test_positions, "outer_fold"] = fold
    if set(assigned["outer_fold"].astype(int)) != set(range(OUTER_FOLDS)):
        raise RuntimeError("Outer assignment omitted a fold")
    audit_outer_assignment(assigned)
    return assigned


def audit_outer_assignment(assigned: pd.DataFrame) -> None:
    table = pd.crosstab(assigned["outer_fold"], assigned["phq_bin"]).reindex(
        index=range(OUTER_FOLDS),
        columns=PHQ_BIN_LABELS,
        fill_value=0,
    )
    if (
        len(assigned) != 189
        or assigned["Participant_ID"].duplicated().any()
        or set(table.sum(axis=1)) != {37, 38}
        or (table.max(axis=0) - table.min(axis=0)).max() > 1
        or table.sum(axis=0).astype(int).to_dict() != EXPECTED_BIN_COUNTS
    ):
        raise RuntimeError(f"Invalid outer PHQ balance:\n{table}")


def assign_inner_roles(
    outer_train: pd.DataFrame,
    *,
    seed: int,
    fold: int,
) -> pd.DataFrame:
    assigned = outer_train.copy().reset_index(drop=True)
    assigned["inner_role"] = "train"
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=INNER_DEV_SUBJECTS,
        random_state=seed * 100 + fold,
    )
    _, dev_positions = next(
        splitter.split(assigned, assigned["phq_bin"])
    )
    assigned.loc[dev_positions, "inner_role"] = "dev"
    if (
        len(assigned.loc[assigned["inner_role"] == "dev"])
        != INNER_DEV_SUBJECTS
        or set(assigned["inner_role"]) != {"train", "dev"}
    ):
        raise RuntimeError(f"Fold {fold}: invalid inner assignment")
    return assigned


def describe(frame: pd.DataFrame, *, fold: int, split: str) -> dict:
    counts = (
        frame["phq_bin"]
        .value_counts()
        .reindex(PHQ_BIN_LABELS, fill_value=0)
    )
    row: dict[str, object] = {
        "fold": fold,
        "split": split,
        "subjects": len(frame),
        "negative": int((frame["binary"] == 0).sum()),
        "positive": int((frame["binary"] == 1).sum()),
        "positive_rate": float(frame["binary"].mean()),
        "phq_mean": float(frame["phq_score"].mean()),
        "phq_sd": float(frame["phq_score"].std(ddof=0)),
    }
    for label in PHQ_BIN_LABELS:
        row[f"phq_{label}_count"] = int(counts[label])
        row[f"phq_{label}_rate"] = float(counts[label] / len(frame))
    for source in EXPECTED_SOURCE_COUNTS:
        row[f"{source}_subjects"] = int(
            (frame["source_split"] == source).sum()
        )
    return row


def manifest_set_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: str(value)):
        digest.update(str(path.resolve()).encode())
        digest.update(b"\0")
        digest.update(file_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_products(
    subjects: pd.DataFrame,
    *,
    available_counts: dict[int, int],
    source_data_root: Path,
    out_root: Path,
    variant: str,
    seed: int,
) -> tuple[list[Path], dict[str, object]]:
    assignments_root = out_root / "assignments"
    manifests_root = out_root / variant
    assignments_root.mkdir(parents=True, exist_ok=True)
    outer = assign_outer_folds(subjects, seed)
    outer_path = assignments_root / "outer_assignments.csv"
    outer.to_csv(outer_path, index=False)

    manifest_paths: list[Path] = []
    distribution_rows: list[dict] = []
    inner_paths: dict[str, str] = {}
    pool_dir = source_data_root / "pool"
    for fold in range(OUTER_FOLDS):
        test = outer.loc[outer["outer_fold"] == fold].copy()
        train_dev = outer.loc[outer["outer_fold"] != fold].copy()
        inner = assign_inner_roles(train_dev, seed=seed, fold=fold)
        train = inner.loc[inner["inner_role"] == "train"].copy()
        dev = inner.loc[inner["inner_role"] == "dev"].copy()
        if (
            set(train["Participant_ID"]) & set(dev["Participant_ID"])
            or set(train_dev["Participant_ID"]) & set(test["Participant_ID"])
            or len(set(train_dev["Participant_ID"]) | set(test["Participant_ID"]))
            != 189
        ):
            raise RuntimeError(f"Fold {fold}: subject leakage")

        inner_path = assignments_root / f"fold_{fold}_inner_assignments.csv"
        inner.to_csv(inner_path, index=False)
        inner_paths[str(fold)] = str(inner_path.resolve())
        for split, frame in (
            ("train", train),
            ("dev", dev),
            ("train_dev", train_dev),
            ("test", test),
        ):
            distribution_rows.append(describe(frame, fold=fold, split=split))

        fold_root = manifests_root / f"fold_{fold}"
        specs = (
            ("train_manifest.csv", train, "train"),
            ("dev_manifest.csv", dev, "val"),
            ("train_dev_manifest.csv", train_dev, "train"),
            ("test_manifest.csv", test, "val"),
        )
        for filename, frame, sampling_role in specs:
            path = fold_root / filename
            write_manifest(
                add_available_counts(frame, available_counts),
                pool_dir=pool_dir,
                source_data_root=source_data_root,
                split=sampling_role,
                output_path=path,
            )
            manifest_paths.append(path)

    distribution = pd.DataFrame(distribution_rows)
    distribution_path = assignments_root / "distributions.csv"
    distribution.to_csv(distribution_path, index=False)
    dev = distribution.loc[distribution["split"] == "dev"]
    for label in PHQ_BIN_LABELS:
        if dev[f"phq_{label}_count"].nunique() != 1:
            raise RuntimeError(f"Inner-dev {label} count differs across folds")
    return manifest_paths, {
        "outer_assignments": str(outer_path.resolve()),
        "inner_assignments": inner_paths,
        "distributions": str(distribution_path.resolve()),
        "distributions_sha256": file_sha256(distribution_path),
    }


def main() -> None:
    args = parse_args()
    if args.out_root.exists() and any(args.out_root.iterdir()):
        raise FileExistsError(f"Output is not empty: {args.out_root}")
    subjects, metadata_paths = load_all_subjects(args.label_dir)
    subjects = add_phq_bins(subjects)
    variant = "offset" if args.offset else "nooffset"
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.source_data_root is None:
        if args.audio_dir is None or args.trans_dir is None:
            raise ValueError(
                "Provide --source-data-root or both --audio-dir and --trans-dir"
            )
        source_data_root = (args.out_root / "source_data").resolve()
        available_counts = prepare_audio_pool(
            subjects,
            pool_dir=source_data_root / "pool",
            audio_dir=args.audio_dir,
            transcript_dir=args.trans_dir,
            offset=args.offset,
        )
        source_config: dict[str, str] | None = None
    else:
        source_data_root = args.source_data_root.resolve()
        available_counts = audit_audio_pool(
            subjects, source_data_root / "pool"
        )
        source_config_path = source_data_root.parent / "manifest_config.json"
        source_config = (
            {
                "path": str(source_config_path.resolve()),
                "sha256": file_sha256(source_config_path),
            }
            if source_config_path.is_file()
            else None
        )
    manifests, assignments = write_products(
        subjects,
        available_counts=available_counts,
        source_data_root=source_data_root,
        out_root=args.out_root,
        variant=variant,
        seed=args.seed,
    )
    config = {
        "namespace": NAMESPACE,
        "subjects": 189,
        "variant": variant,
        "outer_folds": OUTER_FOLDS,
        "inner_dev_subjects": INNER_DEV_SUBJECTS,
        "assignment_seed": args.seed,
        "primary_stratification": ["phq_bin"],
        "phq_bin_edges": list(PHQ_BIN_EDGES),
        "phq_bin_labels": list(PHQ_BIN_LABELS),
        "phq_bin_counts": EXPECTED_BIN_COUNTS,
        "phq_used_for_assignment": True,
        "source_split_used_for_assignment": False,
        "training_subjects": "fold_train_plus_dev",
        "test_subjects_used_for_weight_updates": False,
        "official_test_consumed_as_cv_data": True,
        "independent_test_performance": False,
        "source_data_root": str(source_data_root),
        "source_manifest_config": source_config,
        "pool": str((source_data_root / "pool").resolve()),
        "pool_subjects": len(available_counts),
        "metadata_sha256": {
            source: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for source, path in metadata_paths.items()
        },
        "assignments": assignments,
        "manifest_count": len(manifests),
        "manifest_set_sha256": manifest_set_sha256(manifests),
        "manifests": {
            str(path.relative_to(args.out_root)): file_sha256(path)
            for path in sorted(manifests)
        },
    }
    config_path = args.out_root / "manifest_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(config_path)


if __name__ == "__main__":
    main()
