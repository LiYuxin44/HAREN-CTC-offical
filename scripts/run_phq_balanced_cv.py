#!/usr/bin/env python3
"""Run the published PHQ-balanced 5-fold BCE-only protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SEEDS = (1234, 12345, 123456, 1234567, 12345678, 2024, 2025, 2026, 2027, 2028)
FOLDS = (0, 1, 2, 3, 4)
EPOCHS = 15
SELECTED_EPOCH = 14
PHQ_BINS = ("0-4", "5-9", "10-14", "15-19", "20-24")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(FOLDS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_protocol(args: argparse.Namespace) -> tuple[Path, dict]:
    if (
        len(set(args.seeds)) != len(args.seeds)
        or not set(args.folds).issubset(FOLDS)
    ):
        raise ValueError("Seeds must be unique and folds must be in 0..4")
    config_path = args.manifest_root / "manifest_config.json"
    config = json.loads(config_path.read_text())
    if (
        config.get("subjects") != 189
        or config.get("outer_folds") != 5
        or config.get("variant") not in {"offset", "nooffset"}
        or config.get("primary_stratification") != ["phq_bin"]
        or config.get("phq_bin_labels") != list(PHQ_BINS)
        or config.get("phq_used_for_assignment") is not True
    ):
        raise RuntimeError("Manifest root is not the PHQ-balanced protocol")
    data_root = Path(config["source_data_root"])
    if not (data_root / "pool").is_dir():
        raise FileNotFoundError(data_root / "pool")
    return data_root, config


def command_for(
    args: argparse.Namespace,
    *,
    data_root: Path,
    variant: str,
    fold: int,
    seed: int,
    output_dir: Path,
) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    fold_root = args.manifest_root / variant / f"fold_{fold}"
    train_manifest = fold_root / "train_dev_manifest.csv"
    test_manifest = fold_root / "test_manifest.csv"
    for path in (train_manifest, test_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    return [
        args.python,
        str(repo_root / "scripts" / "run_experiments.py"),
        "--python",
        args.python,
        "--data-root",
        str(data_root),
        "--gpu",
        str(args.gpu),
        "--seeds",
        str(seed),
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        "16",
        "--lr",
        "1e-4",
        "--weight-decay",
        "1e-5",
        "--dropout",
        "0.3",
        "--ctc-enabled",
        "0",
        "--ctc-mode",
        "shared_grad_norm",
        "--ctc-target-mode",
        "neutral",
        "--ctc-weight",
        "0",
        "--ctc-target-ratio",
        "5.0",
        "--ctc-warmup-epochs",
        "5",
        "--ctc-k",
        "10",
        "--ctc-grad-target-ratio",
        "0.003",
        "--ctc-grad-update-interval",
        "1",
        "--ctc-grad-ema-decay",
        "0",
        "--ctc-warmup-ratio",
        "0.1",
        "--ctc-loss-policy",
        "normalized_fp32",
        "--temporal-head-policy",
        "neutral_k1",
        "--crop-alignment-samples",
        "320",
        "--wavlm-mask-policy",
        "true_length",
        "--wavlm-preprocess-policy",
        "valid_then_pad",
        "--wavlm-batch-padding-policy",
        "fixed_10s",
        "--train-crop-policy",
        "epoch_keyed",
        "--optimizer-policy",
        "no_decay_warmup_cosine",
        "--head-arch-policy",
        "legacy_17m",
        "--head-init-policy",
        "legacy_stream",
        "--lr-warmup-ratio",
        "0.1",
        "--lr-min-ratio",
        "0.1",
        "--ema-decay",
        "0",
        "--save-training-state",
        "1",
        "--routing-init-policy",
        "legacy",
        "--sampling-policy",
        "utterance_class_balanced",
        "--aggregation-policy",
        "mean_probability",
        "--temporal-target-policy",
        "local_kmeans_ctc",
        "--grad-diagnostic-interval",
        "0",
        "--grad-clip-norm",
        "5",
        "--grad-clip-policy",
        "task_grouped",
        "--amp-dtype",
        "bf16",
        "--eval-crop-policy",
        "multi3",
        "--eval-window-stride-seconds",
        "5",
        "--test-policy",
        "none",
        "--split-mode",
        "test_tune",
        "--train-manifest",
        str(train_manifest),
        "--val-manifest",
        str(test_manifest),
        "--fold-index",
        str(fold),
        "--workers",
        str(args.workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--log-interval",
        "10",
        "--run-tag",
        f"phq5_seed{seed}_fold{fold}",
        "--output-dir",
        str(output_dir),
    ]


def main() -> None:
    args = parse_args()
    data_root, manifest_config = validate_protocol(args)
    commands = []
    for fold in args.folds:
        for seed in args.seeds:
            output_dir = args.output_root / f"fold_{fold}" / f"seed{seed}"
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(f"Refusing to overwrite {output_dir}")
            commands.append(
                (
                    fold,
                    seed,
                    command_for(
                        args,
                        data_root=data_root,
                        variant=str(manifest_config["variant"]),
                        fold=fold,
                        seed=seed,
                        output_dir=output_dir,
                    ),
                )
            )

    print(
        "WARNING: fold-test is evaluated at every epoch. Epoch 14 is the "
        "frozen post-hoc headline; these are not independent test estimates.",
        flush=True,
    )
    if args.dry_run:
        for _, _, command in commands:
            print(" ".join(command))
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol": "phq5_balanced_posthoc_v1",
        "manifest_config": str(
            (args.manifest_root / "manifest_config.json").resolve()
        ),
        "manifest_namespace": manifest_config.get("namespace"),
        "seeds": args.seeds,
        "folds": args.folds,
        "epochs": EPOCHS,
        "selected_epoch": SELECTED_EPOCH,
        "threshold": 0.5,
        "ctc_enabled": False,
        "ensemble": False,
        "test_selected_epoch": True,
        "independent_test_performance": False,
    }
    (args.output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    for fold, seed, command in commands:
        print(f"[run] fold={fold} seed={seed}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
