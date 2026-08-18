#!/usr/bin/env python3
"""Summarize the fixed epoch-14 PHQ-balanced 5-fold result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from run_phq_balanced_cv import (
    FOLDS,
    SEED_CANDIDATE_COUNT,
    SEED_SELECTION,
    SELECTED_EPOCH,
    SEEDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def subject_predictions(path: Path, manifest_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if set(frame.columns) != {"utt_id", "label", "prob"} or frame.empty:
        raise RuntimeError(f"Invalid prediction file: {path}")
    frame["utt_id"] = frame["utt_id"].astype(str)
    probabilities = frame["prob"].to_numpy(dtype=float)
    if (
        frame["utt_id"].duplicated().any()
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise RuntimeError(f"Invalid prediction values: {path}")

    manifest = pd.read_csv(manifest_path)
    expected = pd.DataFrame(
        {
            "utt_id": manifest["path"].map(lambda value: Path(value).stem),
            "label": manifest["label"].astype(int),
        }
    )
    if (
        expected["utt_id"].duplicated().any()
        or set(frame["utt_id"]) != set(expected["utt_id"])
    ):
        raise RuntimeError(f"Prediction/manifest mismatch: {path}")
    observed_labels = frame.set_index("utt_id")["label"].astype(int)
    expected_labels = expected.set_index("utt_id")["label"].astype(int)
    if not observed_labels.sort_index().equals(expected_labels.sort_index()):
        raise RuntimeError(f"Prediction labels mismatch: {path}")

    frame["subject"] = frame["utt_id"].str.split("_").str[0].astype(int)
    if (frame.groupby("subject")["label"].nunique() != 1).any():
        raise RuntimeError(f"Inconsistent subject labels: {path}")
    return (
        frame.groupby("subject", as_index=False)
        .agg(label=("label", "first"), prob=("prob", "mean"))
        .sort_values("subject")
        .reset_index(drop=True)
    )


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame["label"].astype(int).to_numpy()
    probabilities = frame["prob"].astype(float).to_numpy()
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "f1_macro": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "auc": float(roc_auc_score(labels, probabilities)),
        "f1_pos": float(
            f1_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "f1_neg": float(
            f1_score(labels, predictions, pos_label=0, zero_division=0)
        ),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision_macro": float(
            precision_score(
                labels, predictions, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                labels, predictions, average="macro", zero_division=0
            )
        ),
    }


def prediction_path(run_dir: Path) -> Path:
    paths = list(
        run_dir.glob(f"test_tuning_pred_*_epoch{SELECTED_EPOCH:02d}.csv")
    )
    if len(paths) != 1:
        raise RuntimeError(
            f"{run_dir}: expected one epoch-{SELECTED_EPOCH} prediction file"
        )
    return paths[0]


def main() -> None:
    args = parse_args()
    manifest_config = json.loads(
        (args.manifest_root / "manifest_config.json").read_text()
    )
    variant = str(manifest_config.get("variant", ""))
    if variant not in {"offset", "nooffset"}:
        raise RuntimeError("Manifest config has no supported data variant")
    rows = []
    pooled_predictions = []
    for seed in SEEDS:
        fold_predictions = []
        for fold in FOLDS:
            run_dir = args.runs_root / f"fold_{fold}" / f"seed{seed}"
            config = json.loads((run_dir / "run_config.json").read_text())
            if (
                config.get("seeds") != [seed]
                or config.get("epochs") != 15
                or config.get("ctc_enabled") is not False
                or config.get("split_mode") != "test_tune"
                or config.get("eval_crop_policy") != "multi3"
            ):
                raise RuntimeError(f"{run_dir}: protocol mismatch")
            manifest = (
                args.manifest_root
                / variant
                / f"fold_{fold}"
                / "test_manifest.csv"
            )
            fold_predictions.append(
                subject_predictions(
                    prediction_path(run_dir), manifest
                ).assign(fold=fold)
            )
        pooled = pd.concat(fold_predictions, ignore_index=True)
        if len(pooled) != 189 or pooled["subject"].duplicated().any():
            raise RuntimeError(f"Seed {seed}: incomplete OOF coverage")
        rows.append(
            {
                "seed": seed,
                "subjects": 189,
                "epoch": SELECTED_EPOCH,
                "threshold": 0.5,
                **metrics(pooled),
            }
        )
        pooled_predictions.append(pooled.assign(seed=seed))

    seed_metrics = pd.DataFrame(rows).sort_values("seed")
    predictions = pd.concat(pooled_predictions, ignore_index=True).sort_values(
        ["seed", "subject"]
    )
    metric_names = [
        column
        for column in seed_metrics.columns
        if column not in {"seed", "subjects", "epoch", "threshold"}
    ]
    result = {
        "protocol": "phq5_balanced_posthoc_v1",
        "subjects": 189,
        "folds": 5,
        "seeds": list(SEEDS),
        "seed_count": len(SEEDS),
        "seed_candidate_count": SEED_CANDIDATE_COUNT,
        "seed_selection": SEED_SELECTION,
        "post_hoc_seed_selection": True,
        "test_used_for_seed_selection": True,
        "selected_epoch": SELECTED_EPOCH,
        "epoch_selection": "shared_test_selected",
        "threshold": 0.5,
        "ctc_enabled": False,
        "ensemble_computed": False,
        "test_selected_epoch": True,
        "independent_test_performance": False,
        "mean": {
            name: float(seed_metrics[name].mean()) for name in metric_names
        },
        "sample_sd": {
            name: float(seed_metrics[name].std(ddof=1))
            for name in metric_names
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(args.output_root / "seed_metrics.csv", index=False)
    predictions.to_csv(
        args.output_root / "subject_predictions.csv", index=False
    )
    result_path = args.output_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
