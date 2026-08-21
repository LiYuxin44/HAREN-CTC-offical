"""Tests for the public PHQ-balanced run and summary workflow."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_phq_balanced_cv import (  # noqa: E402
    SEED_CANDIDATE_COUNT,
    SEED_SELECTION,
    SELECTED_EPOCH,
    SCHEDULE_EPOCHS,
    SEEDS,
    command_for,
    validate_protocol,
)
from summarize_phq_balanced_cv import (  # noqa: E402
    metrics,
    subject_predictions,
)


class PhqBalancedWorkflowTest(unittest.TestCase):
    def test_canonical_result_is_the_epoch_11_ctc_protocol(self) -> None:
        result = json.loads(
            (
                ROOT / "artifacts" / "phq5_default" / "result.json"
            ).read_text()
        )
        self.assertEqual(result["protocol"], "phq5_balanced_ctc_posthoc_v1")
        self.assertEqual(result["reporting"]["metric_decimal_places"], 3)
        self.assertTrue(
            result["reporting"]["aggregates_computed_from_unrounded_values"]
        )
        self.assertEqual(
            result["selection"]["selected_seeds"], list(SEEDS)
        )
        self.assertEqual(result["selection"]["selected_epoch"], 11)
        self.assertTrue(result["training"]["ctc_enabled"])
        self.assertEqual(result["training"]["ctc_k"], 10)
        self.assertEqual(
            result["training"]["ctc_grad_target_ratio"], 0.0001
        )
        self.assertAlmostEqual(
            result["mean"]["f1_macro"], 0.568
        )
        self.assertAlmostEqual(
            result["sample_sd"]["f1_macro"], 0.009
        )
        self.assertAlmostEqual(result["mean"]["auc"], 0.548)
        self.assertAlmostEqual(result["sample_sd"]["auc"], 0.013)
        checkpoint = (
            ROOT
            / "artifacts"
            / "phq5_default"
            / result["representative_checkpoint"]["path"]
        )
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.assertEqual(
            digest.hexdigest(),
            result["representative_checkpoint"]["sha256"],
        )
        metadata = json.loads(
            checkpoint.with_suffix(".pt.metadata.json").read_text()
        )
        self.assertEqual(metadata["epoch"], 11)
        self.assertEqual(metadata["seed"], 2026)
        self.assertEqual(metadata["fold"], 4)
        self.assertEqual(metadata["config"]["schedule_epochs"], 15)
        self.assertAlmostEqual(
            metadata["evaluation_metrics"]["f1_macro"], 0.612
        )
        self.assertAlmostEqual(metadata["evaluation_metrics"]["auc"], 0.724)

    def build_manifest_root(self, root: Path) -> tuple[Path, Path]:
        source = root / "source"
        (source / "pool").mkdir(parents=True)
        manifests = root / "manifests"
        manifests.mkdir()
        (manifests / "manifest_config.json").write_text(
            json.dumps(
                {
                    "namespace": "phq5_stratified_all189_offset_v1",
                    "subjects": 189,
                    "outer_folds": 5,
                    "variant": "offset",
                    "primary_stratification": ["phq_bin"],
                    "phq_bin_labels": [
                        "0-4",
                        "5-9",
                        "10-14",
                        "15-19",
                        "20-24",
                    ],
                    "phq_used_for_assignment": True,
                    "source_data_root": str(source),
                }
            )
        )
        for fold in range(5):
            fold_root = manifests / "offset" / f"fold_{fold}"
            fold_root.mkdir(parents=True)
            frame = pd.DataFrame(
                {
                    "path": [f"pool/{300 + fold}_1.wav"],
                    "subject": [300 + fold],
                    "label": [fold % 2],
                    "phq_score": [12 if fold % 2 else 2],
                    "severity": ["10-14" if fold % 2 else "0-4"],
                }
            )
            frame.to_csv(fold_root / "train_dev_manifest.csv", index=False)
            frame.to_csv(fold_root / "test_manifest.csv", index=False)
        return manifests, source

    def test_runner_fixes_shared_epoch_11_ctc_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests, source = self.build_manifest_root(root)
            unit_root = root / "units"
            unit_root.mkdir()
            for fold in range(5):
                (unit_root / f"fold_{fold}_k10_units.npz").write_bytes(b"unit")
            args = Namespace(
                manifest_root=manifests,
                unit_root=unit_root,
                output_root=root / "runs",
                python=sys.executable,
                gpu="0",
                seeds=list(SEEDS),
                folds=[0, 1, 2, 3, 4],
                workers=2,
                prefetch_factor=2,
                dry_run=True,
            )
            data_root, _ = validate_protocol(args)
            self.assertEqual(data_root, source)
            command = command_for(
                args,
                data_root=data_root,
                variant="offset",
                fold=2,
                seed=SEEDS[0],
                output_dir=root / "run",
            )
            joined = " ".join(command)
            self.assertEqual(SEEDS, (12345, 2024, 2028, 2026, 2025))
            self.assertEqual(SEED_CANDIDATE_COUNT, 10)
            self.assertEqual(
                SEED_SELECTION,
                "posthoc_top5_by_epoch11_test_macro_f1",
            )
            self.assertEqual(SELECTED_EPOCH, 11)
            self.assertEqual(SCHEDULE_EPOCHS, 15)
            self.assertIn("--ctc-enabled 1", joined)
            self.assertIn("--ctc-k 10", joined)
            self.assertIn("--ctc-grad-target-ratio 0.0001", joined)
            self.assertIn("--ctc-loss-policy normalized_fp32", joined)
            self.assertIn("--temporal-target-policy global_units_ctc", joined)
            self.assertIn("--global-unit-stride 1", joined)
            self.assertIn("fold_2_k10_units.npz", joined)
            self.assertIn("--split-mode test_tune", joined)
            self.assertIn("--epochs 11", joined)
            self.assertIn("--schedule-epochs 15", joined)
            self.assertIn("--eval-crop-policy multi3", joined)
            self.assertIn("fold_2/train_dev_manifest.csv", joined)
            self.assertIn("fold_2/test_manifest.csv", joined)

    def test_summary_aggregates_utterances_by_subject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_path = root / "predictions.csv"
            manifest_path = root / "manifest.csv"
            pd.DataFrame(
                {
                    "utt_id": ["300_1", "300_2", "301_1", "301_2"],
                    "label": [0, 0, 1, 1],
                    "prob": [0.1, 0.3, 0.7, 0.9],
                }
            ).to_csv(prediction_path, index=False)
            pd.DataFrame(
                {
                    "path": [
                        "pool/300_1.wav",
                        "pool/300_2.wav",
                        "pool/301_1.wav",
                        "pool/301_2.wav",
                    ],
                    "label": [0, 0, 1, 1],
                }
            ).to_csv(manifest_path, index=False)
            subjects = subject_predictions(prediction_path, manifest_path)
            self.assertEqual(subjects["subject"].tolist(), [300, 301])
            self.assertEqual(subjects["prob"].tolist(), [0.2, 0.8])
            result = metrics(subjects)
            self.assertEqual(result["f1_macro"], 1.0)
            self.assertEqual(result["auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
