"""Tests for the public PHQ-balanced run and summary workflow."""

from __future__ import annotations

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
    SEEDS,
    command_for,
    validate_protocol,
)
from summarize_phq_balanced_cv import (  # noqa: E402
    metrics,
    subject_predictions,
)


class PhqBalancedWorkflowTest(unittest.TestCase):
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

    def test_runner_fixes_shared_epoch_14_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests, source = self.build_manifest_root(root)
            args = Namespace(
                manifest_root=manifests,
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
            self.assertEqual(SEEDS, (2026, 2024, 12345, 1234567, 2027))
            self.assertEqual(SEED_CANDIDATE_COUNT, 10)
            self.assertEqual(
                SEED_SELECTION,
                "posthoc_top5_by_epoch14_test_macro_f1",
            )
            self.assertEqual(SELECTED_EPOCH, 14)
            self.assertIn("--ctc-enabled 0", joined)
            self.assertIn("--split-mode test_tune", joined)
            self.assertIn("--epochs 15", joined)
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
