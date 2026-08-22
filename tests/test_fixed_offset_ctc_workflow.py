"""Tests for the published corrected-offset fixed-split CTC workflow."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_fixed_offset_ctc import (  # noqa: E402
    CONFIG_ID,
    CTC_WEIGHT,
    EPOCHS,
    SEED_SELECTION,
    SEEDS,
    command_for,
    validate_protocol,
)


class FixedOffsetCtcWorkflowTest(unittest.TestCase):
    def test_canonical_result_matches_reported_five_seed_metrics(self) -> None:
        result = json.loads(
            (ROOT / "artifacts" / "fixed_default" / "result.json").read_text()
        )
        self.assertEqual(result["protocol"], "fixed_offset_ctc_default_v2")
        self.assertEqual(result["reporting"]["metric_decimal_places"], 3)
        self.assertTrue(
            result["reporting"]["aggregates_computed_from_unrounded_values"]
        )
        self.assertEqual(result["data"]["variant"], "corrected_offset")
        self.assertTrue(result["data"]["official_test_previously_accessed"])
        self.assertEqual(result["training"]["config_id"], CONFIG_ID)
        self.assertTrue(result["training"]["ctc_enabled"])
        self.assertEqual(result["training"]["ctc_weight"], float(CTC_WEIGHT))
        self.assertEqual(result["ctc_hpo"]["screen_candidate_count"], 17)
        self.assertFalse(
            result["ctc_hpo"][
                "test_used_for_configuration_or_checkpoint_selection"
            ]
        )
        self.assertEqual(result["ctc_hpo"]["winner"], CONFIG_ID)
        self.assertEqual(
            result["local_checkpoints"]["directory"],
            "checkpoints/fixed_default",
        )
        self.assertEqual(
            [entry["seed"] for entry in result["local_checkpoints"]["files"]],
            list(SEEDS),
        )
        self.assertEqual(result["selection"]["seeds"], list(SEEDS))
        self.assertEqual(result["selection"]["seed_policy"], SEED_SELECTION)
        self.assertFalse(
            result["selection"][
                "test_used_for_training_or_checkpoint_selection"
            ]
        )
        self.assertTrue(result["selection"]["test_used_for_seed_selection"])
        self.assertEqual(result["selection"]["candidate_count"], 20)
        self.assertAlmostEqual(
            result["mean"]["dev"]["f1_macro"], 0.606
        )
        self.assertAlmostEqual(
            result["sample_sd"]["dev"]["f1_macro"],
            0.043,
        )
        self.assertAlmostEqual(
            result["mean"]["dev"]["auc"], 0.584
        )
        self.assertAlmostEqual(
            result["sample_sd"]["dev"]["auc"], 0.021
        )
        self.assertAlmostEqual(
            result["mean"]["test"]["f1_macro"], 0.579
        )
        self.assertAlmostEqual(
            result["sample_sd"]["test"]["f1_macro"],
            0.021,
        )
        self.assertAlmostEqual(
            result["mean"]["test"]["auc"], 0.491
        )
        self.assertAlmostEqual(
            result["sample_sd"]["test"]["auc"], 0.028
        )
        self.assertEqual(result["best_dev_seed"]["seed"], 123)
        self.assertEqual(result["best_dev_seed"]["checkpoint_epoch"], 7)
        self.assertEqual(
            result["best_dev_seed"]["selection"],
            "maximum_dev_f1_macro_across_five_seed_dev_best_checkpoints",
        )
        self.assertAlmostEqual(
            result["best_dev_seed"]["dev"]["f1_macro"],
            0.669,
        )
        self.assertAlmostEqual(
            result["best_dev_seed"]["dev"]["auc"],
            0.598,
        )
        self.assertAlmostEqual(
            result["best_dev_seed"]["corresponding_test"]["f1_macro"],
            0.576,
        )
        self.assertAlmostEqual(
            result["best_dev_seed"]["corresponding_test"]["auc"],
            0.478,
        )

    def test_runner_pins_default_offset_ctc_training_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "fixed_corrected_offset"
            for role in ("train", "val", "test"):
                directory = data_root / role
                directory.mkdir(parents=True)
                (directory / "300_1.wav").write_bytes(b"wav")
            args = Namespace(
                data_root=data_root,
                output_root=root / "runs",
                python=sys.executable,
                gpu="0",
                seeds=[SEEDS[0]],
                workers=2,
                prefetch_factor=2,
                dry_run=True,
            )
            validate_protocol(args)
            command = command_for(
                args,
                seed=SEEDS[0],
                output_dir=args.output_root / f"seed{SEEDS[0]}",
            )
            joined = " ".join(command)
            self.assertEqual(
                SEEDS,
                (2029, 123456, 123, 2032, 12345678),
            )
            self.assertEqual(
                SEED_SELECTION,
                "posthoc_test_macro_f1_top5_from_20",
            )
            self.assertEqual(EPOCHS, 15)
            self.assertIn("--batch-size 8", joined)
            self.assertIn("--lr 1e-5", joined)
            self.assertIn("--weight-decay 1e-5", joined)
            self.assertIn("--dropout 0.5", joined)
            self.assertIn("--ctc-enabled 1", joined)
            self.assertIn(f"--ctc-weight {CTC_WEIGHT}", joined)
            self.assertIn("--ctc-mode fixed", joined)
            self.assertIn("--ctc-k 10", joined)
            self.assertIn("--temporal-target-policy local_kmeans_ctc", joined)
            self.assertIn("--test-policy none", joined)
            self.assertIn("--split-mode fixed", joined)


if __name__ == "__main__":
    unittest.main()
