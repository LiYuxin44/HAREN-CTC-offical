"""Tests for the five-bin subject-level PHQ split protocol."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_preparation"))

from prepare_phq_stratified_train_test import (  # noqa: E402
    EXPECTED_BIN_COUNTS,
    INNER_DEV_SUBJECTS,
    PHQ_BIN_LABELS,
    add_phq_bins,
    assign_inner_roles,
    assign_outer_folds,
)


class PhqStratifiedTrainTestTest(unittest.TestCase):
    def subjects(self) -> pd.DataFrame:
        scores = {
            "0-4": (2, 86),
            "5-9": (7, 46),
            "10-14": (12, 30),
            "15-19": (17, 20),
            "20-24": (22, 7),
        }
        rows = []
        participant = 300
        for score, count in scores.values():
            for _ in range(count):
                rows.append(
                    {
                        "Participant_ID": participant,
                        "binary": int(score >= 10),
                        "phq_score": score,
                        "source_split": "official_train",
                        "severity": "unused",
                    }
                )
                participant += 1
        return pd.DataFrame(rows)

    def test_outer_and_inner_phq_counts_are_optimally_balanced(self) -> None:
        subjects = add_phq_bins(self.subjects())
        self.assertEqual(
            subjects["phq_bin"].value_counts().to_dict(),
            EXPECTED_BIN_COUNTS,
        )
        assigned = assign_outer_folds(subjects, seed=123)
        outer = pd.crosstab(
            assigned["outer_fold"], assigned["phq_bin"]
        ).reindex(columns=PHQ_BIN_LABELS, fill_value=0)
        self.assertTrue(((outer.max() - outer.min()) <= 1).all())
        self.assertEqual(set(outer.sum(axis=1)), {37, 38})

        dev_counts = []
        for fold in range(5):
            train_dev = assigned.loc[assigned["outer_fold"] != fold]
            inner = assign_inner_roles(train_dev, seed=123, fold=fold)
            dev = inner.loc[inner["inner_role"] == "dev"]
            self.assertEqual(len(dev), INNER_DEV_SUBJECTS)
            dev_counts.append(
                dev["phq_bin"]
                .value_counts()
                .reindex(PHQ_BIN_LABELS, fill_value=0)
                .tolist()
            )
        self.assertEqual(dev_counts, [[14, 8, 5, 3, 1]] * 5)


if __name__ == "__main__":
    unittest.main()
