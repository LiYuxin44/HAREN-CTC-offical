"""Tests for packed fold-specific HuBERT unit caches."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from global_unit_cache import (  # noqa: E402
    PackedUnitCache,
    save_packed_unit_cache,
)


class GlobalUnitCacheTest(unittest.TestCase):
    def test_round_trip_and_crop_lookup(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fold0_k10.npz"
            save_packed_unit_cache(
                path,
                {
                    "300_0001": np.arange(700) % 10,
                    "301_0002": np.arange(20) % 10,
                },
                metadata={
                    "k": 10,
                    "fold": 0,
                    "codebook_fit_split": "train_only",
                },
            )
            cache = PackedUnitCache(path, expected_k=10)
            self.assertEqual(len(cache), 2)
            cropped = cache.crop_sequence(
                "/audio/300_0001.wav",
                crop_start_samples=3200,
                frame_length=5,
            )
            self.assertEqual(cropped.tolist(), [0, 1, 2, 3, 4])
            aligned = cache.crop_sequence(
                "/audio/300_0001.wav",
                crop_start_samples=3200,
                frame_length=5,
                require_aligned=True,
            )
            self.assertEqual(aligned.tolist(), cropped.tolist())
            with self.assertRaises(ValueError):
                cache.crop_sequence(
                    "/audio/300_0001.wav",
                    crop_start_samples=3201,
                    frame_length=5,
                    require_aligned=True,
                )
            self.assertEqual(
                cache.metadata["codebook_fit_split"], "train_only"
            )

    def test_cache_rejects_wrong_k_and_short_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "units.npz"
            save_packed_unit_cache(
                path,
                {"300_0001": [0, 1, 2]},
                metadata={"k": 10, "fold": 0},
            )
            with self.assertRaises(ValueError):
                PackedUnitCache(path, expected_k=50)
            cache = PackedUnitCache(path, expected_k=10)
            with self.assertRaises(ValueError):
                cache.crop_sequence(
                    "300_0001.wav",
                    crop_start_samples=0,
                    frame_length=4,
                )


if __name__ == "__main__":
    unittest.main()
