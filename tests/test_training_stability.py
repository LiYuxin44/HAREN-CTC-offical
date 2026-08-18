"""Unit tests for optimizer, EMA, and restart helpers."""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
import copy
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training_stability import (  # noqa: E402
    TrainableEMA,
    adamw_parameter_groups,
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
    warmup_cosine_multiplier,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.norm = torch.nn.LayerNorm(2)
        self.group_assignment = torch.nn.Parameter(torch.zeros(4, 2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(value)).square().mean()


class TrainingStabilityTest(unittest.TestCase):
    def test_adamw_groups_exclude_bias_norm_and_routing(self) -> None:
        model = TinyModel()
        groups, counts = adamw_parameter_groups(model, 0.01)
        self.assertEqual(groups[0]["weight_decay"], 0.01)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertEqual(
            counts["decay_parameters"] + counts["no_decay_parameters"],
            sum(parameter.numel() for parameter in model.parameters()),
        )
        self.assertGreater(counts["decay_parameters"], 0)
        self.assertGreater(counts["no_decay_parameters"], 0)

    def test_warmup_cosine_schedule_has_expected_endpoints(self) -> None:
        values = [
            warmup_cosine_multiplier(
                step,
                total_steps=100,
                warmup_steps=10,
                min_ratio=0.1,
            )
            for step in range(101)
        ]
        self.assertAlmostEqual(values[0], 0.1)
        self.assertAlmostEqual(values[9], 1.0)
        self.assertAlmostEqual(values[10], 1.0)
        self.assertAlmostEqual(values[-1], 0.1)
        self.assertTrue(
            all(left >= right for left, right in zip(values[10:], values[11:]))
        )

    def test_ema_context_swaps_and_restores_parameters(self) -> None:
        model = TinyModel()
        ema = TrainableEMA(model, decay=0.5)
        original = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(2.0)
        current = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        ema.update(model)
        with ema.average_parameters(model):
            for name, parameter in model.named_parameters():
                self.assertTrue(
                    torch.allclose(parameter, original[name] + 1.0)
                )
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, current[name]))

        restored = TrainableEMA(model, decay=0.5)
        restored.load_state_dict(ema.state_dict(), model)
        self.assertEqual(restored.updates, 1)
        for name in ema.shadow:
            self.assertTrue(torch.equal(ema.shadow[name], restored.shadow[name]))

    def test_rng_round_trip_replays_all_cpu_generators(self) -> None:
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        state = capture_rng_state(include_cuda=False)
        expected = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
        )
        restore_rng_state(state, include_cuda=False)
        actual = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
        )
        self.assertEqual(expected, actual)

    def test_atomic_torch_save_replaces_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            atomic_torch_save({"value": 1}, path)
            atomic_torch_save({"value": 2}, path)
            self.assertEqual(
                torch.load(path, weights_only=False)["value"], 2
            )
            self.assertFalse(path.with_name("state.pt.tmp").exists())

    def test_epoch_boundary_restart_matches_uninterrupted_training(self) -> None:
        torch.manual_seed(42)
        initial = TinyModel().state_dict()

        def setup():
            model = TinyModel()
            model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: warmup_cosine_multiplier(
                    step,
                    total_steps=4,
                    warmup_steps=1,
                    min_ratio=0.1,
                ),
            )
            ema = TrainableEMA(model, decay=0.9)
            return model, optimizer, scheduler, ema

        def train_steps(model, optimizer, scheduler, ema, count):
            for _ in range(count):
                optimizer.zero_grad(set_to_none=True)
                loss = model(torch.randn(5, 3))
                loss.backward()
                optimizer.step()
                scheduler.step()
                ema.update(model)

        uninterrupted = setup()
        torch.manual_seed(100)
        train_steps(*uninterrupted, 4)

        split = setup()
        torch.manual_seed(100)
        train_steps(*split, 2)
        state = {
            "model": copy.deepcopy(split[0].state_dict()),
            "optimizer": copy.deepcopy(split[1].state_dict()),
            "scheduler": copy.deepcopy(split[2].state_dict()),
            "ema": copy.deepcopy(split[3].state_dict()),
            "rng": capture_rng_state(include_cuda=False),
        }
        resumed = setup()
        resumed[0].load_state_dict(state["model"])
        resumed[1].load_state_dict(state["optimizer"])
        resumed[2].load_state_dict(state["scheduler"])
        resumed[3].load_state_dict(state["ema"], resumed[0])
        restore_rng_state(state["rng"], include_cuda=False)
        train_steps(*resumed, 2)

        for expected, actual in zip(
            uninterrupted[0].parameters(), resumed[0].parameters()
        ):
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(
            uninterrupted[2].state_dict(), resumed[2].state_dict()
        )
        for name in uninterrupted[3].shadow:
            self.assertTrue(
                torch.equal(
                    uninterrupted[3].shadow[name],
                    resumed[3].shadow[name],
                )
            )


if __name__ == "__main__":
    unittest.main()
