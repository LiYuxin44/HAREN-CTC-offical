"""Optimizer, EMA, and restart helpers for stable HAREN-CTC training."""

from __future__ import annotations

import contextlib
import math
import os
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch


def adamw_parameter_groups(
    module: torch.nn.Module, weight_decay: float
) -> tuple[list[dict], dict[str, int]]:
    """Separate bias, normalization, and routing logits from decayed weights."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        exclude = (
            parameter.ndim == 1
            or name.endswith(".bias")
            or name.endswith("group_assignment")
        )
        (no_decay if exclude else decay).append(parameter)
    if not decay or not no_decay:
        raise ValueError("Expected both decayed and non-decayed parameters")
    groups = [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    counts = {
        "decay_parameters": sum(parameter.numel() for parameter in decay),
        "no_decay_parameters": sum(
            parameter.numel() for parameter in no_decay
        ),
    }
    return groups, counts


def warmup_cosine_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float,
) -> float:
    """Return linear-warmup then cosine-decay learning-rate multiplier."""
    if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
        raise ValueError("Invalid total_steps or warmup_steps")
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio must be within [0, 1]")
    current = max(0, min(int(step), int(total_steps)))
    if current < warmup_steps:
        return float(current + 1) / float(max(1, warmup_steps))
    progress = float(current - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_ratio + (1.0 - min_ratio) * cosine)


class TrainableEMA:
    """Track an exponential moving average of trainable parameters."""

    def __init__(self, module: torch.nn.Module, decay: float) -> None:
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("EMA decay must be within (0, 1)")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        if not self.shadow:
            raise ValueError("EMA requires trainable parameters")

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        current = dict(module.named_parameters())
        if set(self.shadow) - set(current):
            raise RuntimeError("EMA parameter inventory changed")
        for name, average in self.shadow.items():
            parameter = current[name]
            average.mul_(self.decay).add_(
                parameter.detach(), alpha=1.0 - self.decay
            )
        self.updates += 1

    @contextlib.contextmanager
    def average_parameters(
        self, module: torch.nn.Module
    ) -> Iterator[None]:
        current = dict(module.named_parameters())
        backup = {
            name: current[name].detach().clone() for name in self.shadow
        }
        try:
            with torch.no_grad():
                for name, average in self.shadow.items():
                    current[name].copy_(average)
            yield
        finally:
            with torch.no_grad():
                for name, value in backup.items():
                    current[name].copy_(value)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": {
                name: value.detach().cpu()
                for name, value in self.shadow.items()
            },
        }

    def load_state_dict(
        self, state: dict, module: torch.nn.Module
    ) -> None:
        if float(state["decay"]) != self.decay:
            raise RuntimeError("EMA decay mismatch")
        if set(state["shadow"]) != set(self.shadow):
            raise RuntimeError("EMA parameter inventory mismatch")
        current = dict(module.named_parameters())
        self.shadow = {
            name: value.to(
                device=current[name].device,
                dtype=current[name].dtype,
            )
            for name, value in state["shadow"].items()
        }
        self.updates = int(state["updates"])


def capture_rng_state(*, include_cuda: bool) -> dict:
    """Capture parent-process RNG state for epoch-boundary restart."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if include_cuda and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict, *, include_cuda: bool) -> None:
    """Restore a state produced by capture_rng_state."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if include_cuda and "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    """Atomically replace a torch checkpoint on the same filesystem."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
