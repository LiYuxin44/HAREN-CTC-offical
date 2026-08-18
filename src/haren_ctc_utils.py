"""Pure tensor helpers for HAREN-CTC masking and loss control."""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


def trainable_state_hash(module: torch.nn.Module) -> str:
    """Hash trainable parameters for paired-initialization provenance."""
    return named_parameter_hash(
        (
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        )
    )


def named_parameter_hash(
    named_parameters,
) -> str:
    """Hash an explicitly selected named-parameter sequence."""
    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def crop_waveform(
    waveform: torch.Tensor, max_samples: int, start: int = 0
) -> torch.Tensor:
    """Crop without padding; short waveforms preserve their true length."""
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [channels, samples]")
    if max_samples <= 0 or start < 0:
        raise ValueError("max_samples must be positive and start nonnegative")
    total = waveform.size(-1)
    if total <= max_samples:
        return waveform
    if start + max_samples > total:
        raise ValueError("crop exceeds waveform bounds")
    return waveform[:, start : start + max_samples]


def crop_and_pad_waveform(
    waveform: torch.Tensor, max_samples: int, start: int = 0
) -> tuple[torch.Tensor, int]:
    """Crop to a fixed WavLM window, then zero-pad while retaining true length."""
    cropped = crop_waveform(waveform, max_samples, start)
    valid_samples = int(cropped.size(-1))
    if valid_samples < max_samples:
        cropped = torch.nn.functional.pad(
            cropped, (0, max_samples - valid_samples)
        )
    return cropped, valid_samples


def deterministic_crop_start(
    identifier: str,
    total_samples: int,
    segment_samples: int,
    *,
    seed: int,
    epoch: int,
    draw: int,
    alignment: int = 1,
) -> int:
    """Select a train crop independently of DataLoader worker scheduling."""
    if total_samples < 0 or segment_samples <= 0 or int(alignment) <= 0:
        raise ValueError("Invalid waveform/segment length")
    if epoch <= 0 or draw < 0:
        raise ValueError("epoch must be positive and draw nonnegative")
    overflow = max(0, int(total_samples) - int(segment_samples))
    if overflow == 0:
        return 0
    digest = hashlib.sha256(
        (
            f"{int(seed)}\0{int(epoch)}\0{int(draw)}\0"
            f"{str(identifier)}"
        ).encode("utf-8")
    ).digest()
    aligned_choices = overflow // int(alignment) + 1
    choice = int.from_bytes(digest[:8], byteorder="big") % aligned_choices
    return int(choice) * int(alignment)


class EpochSeededWeightedSampler(Sampler[tuple[int, int, int]]):
    """Yield weighted indices plus deterministic epoch/draw crop keys."""

    def __init__(
        self,
        weights: Sequence[float] | torch.Tensor,
        *,
        num_samples: int,
        seed: int,
    ) -> None:
        self.weights = torch.as_tensor(
            weights, dtype=torch.double, device="cpu"
        )
        if (
            self.weights.ndim != 1
            or self.weights.numel() == 0
            or not torch.isfinite(self.weights).all()
            or torch.any(self.weights < 0)
            or float(self.weights.sum().item()) <= 0
        ):
            raise ValueError("weights must be a finite, nonnegative 1-D vector")
        if int(num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) <= 0:
            raise ValueError("epoch must be positive")
        self.epoch = int(epoch)

    def __iter__(self):
        seed_material = hashlib.sha256(
            f"{self.seed}\0{self.epoch}".encode("ascii")
        ).digest()
        epoch_seed = int.from_bytes(
            seed_material[:8], byteorder="big"
        ) % (2**63 - 1)
        generator = torch.Generator().manual_seed(epoch_seed)
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(
            (int(index), int(self.epoch), int(draw))
            for draw, index in enumerate(indices)
        )

    def __len__(self) -> int:
        return self.num_samples


def prepare_wavlm_inputs(
    waves: Sequence[torch.Tensor],
    valid_samples: Sequence[int],
    extractor,
    *,
    preprocess_policy: str,
    mask_policy: str,
    padding_policy: str,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply WavLM normalization/padding in an explicit, auditable order."""
    if preprocess_policy not in {"legacy_prepad", "valid_then_pad"}:
        raise ValueError(
            f"Unsupported WavLM preprocess policy: {preprocess_policy}"
        )
    if mask_policy not in {"legacy_full", "true_length"}:
        raise ValueError(f"Unsupported WavLM mask policy: {mask_policy}")
    if padding_policy not in {"fixed_10s", "longest"}:
        raise ValueError(
            f"Unsupported WavLM batch padding policy: {padding_policy}"
        )
    if max_samples <= 0 or len(waves) != len(valid_samples) or not waves:
        raise ValueError("Invalid wave batch, lengths, or max_samples")
    if preprocess_policy == "legacy_prepad" and padding_policy != "fixed_10s":
        raise ValueError("legacy_prepad requires fixed_10s padding")

    trimmed: list[torch.Tensor] = []
    lengths: list[int] = []
    for wave, requested_length in zip(waves, valid_samples):
        if wave.ndim != 1:
            raise ValueError("Each waveform must be one-dimensional")
        length = int(requested_length)
        if length <= 0 or length > wave.numel() or length > max_samples:
            raise ValueError("A valid waveform length is out of bounds")
        trimmed.append(wave[:length].detach().cpu())
        lengths.append(length)

    if preprocess_policy == "legacy_prepad":
        processor_waves = [
            F.pad(wave, (0, max_samples - wave.numel())).numpy()
            for wave in trimmed
        ]
        padding = True
        extra_kwargs = {}
    else:
        processor_waves = [wave.numpy() for wave in trimmed]
        padding = "max_length" if padding_policy == "fixed_10s" else True
        extra_kwargs = (
            {"max_length": max_samples, "truncation": True}
            if padding_policy == "fixed_10s"
            else {}
        )

    outputs = extractor(
        processor_waves,
        sampling_rate=16000,
        return_tensors="pt",
        padding=padding,
        return_attention_mask=True,
        **extra_kwargs,
    )
    input_values = outputs.input_values
    processor_mask = outputs.attention_mask.to(dtype=torch.long)
    true_length_mask = lengths_to_attention_mask(
        torch.tensor(lengths, dtype=torch.long), input_values.size(1)
    )
    expected_processor_mask = (
        torch.ones_like(processor_mask)
        if preprocess_policy == "legacy_prepad"
        else true_length_mask
    )
    if not torch.equal(processor_mask, expected_processor_mask):
        raise RuntimeError("Feature extractor returned an unexpected mask")
    attention_mask = (
        torch.ones_like(processor_mask)
        if mask_policy == "legacy_full"
        else true_length_mask
    )
    return input_values, attention_mask


def evaluation_crop_start(
    total_samples: int,
    segment_samples: int,
    policy: str,
    *,
    alignment: int = 1,
) -> int:
    """Return deterministic head/center/tail start for evaluation."""
    if total_samples < 0 or segment_samples <= 0 or int(alignment) <= 0:
        raise ValueError("Invalid waveform/segment length")
    if policy not in {"head", "center", "tail"}:
        raise ValueError(f"Unsupported evaluation crop policy: {policy}")
    overflow = max(0, int(total_samples) - int(segment_samples))
    requested = 0
    if policy == "center":
        requested = overflow // 2
    elif policy == "tail":
        requested = overflow
    return (requested // int(alignment)) * int(alignment)


def multi_view_crop_starts(
    total_samples: int,
    segment_samples: int,
    *,
    alignment: int = 1,
) -> list[int]:
    """Return unique aligned head/center/tail evaluation starts."""
    return sorted(
        {
            evaluation_crop_start(
                total_samples,
                segment_samples,
                policy,
                alignment=alignment,
            )
            for policy in ("head", "center", "tail")
        }
    )


def sliding_window_starts(
    total_samples: int, segment_samples: int, stride_samples: int
) -> list[int]:
    """Cover a waveform with fixed windows and an end-aligned final window."""
    if (
        total_samples < 0
        or segment_samples <= 0
        or stride_samples <= 0
        or stride_samples > segment_samples
    ):
        raise ValueError("Invalid waveform, segment, or sliding-window stride")
    overflow = max(0, int(total_samples) - int(segment_samples))
    if overflow == 0:
        return [0]
    starts = list(range(0, overflow + 1, int(stride_samples)))
    if starts[-1] != overflow:
        starts.append(overflow)
    return starts


def collapse_sliding_window_predictions(
    paths: Sequence[str],
    subjects: Sequence[str],
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> tuple[list[str], list[str], list[int], list[float]]:
    """Average windows per utterance before subject-level aggregation."""
    if not (
        len(paths)
        == len(subjects)
        == len(labels)
        == len(probabilities)
        > 0
    ):
        raise ValueError("Sliding prediction fields must have equal length")
    grouped: dict[str, dict[str, object]] = {}
    for path, subject, label, probability in zip(
        paths, subjects, labels, probabilities
    ):
        stem, extension = os.path.splitext(str(path))
        if "__window" not in stem:
            raise ValueError(f"Missing sliding-window suffix: {path}")
        original_stem, window_index = stem.rsplit("__window", 1)
        if not window_index.isdigit() or not math.isfinite(float(probability)):
            raise ValueError(f"Invalid sliding-window prediction: {path}")
        original_path = original_stem + extension
        record = grouped.setdefault(
            original_path,
            {
                "subject": str(subject),
                "label": int(label),
                "probability_sum": 0.0,
                "windows": 0,
            },
        )
        if (
            record["subject"] != str(subject)
            or record["label"] != int(label)
        ):
            raise ValueError("Sliding windows disagree on subject or label")
        record["probability_sum"] = float(record["probability_sum"]) + float(
            probability
        )
        record["windows"] = int(record["windows"]) + 1
    return (
        list(grouped),
        [str(record["subject"]) for record in grouped.values()],
        [int(record["label"]) for record in grouped.values()],
        [
            float(record["probability_sum"]) / int(record["windows"])
            for record in grouped.values()
        ],
    )


def lengths_to_attention_mask(
    valid_lengths: torch.Tensor, max_length: int
) -> torch.Tensor:
    """Create an exact sample-level attention mask from waveform lengths."""
    lengths = valid_lengths.to(dtype=torch.long)
    if lengths.ndim != 1 or torch.any(lengths < 0):
        raise ValueError("valid_lengths must be a nonnegative 1-D tensor")
    if torch.any(lengths > int(max_length)):
        raise ValueError("A valid length exceeds max_length")
    steps = torch.arange(int(max_length), device=lengths.device)
    return (steps.unsqueeze(0) < lengths.unsqueeze(1)).to(dtype=torch.long)


def feature_output_lengths(
    input_lengths: torch.Tensor,
    conv_kernel: Sequence[int],
    conv_stride: Sequence[int],
) -> torch.Tensor:
    """Compute wav2vec2-family convolutional output lengths."""
    lengths = input_lengths.to(dtype=torch.long)
    for kernel, stride in zip(conv_kernel, conv_stride):
        lengths = torch.div(
            lengths - int(kernel), int(stride), rounding_mode="floor"
        ) + 1
    return lengths.clamp_min(0)


def masked_kmeans_batched(
    features: torch.Tensor,
    valid_lengths: torch.Tensor,
    n_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Run independent batched k-means while excluding padded frames."""
    if features.ndim != 3:
        raise ValueError("features must have shape [batch, time, hidden]")
    batch_size, time_steps, hidden_size = features.shape
    lengths = valid_lengths.to(device=features.device, dtype=torch.long)
    if lengths.shape != (batch_size,):
        raise ValueError("valid_lengths must have shape [batch]")
    if torch.any(lengths <= 0) or torch.any(lengths > time_steps):
        raise ValueError("valid_lengths must be within [1, time_steps]")

    clusters = min(int(n_clusters), int(lengths.min().item()))
    if clusters <= 0:
        raise ValueError("n_clusters must be positive")

    fractions = torch.linspace(
        0.0, 1.0, steps=clusters, device=features.device
    ).unsqueeze(0)
    indices = (fractions * (lengths - 1).unsqueeze(1)).round().long()
    gather_index = indices.unsqueeze(-1).expand(-1, -1, hidden_size)
    centroids = features.gather(1, gather_index).clone()
    valid_mask = (
        torch.arange(time_steps, device=features.device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )

    for _ in range(max_iter):
        labels = torch.argmin(torch.cdist(features, centroids), dim=-1)
        labels = labels.masked_fill(~valid_mask, 0)
        expanded = labels.unsqueeze(-1).expand(-1, -1, hidden_size)
        weighted_features = features * valid_mask.unsqueeze(-1)
        sums = torch.zeros_like(centroids)
        sums.scatter_add_(1, expanded, weighted_features)
        counts = torch.zeros(
            (batch_size, clusters),
            dtype=features.dtype,
            device=features.device,
        )
        counts.scatter_add_(1, labels, valid_mask.to(features.dtype))
        new_centroids = torch.where(
            counts.unsqueeze(-1) > 0,
            sums / counts.clamp_min(1).unsqueeze(-1),
            centroids,
        )
        shift = torch.norm(new_centroids - centroids, dim=-1).mean(dim=-1)
        centroids = new_centroids
        if torch.all(shift < tol):
            break

    labels = torch.argmin(torch.cdist(features, centroids), dim=-1)
    return labels.masked_fill(~valid_mask, 0)


def build_ctc_targets(
    cluster_ids: torch.Tensor,
    valid_lengths: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    target_mode: str,
    unit_stride: int = 1,
    reverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse repeated clusters and optionally use class-disjoint tokens."""
    if target_mode not in {"label_shifted", "neutral"}:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    if int(unit_stride) <= 0:
        raise ValueError("unit_stride must be positive")
    lengths = valid_lengths.to(device=cluster_ids.device, dtype=torch.long)
    labels = labels.to(device=cluster_ids.device, dtype=torch.long)
    sequences: list[torch.Tensor] = []
    target_lengths: list[int] = []
    for index in range(cluster_ids.size(0)):
        sequence = cluster_ids[
            index, : int(lengths[index].item()) : int(unit_stride)
        ]
        changes = torch.cat(
            [
                torch.ones(1, dtype=torch.bool, device=cluster_ids.device),
                sequence[1:] != sequence[:-1],
            ]
        )
        sequence = sequence[changes].to(dtype=torch.long)
        if reverse:
            sequence = sequence.flip(0)
        if target_mode == "label_shifted" and int(labels[index].item()) == 1:
            sequence = sequence + int(k)
        sequences.append(sequence)
        target_lengths.append(int(sequence.numel()))
    flat_targets = torch.cat(sequences) if sequences else torch.empty(
        0, dtype=torch.long, device=cluster_ids.device
    )
    return flat_targets, torch.tensor(
        target_lengths, dtype=torch.long, device=cluster_ids.device
    )


def temporal_logit_view(
    logits: torch.Tensor,
    *,
    k: int,
    target_mode: str,
    framewise: bool,
) -> tuple[torch.Tensor, int | None]:
    """Remove impossible classes for neutral or framewise temporal losses."""
    if logits.ndim != 3 or logits.size(-1) not in {
        int(k) + 1,
        2 * int(k) + 1,
    }:
        raise ValueError("Temporal logits do not match K+1 or 2K+1 head")
    if target_mode not in {"neutral", "label_shifted"}:
        raise ValueError(f"Unsupported temporal target mode: {target_mode}")
    if logits.size(-1) == int(k) + 1:
        if target_mode != "neutral":
            raise ValueError("K+1 temporal head only supports neutral targets")
        if framewise:
            return logits[:, :, : int(k)], None
        return logits, int(k)
    if framewise:
        classes = int(k) if target_mode == "neutral" else 2 * int(k)
        return logits[:, :, :classes], None
    if target_mode == "neutral":
        return torch.cat((logits[:, :, :k], logits[:, :, -1:]), dim=-1), int(k)
    return logits, 2 * int(k)


def normalized_ctc_loss(
    logits: torch.Tensor,
    flat_targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    blank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute auditable FP32 CTC, normalized per target token and utterance."""
    if logits.ndim != 3:
        raise ValueError("CTC logits must have shape [batch, time, classes]")
    batch_size, time_steps, classes = logits.shape
    inputs = input_lengths.to(device=logits.device, dtype=torch.long)
    targets = target_lengths.to(device=logits.device, dtype=torch.long)
    flat_targets = flat_targets.to(device=logits.device, dtype=torch.long)
    if (
        inputs.shape != (batch_size,)
        or targets.shape != (batch_size,)
        or torch.any(inputs <= 0)
        or torch.any(inputs > time_steps)
        or torch.any(targets <= 0)
        or torch.any(targets > inputs)
        or int(flat_targets.numel()) != int(targets.sum().item())
        or not 0 <= int(blank) < classes
    ):
        raise ValueError("Invalid CTC logits, targets, or lengths")
    with torch.amp.autocast(device_type=logits.device.type, enabled=False):
        log_probs = F.log_softmax(
            logits.float(), dim=-1
        ).transpose(0, 1).contiguous()
        per_example = F.ctc_loss(
            log_probs,
            flat_targets,
            inputs,
            targets,
            blank=int(blank),
            reduction="none",
            zero_infinity=False,
        )
        normalized = per_example / targets.to(dtype=torch.float32)
        loss = normalized.mean()
    if not torch.isfinite(per_example).all() or not torch.isfinite(loss):
        raise RuntimeError("Non-finite FP32 CTC loss")
    return loss, per_example


def normalized_frame_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average frame CE within each utterance, then across utterances."""
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("Frame logits/targets must have [batch, time] alignment")
    batch_size, time_steps, classes = logits.shape
    lengths = valid_lengths.to(device=logits.device, dtype=torch.long)
    if (
        lengths.shape != (batch_size,)
        or torch.any(lengths <= 0)
        or torch.any(lengths > time_steps)
    ):
        raise ValueError("Invalid framewise valid lengths")
    with torch.amp.autocast(device_type=logits.device.type, enabled=False):
        losses = F.cross_entropy(
            logits.float().reshape(-1, classes),
            targets.to(device=logits.device, dtype=torch.long).reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape(batch_size, time_steps)
        mask = (
            torch.arange(time_steps, device=logits.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        if torch.any(targets.to(logits.device)[mask] < 0):
            raise ValueError("A valid frame target is ignored")
        per_example = (losses * mask).sum(dim=1) / lengths.float()
        loss = per_example.mean()
    if not torch.isfinite(per_example).all() or not torch.isfinite(loss):
        raise RuntimeError("Non-finite FP32 frame CE loss")
    return loss, per_example


def ctc_greedy_edit_counts(
    logits: torch.Tensor,
    flat_targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    blank: int,
) -> dict[str, int]:
    """Return greedy CTC edit counts without hiding insertion errors."""
    if logits.ndim != 3:
        raise ValueError("CTC logits must have shape [batch, time, classes]")
    batch_size, time_steps, classes = logits.shape
    inputs = input_lengths.detach().cpu().to(torch.long)
    lengths = target_lengths.detach().cpu().to(torch.long)
    targets = flat_targets.detach().cpu().to(torch.long)
    if (
        inputs.shape != (batch_size,)
        or lengths.shape != (batch_size,)
        or torch.any(inputs <= 0)
        or torch.any(inputs > time_steps)
        or torch.any(lengths <= 0)
        or int(lengths.sum().item()) != int(targets.numel())
        or not 0 <= int(blank) < classes
    ):
        raise ValueError("Invalid greedy CTC inputs")

    def edit_distance(left: list[int], right: list[int]) -> int:
        previous = list(range(len(right) + 1))
        for left_index, left_value in enumerate(left, start=1):
            current = [left_index]
            for right_index, right_value in enumerate(right, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[right_index] + 1,
                        previous[right_index - 1]
                        + int(left_value != right_value),
                    )
                )
            previous = current
        return previous[-1]

    predictions = logits.detach().float().argmax(dim=-1).cpu()
    target_offset = 0
    edits = 0
    tokens = 0
    exact = 0
    predicted_tokens = 0
    for index in range(batch_size):
        raw = predictions[index, : int(inputs[index])].tolist()
        collapsed = [
            value
            for position, value in enumerate(raw)
            if position == 0 or value != raw[position - 1]
        ]
        decoded = [value for value in collapsed if value != int(blank)]
        target_length = int(lengths[index])
        target = targets[
            target_offset : target_offset + target_length
        ].tolist()
        target_offset += target_length
        distance = edit_distance(decoded, target)
        edits += distance
        tokens += target_length
        predicted_tokens += len(decoded)
        exact += int(distance == 0)
    return {
        "edit_distance": int(edits),
        "target_tokens": int(tokens),
        "predicted_tokens": int(predicted_tokens),
        "exact_sequences": int(exact),
        "sequences": int(batch_size),
    }


def effective_ctc_weight(
    *,
    cls_loss: torch.Tensor,
    ctc_loss: torch.Tensor,
    mode: str,
    max_weight: float,
    warmup_factor: float,
    target_ratio: float,
) -> float:
    """Return a detached scalar CTC multiplier for this optimization step."""
    factor = min(max(float(warmup_factor), 0.0), 1.0)
    if mode == "fixed":
        return float(max_weight) * factor
    if mode != "adaptive_ratio":
        raise ValueError(f"Unsupported CTC mode: {mode}")
    if ctc_loss.detach().item() <= 0:
        return 0.0
    ratio_cap = (
        float(target_ratio)
        * factor
        * float(cls_loss.detach().item())
        / float(ctc_loss.detach().item())
    )
    return min(float(max_weight) * factor, ratio_cap)


class SharedGradientRatioController:
    """Bound auxiliary influence using shared-gradient and loss ratios."""

    def __init__(
        self,
        *,
        target_ratio: float,
        max_weight: float,
        loss_ratio_cap: float,
        warmup_steps: int,
        update_interval: int = 10,
        ema_decay: float = 0.9,
    ) -> None:
        if (
            target_ratio <= 0
            or max_weight <= 0
            or loss_ratio_cap <= 0
            or warmup_steps < 0
            or update_interval <= 0
            or not 0 <= ema_decay < 1
        ):
            raise ValueError("Invalid shared-gradient controller configuration")
        self.target_ratio = float(target_ratio)
        self.max_weight = float(max_weight)
        self.loss_ratio_cap = float(loss_ratio_cap)
        self.warmup_steps = int(warmup_steps)
        self.update_interval = int(update_interval)
        self.ema_decay = float(ema_decay)
        self.ema_weight: float | None = None
        self.last_diagnostics: dict[str, float] | None = None
        self.last_ema_update_step: int | None = None

    def weight(
        self,
        *,
        primary_loss: torch.Tensor,
        auxiliary_loss: torch.Tensor,
        shared_parameters: Sequence[torch.nn.Parameter],
        step_index: int,
    ) -> tuple[float, dict[str, float]]:
        if int(step_index) < 0 or auxiliary_loss.detach().item() <= 0:
            raise ValueError("Gradient balancing requires a positive loss and step")
        # The current-batch norms are always measured: reusing an older norm
        # would make the purported instantaneous safety cap stale. The update
        # interval controls only how often the smoothed proposal is refreshed.
        diagnostics = loss_gradient_diagnostics(
            primary_loss, auxiliary_loss, shared_parameters
        )
        primary_norm = diagnostics["primary_gradient_norm"]
        auxiliary_norm = diagnostics["auxiliary_gradient_norm"]
        if primary_norm <= 0 or auxiliary_norm <= 0:
            raise RuntimeError("Shared primary/auxiliary gradient is zero")
        due = (
            self.ema_weight is None
            or int(step_index) % self.update_interval == 0
        )
        proposed = self.target_ratio * primary_norm / auxiliary_norm
        if due:
            self.ema_weight = (
                proposed
                if self.ema_weight is None
                else self.ema_decay * self.ema_weight
                + (1.0 - self.ema_decay) * proposed
            )
            self.last_ema_update_step = int(step_index)
        self.last_diagnostics = diagnostics
        if (
            self.ema_weight is None
            or self.last_diagnostics is None
            or self.last_ema_update_step is None
        ):
            raise RuntimeError("Gradient controller failed to initialize")
        warmup = (
            1.0
            if self.warmup_steps == 0
            else min(1.0, (int(step_index) + 1) / self.warmup_steps)
        )
        loss_cap = (
            self.loss_ratio_cap
            * warmup
            * float(primary_loss.detach().item())
            / float(auxiliary_loss.detach().item())
        )
        instantaneous_gradient_cap = (
            self.target_ratio
            * primary_norm
            / auxiliary_norm
        )
        # EMA smooths increases but must never let a historically large
        # weight overshoot the current batch's shared-gradient budget.
        weight = min(
            self.max_weight * warmup,
            self.ema_weight * warmup,
            instantaneous_gradient_cap * warmup,
            loss_cap,
        )
        diagnostics = {
            **self.last_diagnostics,
            "gradient_balance_updated": float(due),
            "gradient_balance_measurement_age_steps": 0.0,
            "gradient_balance_ema_age_steps": float(
                int(step_index) - self.last_ema_update_step
            ),
            "gradient_balance_warmup": float(warmup),
            "gradient_balance_ema_weight": float(self.ema_weight),
            "gradient_balance_loss_cap": float(loss_cap),
            "gradient_balance_instantaneous_cap": float(
                instantaneous_gradient_cap
            ),
            "effective_auxiliary_weight": float(weight),
            "weighted_auxiliary_to_primary_gradient_ratio": float(
                weight
                * auxiliary_norm
                / max(primary_norm, 1e-12)
            ),
        }
        return float(weight), diagnostics

    def state_dict(self) -> dict[str, object]:
        return {
            "ema_weight": self.ema_weight,
            "last_diagnostics": self.last_diagnostics,
            "last_ema_update_step": self.last_ema_update_step,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.ema_weight = (
            None
            if state.get("ema_weight") is None
            else float(state["ema_weight"])
        )
        diagnostics = state.get("last_diagnostics")
        self.last_diagnostics = (
            None
            if diagnostics is None
            else {str(key): float(value) for key, value in diagnostics.items()}
        )
        self.last_ema_update_step = (
            None
            if state.get("last_ema_update_step") is None
            else int(state["last_ema_update_step"])
        )


def routing_initial_logits(
    num_layers: int,
    *,
    alpha: float = 0.95,
    policy: str = "probability_correct",
    min_probability: float = 0.05,
) -> torch.Tensor:
    """Build two-group routing logits under legacy or probability-correct init."""
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be within (0, 1)")
    if not 0.0 < min_probability < 0.5:
        raise ValueError("min_probability must be within (0, 0.5)")
    if policy == "legacy":
        probabilities = alpha ** torch.arange(num_layers, dtype=torch.float64)
        probabilities = probabilities.clamp(1e-6, 1.0 - 1e-6)
        odds = torch.log(probabilities / (1.0 - probabilities))
        return torch.stack((odds, -odds), dim=-1).to(dtype=torch.float32)
    if policy != "probability_correct":
        raise ValueError(f"Unsupported routing initialization policy: {policy}")
    # Start at alpha rather than 1.0 so the first layer remains trainable.
    probabilities = alpha ** torch.arange(
        1, num_layers + 1, dtype=torch.float64
    )
    probabilities = probabilities.clamp(
        min_probability, 1.0 - min_probability
    )
    # softmax(log(p), log(1-p)) is exactly (p, 1-p).
    return torch.stack(
        (torch.log(probabilities), torch.log1p(-probabilities)), dim=-1
    ).to(dtype=torch.float32)


def subject_balanced_sample_weights(
    labels: Sequence[int], subjects: Sequence[str]
) -> torch.Tensor:
    """Give every class and every subject within a class equal expected mass."""
    if len(labels) != len(subjects) or not labels:
        raise ValueError("labels and subjects must be nonempty and equally sized")
    normalized_labels = [int(label) for label in labels]
    normalized_subjects = [str(subject) for subject in subjects]
    subject_labels: dict[str, int] = {}
    utterance_counts = Counter(normalized_subjects)
    for label, subject in zip(normalized_labels, normalized_subjects):
        prior = subject_labels.setdefault(subject, label)
        if prior != label:
            raise ValueError(f"Subject {subject} has inconsistent labels")
    subjects_per_class = Counter(subject_labels.values())
    if len(subjects_per_class) < 2:
        raise ValueError("subject-balanced sampling requires at least two classes")
    values = [
        1.0
        / (
            float(subjects_per_class[label])
            * float(utterance_counts[subject])
        )
        for label, subject in zip(normalized_labels, normalized_subjects)
    ]
    return torch.tensor(values, dtype=torch.double)


def aggregate_subject_predictions(
    probabilities: Sequence[float],
    subjects: Sequence[str],
    labels: Sequence[int],
    *,
    policy: str = "mean_probability",
) -> tuple[list[float], list[int], list[str]]:
    """Aggregate utterance probabilities without mixing subject identities."""
    if not (
        len(probabilities) == len(subjects) == len(labels)
        and len(probabilities) > 0
    ):
        raise ValueError("probabilities, subjects, and labels must align")
    supported = {
        "mean_probability",
        "confidence_weighted_probability",
        "confidence_weighted_vote",
    }
    if policy not in supported:
        raise ValueError(f"Unsupported aggregation policy: {policy}")
    grouped: dict[str, dict[str, object]] = {}
    for probability, subject, label in zip(probabilities, subjects, labels):
        value = float(probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid probability: {value}")
        subject = str(subject)
        label = int(label)
        if subject not in grouped:
            grouped[subject] = {"probabilities": [], "label": label}
        elif int(grouped[subject]["label"]) != label:
            raise ValueError(f"Subject {subject} has inconsistent labels")
        grouped[subject]["probabilities"].append(value)

    def subject_key(value: str) -> tuple[int, int | str]:
        return (0, int(value)) if value.isdigit() else (1, value)

    scores: list[float] = []
    subject_labels: list[int] = []
    ordered_subjects = sorted(grouped, key=subject_key)
    for subject in ordered_subjects:
        values = torch.tensor(
            grouped[subject]["probabilities"], dtype=torch.float64
        )
        if policy == "mean_probability":
            score = values.mean()
        else:
            weights = (values - 0.5).abs().clamp_min(1e-6)
            payload = values
            if policy == "confidence_weighted_vote":
                payload = (values >= 0.5).to(dtype=values.dtype)
            score = (weights * payload).sum() / weights.sum()
        scores.append(float(score.item()))
        subject_labels.append(int(grouped[subject]["label"]))
    return scores, subject_labels, ordered_subjects


def loss_gradient_diagnostics(
    primary_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> dict[str, float]:
    """Measure primary/auxiliary gradient norms and cosine on shared parameters."""
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable shared parameters supplied")
    primary_gradients = torch.autograd.grad(
        primary_loss,
        trainable,
        retain_graph=True,
        allow_unused=True,
    )
    auxiliary_gradients = torch.autograd.grad(
        auxiliary_loss,
        trainable,
        retain_graph=True,
        allow_unused=True,
    )
    primary_sq = torch.zeros((), device=primary_loss.device, dtype=torch.float32)
    auxiliary_sq = torch.zeros_like(primary_sq)
    dot = torch.zeros_like(primary_sq)
    common = 0
    for primary, auxiliary in zip(primary_gradients, auxiliary_gradients):
        if primary is not None:
            primary_float = primary.detach().float()
            primary_sq = primary_sq + primary_float.square().sum()
        if auxiliary is not None:
            auxiliary_float = auxiliary.detach().float()
            auxiliary_sq = auxiliary_sq + auxiliary_float.square().sum()
        if primary is not None and auxiliary is not None:
            dot = dot + (
                primary.detach().float() * auxiliary.detach().float()
            ).sum()
            common += 1
    primary_norm = primary_sq.sqrt()
    auxiliary_norm = auxiliary_sq.sqrt()
    denominator = (primary_norm * auxiliary_norm).clamp_min(1e-12)
    return {
        "primary_gradient_norm": float(primary_norm.item()),
        "auxiliary_gradient_norm": float(auxiliary_norm.item()),
        "gradient_cosine": float((dot / denominator).item()),
        "shared_gradient_tensors": float(common),
    }


def waveform_quality_metrics(
    waveform: torch.Tensor,
    *,
    frame_samples: int = 320,
    speech_threshold_dbfs: float = -40.0,
) -> dict[str, float]:
    """Return RMS and the fraction of 20-ms frames above an energy threshold."""
    values = waveform.detach().float().reshape(-1)
    if values.numel() == 0 or frame_samples <= 0:
        raise ValueError("waveform and frame_samples must be nonempty/positive")
    overall_rms = values.square().mean().sqrt()
    complete = values.numel() // int(frame_samples)
    frame_rms_parts = []
    if complete:
        full_frames = values[: complete * int(frame_samples)].reshape(
            complete, int(frame_samples)
        )
        frame_rms_parts.append(full_frames.square().mean(dim=1).sqrt())
    if complete * int(frame_samples) < values.numel():
        tail = values[complete * int(frame_samples) :]
        frame_rms_parts.append(tail.square().mean().sqrt().reshape(1))
    frame_rms = torch.cat(frame_rms_parts)
    threshold = 10.0 ** (float(speech_threshold_dbfs) / 20.0)
    speech_fraction = (frame_rms >= threshold).float().mean()
    return {
        "rms": float(overall_rms.item()),
        "speech_frame_fraction": float(speech_fraction.item()),
    }
