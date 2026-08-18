"""Packed, provenance-aware global HuBERT unit caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = 2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identifier_set_sha256(identifiers: Iterable[str]) -> str:
    """Hash an identifier set with a stable, order-independent encoding."""
    canonical = "\n".join(sorted({str(value) for value in identifiers})) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def utterance_id(path: str | Path) -> str:
    return Path(path).stem


def save_packed_unit_cache(
    path: Path,
    sequences: Mapping[str, Sequence[int] | np.ndarray],
    *,
    metadata: Mapping[str, object],
) -> None:
    """Save variable-length unit sequences without pickle/object arrays."""
    if not sequences:
        raise ValueError("Cannot save an empty unit cache")
    identifiers = sorted(str(identifier) for identifier in sequences)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate utterance identifiers")
    offsets = [0]
    packed = []
    for identifier in identifiers:
        sequence = np.asarray(sequences[identifier], dtype=np.int64).reshape(-1)
        if sequence.size == 0 or np.any(sequence < 0):
            raise ValueError(f"Invalid unit sequence for {identifier}")
        packed.append(sequence.astype(np.uint16, copy=False))
        offsets.append(offsets[-1] + int(sequence.size))
    payload_metadata = {
        **dict(metadata),
        "schema_version": SCHEMA_VERSION,
        "utterances": len(identifiers),
        "token_count": int(offsets[-1]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            identifiers=np.asarray(identifiers, dtype=np.str_),
            offsets=np.asarray(offsets, dtype=np.int64),
            tokens=np.concatenate(packed).astype(np.uint16, copy=False),
            metadata_json=np.asarray(
                json.dumps(payload_metadata, sort_keys=True), dtype=np.str_
            ),
        )
    temporary.replace(path)


class PackedUnitCache:
    """Read-only lookup for full-utterance units sliced to a waveform crop."""

    def __init__(self, path: str | Path, *, expected_k: int | None = None):
        self.path = Path(path).resolve()
        with np.load(self.path, allow_pickle=False) as payload:
            identifiers = payload["identifiers"].astype(str).tolist()
            self.offsets = payload["offsets"].astype(np.int64, copy=True)
            self.tokens = payload["tokens"].astype(np.int64, copy=True)
            self.metadata = json.loads(str(payload["metadata_json"].item()))
        if int(self.metadata.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported unit-cache schema: {self.path}")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"Duplicate unit-cache identifiers: {self.path}")
        if self.offsets.shape != (len(identifiers) + 1,):
            raise ValueError(f"Invalid unit-cache offsets: {self.path}")
        if (
            self.offsets[0] != 0
            or self.offsets[-1] != len(self.tokens)
            or np.any(np.diff(self.offsets) <= 0)
        ):
            raise ValueError(f"Corrupt unit-cache packing: {self.path}")
        if expected_k is not None and int(self.metadata.get("k", -1)) != int(
            expected_k
        ):
            raise ValueError(
                f"Unit-cache K={self.metadata.get('k')} != {expected_k}"
            )
        self._indices = {
            identifier: index for index, identifier in enumerate(identifiers)
        }
        self.sha256 = file_sha256(self.path)

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def identifiers(self) -> frozenset[str]:
        return frozenset(self._indices)

    def full_sequence(self, path: str | Path) -> np.ndarray:
        identifier = utterance_id(path)
        if identifier not in self._indices:
            raise KeyError(f"Missing global units for {identifier}")
        index = self._indices[identifier]
        return self.tokens[
            self.offsets[index] : self.offsets[index + 1]
        ].copy()

    def crop_sequence(
        self,
        path: str | Path,
        *,
        crop_start_samples: int,
        frame_length: int,
        frame_stride_samples: int = 320,
        require_aligned: bool = False,
    ) -> torch.Tensor:
        """Slice full-audio units to the matching 10-second crop."""
        if (
            crop_start_samples < 0
            or frame_length <= 0
            or frame_stride_samples <= 0
        ):
            raise ValueError("Crop start/frame length must be valid")
        if require_aligned and (
            int(crop_start_samples) % int(frame_stride_samples) != 0
        ):
            raise ValueError(
                "Global-unit crop start is not aligned to the frame grid"
            )
        sequence = self.full_sequence(path)
        start_frame = int(crop_start_samples) // int(frame_stride_samples)
        stop_frame = start_frame + int(frame_length)
        cropped = sequence[start_frame:stop_frame]
        if len(cropped) != int(frame_length):
            raise ValueError(
                f"Unit cache too short for {utterance_id(path)}: "
                f"need [{start_frame}:{stop_frame}], have {len(sequence)}"
            )
        return torch.from_numpy(cropped.astype(np.int64, copy=False))
