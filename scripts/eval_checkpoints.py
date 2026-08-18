#!/usr/bin/env python3
"""Evaluate frozen HAREN checkpoints once on the official fixed-split test set."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


REQUIRED_COLUMNS = {
    "variant",
    "seed",
    "checkpoint_path",
    "data_root",
    "learning_rate",
    "batch_size",
    "weight_decay",
    "dropout",
    "ctc_enabled",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-index", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--variant")
    parser.add_argument("--ctc-enabled", type=int, choices=(0, 1))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Checkpoint index missing columns: {sorted(missing)}")
        return list(reader)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.checkpoint_index)
    if args.variant:
        rows = [row for row in rows if row["variant"] == args.variant]
    if args.ctc_enabled is not None:
        rows = [
            row
            for row in rows
            if int(row["ctc_enabled"]) == args.ctc_enabled
        ]
    if not rows:
        raise ValueError("No checkpoints matched the requested filters")

    runner = Path(__file__).with_name("run_experiments.py")
    seen = set()
    for row in rows:
        key = (row["variant"], int(row["ctc_enabled"]), int(row["seed"]))
        if key in seen:
            raise ValueError(f"Duplicate checkpoint row: {key}")
        seen.add(key)

        checkpoint = Path(row["checkpoint_path"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        ctc_tag = "ctc_on" if int(row["ctc_enabled"]) else "ctc_off"
        output_dir = (
            args.output_root
            / row["variant"]
            / ctc_tag
            / f"seed{int(row['seed'])}"
        )
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite {output_dir}")

        command = [
            args.python,
            str(runner),
            "--python",
            args.python,
            "--data-root",
            row["data_root"],
            "--seeds",
            str(int(row["seed"])),
            "--epochs",
            "15",
            "--batch-size",
            str(int(row["batch_size"])),
            "--lr",
            row["learning_rate"],
            "--weight-decay",
            row["weight_decay"],
            "--dropout",
            row["dropout"],
            "--ctc-enabled",
            str(int(row["ctc_enabled"])),
            "--test-policy",
            "final_only",
            "--eval-checkpoint",
            str(checkpoint),
            "--split-mode",
            "fixed",
            "--workers",
            str(args.workers),
            "--prefetch-factor",
            str(args.prefetch_factor),
            "--run-tag",
            f"final_test_{row['variant']}_{ctc_tag}",
            "--output-dir",
            str(output_dir),
        ]
        print("[eval] " + " ".join(command), flush=True)
        subprocess.run(command, check=True)

    print(f"Evaluated {len(rows)} frozen checkpoints exactly once.")


if __name__ == "__main__":
    main()
