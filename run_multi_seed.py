"""
run_multi_seed.py — Outer experiment launcher.

Runs the full grid of:
  models:      tinyunet, pbunet         (configurable via --models)
  conditions:  no_protection, protect_25, and configured RMT conditions
                                            (configurable via --conditions)
  seeds:       42, 101, 202, 303        (configurable via --seeds)

Each (model, condition, seed) combination is run sequentially.  A summary
CSV is written to runs/summary_results.csv after EVERY completed seed, so
a GPU interruption can never lose already-finished results.

Re-running this script is safe: train_condition() detects runs that already
have test_metrics.json and skips them immediately (no training, no I/O
beyond a single file-existence check).

Usage examples
--------------
# Full experiment (default settings):
python run_multi_seed.py

# Dry run — only tinyunet, seed 42:
python run_multi_seed.py --models tinyunet --seeds 42

# Custom conditions:
python run_multi_seed.py --conditions no_protection protect_25
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from config import RUNS_ROOT, CONDITIONS, RMT_CONDITIONS
from train import train_condition

# Experiment defaults
DEFAULT_MODELS     = ["tinyunet", "pbunet"]
DEFAULT_CONDITIONS = ["no_protection", "protect_25", *RMT_CONDITIONS]
DEFAULT_SEEDS      = [42, 101, 202, 303]

# Keep the completed original summary immutable. New launches merge it in
# memory and write additions to this separate summary file.
LEGACY_SUMMARY_PATH = RUNS_ROOT / "summary_results.csv"
SUMMARY_PATH = RUNS_ROOT / "summary_results_rmt.csv"

SUMMARY_FIELDS = [
    "model",
    "condition",
    "seed",
    "protection_percent",
    "best_epoch",
    "best_val_dice",
    "test_dice",
    "test_iou",
    "test_precision",
    "test_recall",
    "test_pixel_accuracy",
    "test_pred_foreground_fraction",
    "test_gt_foreground_fraction",
    "test_loss",
    "rmt_block_size",
    "rmt_noise_level",
    "rmt_transform_seed",
    "rmt_key_id",
]


def load_existing_summary() -> list:
    """Merge legacy and new rows without editing the original summary."""
    rows = []
    for path in (LEGACY_SUMMARY_PATH, SUMMARY_PATH):
        if path.exists():
            with open(path, newline="") as f:
                rows.extend(csv.DictReader(f))
    merged = {}
    for row in rows:
        merged[row_key(row)] = row
    return list(merged.values())


def save_summary(rows: list):
    """Overwrite summary_results.csv with all rows collected so far."""
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict) -> tuple:
    return (row["model"], row["condition"], str(row["seed"]))


def print_aggregate_table(rows: list):
    """Print mean ± std of test Dice and IoU across seeds, grouped by model × condition."""
    import statistics

    groups: dict = {}
    for r in rows:
        key = (r["model"], r["condition"])
        groups.setdefault(key, []).append(float(r["test_dice"]))

    print("\n" + "=" * 60)
    print(f"{'Model':<12} {'Condition':<20} {'N':>3} {'Mean Dice':>10} {'Std Dice':>10}")
    print("-" * 60)
    for (model, cond), dices in sorted(groups.items()):
        n    = len(dices)
        mean = statistics.mean(dices)
        std  = statistics.stdev(dices) if n > 1 else 0.0
        print(f"{model:<12} {cond:<20} {n:>3} {mean:>10.4f} {std:>10.4f}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed experiment launcher for TinyUNet and PBUNet."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=["tinyunet", "pbunet"],
        help="Model(s) to train (default: tinyunet pbunet).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        choices=[*CONDITIONS, *RMT_CONDITIONS],
        help="Condition(s); RMT conditions are scheduled for TinyUNet only.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to run (default: 42 101 202 303).",
    )
    parser.add_argument("--epochs",     type=int,   default=35)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--workers",    type=int,   default=0)
    args = parser.parse_args()

    # Build the full grid
    if any(condition in RMT_CONDITIONS for condition in args.conditions) and "tinyunet" not in args.models:
        parser.error("RMT conditions require TinyUNet in --models.")

    grid = []
    for condition in args.conditions:
        eligible_models = (
            [model for model in args.models if model == "tinyunet"]
            if condition in RMT_CONDITIONS
            else args.models
        )
        grid.extend(
            (model, condition, seed)
            for model in eligible_models
            for seed in args.seeds
        )

    if not grid:
        parser.error("No eligible model/condition combinations were selected.")

    total = len(grid)
    print(f"\nExperiment grid: {total} runs")
    print(f"  Models:     {args.models}")
    print(f"  Conditions: {args.conditions}")
    print(f"  Seeds:      {args.seeds}")
    print(f"  Epochs:     {args.epochs}, BS: {args.batch_size}, LR: {args.lr}\n")

    # Load any previously completed rows so we can append
    completed_rows = load_existing_summary()
    completed_keys = {row_key(r) for r in completed_rows}

    for i, (model, condition, seed) in enumerate(grid, 1):
        key = (model, condition, str(seed))
        status = "SKIP" if key in completed_keys else "RUN "
        print(f"[{i:02d}/{total:02d}] {status}  {model:<12} {condition:<20} seed={seed}")

    print()

    # Run / resume
    all_rows = list(completed_rows)   # start from existing data

    for i, (model, condition, seed) in enumerate(grid, 1):
        key = (model, condition, str(seed))

        if key in completed_keys:
            # Already done: metrics were loaded above; nothing to run
            print(f"\n[{i:02d}/{total:02d}] Skipping {model}/{condition}/seed_{seed} (already complete)")
            continue

        print(f"\n[{i:02d}/{total:02d}] Starting {model}/{condition}/seed_{seed}")

        result = train_condition(
            model_name=model,
            condition=condition,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=seed,
            workers=args.workers,
        )

        # Persist immediately after this seed finishes
        all_rows.append(result)
        completed_keys.add(key)
        save_summary(all_rows)
        print(f"\n  → Saved to {SUMMARY_PATH}  ({len(all_rows)}/{total} rows)")

    # Final aggregate statistics
    print_aggregate_table(all_rows)
    print(f"All results saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
