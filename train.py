"""
train.py — Single-run training script.

Supports:
  --model   tinyunet | pbunet
  --condition  <condition name from config.CONDITIONS>
  --seed    <integer seed>

Output directory: runs/<model>/<condition>/seed_<seed>/
  best_model.pt       — best checkpoint by val Dice
  history.csv         — per-epoch metrics
  test_metrics.json   — final test results
  training_curve.png  — validation Dice/IoU plot

Resume logic:
  If test_metrics.json already exists in the run directory this script exits
  immediately, treating the run as complete.  This allows GPU interruption
  recovery: re-launch the same command and already-finished seeds are skipped.
"""

import argparse
import csv
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import PROCESSED_ROOT, RUNS_ROOT, CONDITIONS
from dataset import MedicalSegmentationDataset
from metrics import segmentation_metrics
from model import TinyUNet


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def dice_loss(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def combined_loss(logits, targets):
    """0.5 × BCE + 0.5 × Dice (as specified in experiment protocol)."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return 0.5 * bce + 0.5 * dice_loss(logits, targets)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    totals = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "pixel_accuracy": 0.0,
        "pred_foreground_fraction": 0.0,
        "gt_foreground_fraction": 0.0,
    }
    n_batches = 0

    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        logits  = model(images)
        loss    = combined_loss(logits, masks)
        metrics = segmentation_metrics(logits, masks)

        total_loss += loss.item()
        for key in totals:
            totals[key] += metrics[key]
        n_batches += 1

    result = {
        key: value / max(1, n_batches)
        for key, value in totals.items()
    }
    result["loss"] = total_loss / max(1, n_batches)
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def make_loader(path, batch_size, shuffle, workers):
    ds = MedicalSegmentationDataset(path)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return ds, loader


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_history(history, output_path):
    epochs = [x["epoch"] for x in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [x["val_dice"] for x in history], label="Validation Dice")
    ax.plot(epochs, [x["val_iou"]  for x in history], label="Validation IoU")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_name: str, device):
    """Instantiate the requested model and move it to `device`.

    For PBUNet the ownership mask is constructed internally using the
    deterministic geometric formula — not inferred from pixel values.
    """
    if model_name == "tinyunet":
        return TinyUNet().to(device)

    if model_name == "pbunet":
        from pbunet import PBUNet   # late import avoids scipy requirement when unused
        return PBUNet(
            region_percent=25,
            base=12,
            halo_width=1,
            img_size=256,
            in_channels=3,
        ).to(device)

    raise ValueError(f"Unknown model: {model_name!r}. Choose 'tinyunet' or 'pbunet'.")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_condition(
    model_name: str = "tinyunet",
    condition:  str = "no_protection",
    epochs:     int = 35,
    batch_size: int = 16,
    lr:         float = 1e-3,
    seed:       int = 42,
    workers:    int = 0,
) -> dict:
    """Train one (model, condition, seed) combination.

    Returns the test-metrics dict.  If the run directory already contains a
    completed test_metrics.json the function loads and returns it without
    re-training (interruption-safe resume).

    Output directory: runs/<model_name>/<condition>/seed_<seed>/
    """
    # ---- Run directory (seed-level) ----------------------------------------
    run_dir = RUNS_ROOT / model_name / condition / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "test_metrics.json"

    # Resume: skip if already finished
    if metrics_path.exists():
        print(
            f"[SKIP] {model_name}/{condition}/seed_{seed} — "
            f"test_metrics.json already exists."
        )
        return json.loads(metrics_path.read_text())

    # ---- Reproducibility ---------------------------------------------------
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Model:     {model_name}")
    print(f"Condition: {condition}")
    print(f"Seed:      {seed}")
    print(f"Device:    {device}")
    print(f"{'='*60}\n")

    # ---- Data loaders -------------------------------------------------------
    base_path = PROCESSED_ROOT / condition
    _, train_loader = make_loader(base_path / "train", batch_size, True,  workers)
    _, val_loader   = make_loader(base_path / "val",   batch_size, False, workers)
    _, test_loader  = make_loader(base_path / "test",  batch_size, False, workers)

    # ---- Model, optimiser, scheduler ----------------------------------------
    model = build_model(model_name, device)

    # Plain Adam, fixed lr — matches the stated experiment specification.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # ---- Training loop ------------------------------------------------------
    best_path     = run_dir / "best_model.pt"
    best_val_dice = -1.0
    history       = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"{model_name}/{condition}/seed_{seed}  epoch {epoch}/{epochs}",
        )

        for images, masks, _ in pbar:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss   = combined_loss(logits, masks)
            loss.backward()
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # No LR scheduler — lr stays fixed at the initial value.

        val = evaluate(model, val_loader, device)

        row = {
            "epoch":      epoch,
            "train_loss": running / max(1, len(train_loader)),
            "val_loss":   val["loss"],
            "val_dice":   val["dice"],
            "val_iou":    val["iou"],
            "val_precision": val["precision"],
            "val_recall":    val["recall"],
            "val_pred_fg":   val["pred_foreground_fraction"],
            "val_gt_fg":     val["gt_foreground_fraction"],
            "lr":         optimizer.param_groups[0]["lr"],   # constant throughout
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"Dice={val['dice']:.4f} | "
            f"IoU={val['iou']:.4f} | "
            f"PredFG={val['pred_foreground_fraction']:.3f} | "
            f"GTFG={val['gt_foreground_fraction']:.3f}"
        )

        if val["dice"] > best_val_dice:
            best_val_dice = val["dice"]
            torch.save(
                {
                    "model_state":   model.state_dict(),
                    "model_name":    model_name,
                    "condition":     condition,
                    "seed":          seed,
                    "epoch":         epoch,
                    "best_val_dice": best_val_dice,
                },
                best_path,
            )

    # ---- Persist training history ------------------------------------------
    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    plot_history(history, run_dir / "training_curve.png")

    # ---- Final test evaluation on best checkpoint ---------------------------
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test = evaluate(model, test_loader, device)

    result = {
        "model":         model_name,
        "condition":     condition,
        "seed":          seed,
        "protection_percent": CONDITIONS[condition],
        "best_epoch":    checkpoint["epoch"],
        "best_val_dice": checkpoint["best_val_dice"],
        "test_loss":     test["loss"],
        "test_dice":     test["dice"],
        "test_iou":      test["iou"],
        "test_precision":  test["precision"],
        "test_recall":     test["recall"],
        "test_pixel_accuracy":         test["pixel_accuracy"],
        "test_pred_foreground_fraction": test["pred_foreground_fraction"],
        "test_gt_foreground_fraction":   test["gt_foreground_fraction"],
    }

    # Persist immediately — so GPU interruption after this point loses nothing
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train one (model, condition, seed) combination."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["tinyunet", "pbunet"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=list(CONDITIONS),
        help="Protection condition (must match a key in config.CONDITIONS).",
    )
    parser.add_argument("--epochs",     type=int,   default=35)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--workers",    type=int,   default=0)
    args = parser.parse_args()

    train_condition(
        model_name=args.model,
        condition=args.condition,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
