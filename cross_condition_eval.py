import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import PROCESSED_ROOT, RUNS_ROOT, CONDITION_ORDER, CONDITIONS
from dataset import MedicalSegmentationDataset
from model import TinyUNet
from train import evaluate

def evaluate_matrix(batch_size=16, workers=0):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dice_matrix = np.zeros((5,5), dtype=np.float32)
    iou_matrix = np.zeros((5,5), dtype=np.float32)
    rows = []

    for i, train_condition in enumerate(CONDITION_ORDER):
        model = TinyUNet().to(device)
        checkpoint = torch.load(
            RUNS_ROOT/train_condition/"best_model.pt",
            map_location=device
        )
        model.load_state_dict(checkpoint["model_state"])

        for j, test_condition in enumerate(CONDITION_ORDER):
            ds = MedicalSegmentationDataset(
                PROCESSED_ROOT/test_condition/"test"
            )

            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=torch.cuda.is_available()
            )

            result = evaluate(model, loader, device)

            dice_matrix[i,j] = result["dice"]
            iou_matrix[i,j] = result["iou"]

            rows.append({
                "trained_on": train_condition,
                "tested_on": test_condition,
                "train_protection": CONDITIONS[train_condition],
                "test_protection": CONDITIONS[test_condition],
                "dice": result["dice"],
                "iou": result["iou"],
                "precision": result["precision"],
                "recall": result["recall"],
                "pred_foreground_fraction": result["pred_foreground_fraction"],
                "gt_foreground_fraction": result["gt_foreground_fraction"],
            })

            print(
                f"Train {train_condition:15s} -> "
                f"Test {test_condition:15s} "
                f"Dice={result['dice']:.4f}"
            )

    return dice_matrix, iou_matrix, rows

def heatmap(matrix, title, output_path):
    labels = ["0%","25%","50%","75%","100%"]

    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(5))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(5))
    ax.set_yticklabels(labels)

    ax.set_xlabel("Test protection")
    ax.set_ylabel("Training protection")
    ax.set_title(title)

    for i in range(5):
        for j in range(5):
            ax.text(
                j, i, f"{matrix[i,j]:.3f}",
                ha="center", va="center"
            )

    fig.colorbar(im, ax=ax, label="Score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    out_dir = RUNS_ROOT/"cross_condition"
    out_dir.mkdir(parents=True, exist_ok=True)

    dice_matrix, iou_matrix, rows = evaluate_matrix()

    labels = ["0","25","50","75","100"]

    pd.DataFrame(
        dice_matrix,
        index=labels,
        columns=labels
    ).to_csv(out_dir/"dice_matrix.csv")

    pd.DataFrame(
        iou_matrix,
        index=labels,
        columns=labels
    ).to_csv(out_dir/"iou_matrix.csv")

    pd.DataFrame(rows).to_csv(
        out_dir/"all_cross_condition_results.csv",
        index=False
    )

    heatmap(
        dice_matrix,
        "Cross-condition Dice",
        out_dir/"dice_heatmap.png"
    )

    heatmap(
        iou_matrix,
        "Cross-condition IoU",
        out_dir/"iou_heatmap.png"
    )

    print(f"\nSaved cross-condition results to: {out_dir}")

if __name__ == "__main__":
    main()
