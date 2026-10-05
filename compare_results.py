import json
import matplotlib.pyplot as plt
import pandas as pd
import torch

from config import PROCESSED_ROOT, RUNS_ROOT, CONDITIONS, DISPLAY_ORDER
from dataset import MedicalSegmentationDataset
from model import TinyUNet

LABELS = {
    "no_protection": "No Protection",
    "protect_25": "25% Protection",
    "protect_50": "50% Protection",
    "protect_75": "75% Protection",
    "protect_100": "100% Protection",
}

def load_results():
    rows = []
    for condition in CONDITIONS:
        path = RUNS_ROOT/condition/"test_metrics.json"
        if not path.exists():
            raise RuntimeError(f"Missing result: {path}")
        rows.append(json.loads(path.read_text()))
    return pd.DataFrame(rows)

def metric_curve(df, out_dir):
    df = df.sort_values("protection_percent")
    fig, ax = plt.subplots(figsize=(9,5))

    for key, label in [
        ("test_dice","Dice"),
        ("test_iou","IoU"),
        ("test_precision","Precision"),
        ("test_recall","Recall")
    ]:
        ax.plot(
            df["protection_percent"],
            df[key],
            marker="o",
            label=label
        )

    ax.set_xlabel("Protected image area (%)")
    ax.set_ylabel("Score")
    ax.set_xticks([0,25,50,75,100])
    ax.set_ylim(0,1)
    ax.set_title("TinyUNet performance vs CKKS protection")
    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir/"metrics_vs_protection.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def dice_drop_plot(df, out_dir):
    df = df.sort_values("protection_percent").copy()

    baseline = float(
        df.loc[
            df["protection_percent"] == 0,
            "test_dice"
        ].iloc[0]
    )

    df["dice_drop"] = baseline - df["test_dice"]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(
        df["protection_percent"].astype(str) + "%",
        df["dice_drop"]
    )
    ax.set_xlabel("Protected image area")
    ax.set_ylabel("Absolute Dice decrease vs no protection")
    ax.set_title("Segmentation utility loss")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir/"dice_drop.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

@torch.no_grad()
def prediction_comparison(out_dir, sample_index=0):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    fig, axes = plt.subplots(3,5,figsize=(18,10))

    for col, condition in enumerate(DISPLAY_ORDER):
        ds = MedicalSegmentationDataset(
            PROCESSED_ROOT/condition/"test"
        )

        image, mask, _ = ds[sample_index]

        model = TinyUNet().to(device)
        checkpoint = torch.load(
            RUNS_ROOT/condition/"best_model.pt",
            map_location=device
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        logits = model(image.unsqueeze(0).to(device))
        pred = (
            torch.sigmoid(logits)[0,0] >= 0.5
        ).float().cpu().numpy()

        axes[0,col].imshow(image.permute(1,2,0).numpy())
        axes[0,col].set_title(LABELS[condition])

        axes[1,col].imshow(mask[0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[2,col].imshow(pred, cmap="gray", vmin=0, vmax=1)

        for row in range(3):
            axes[row,col].axis("off")

    axes[0,0].set_ylabel("Input")
    axes[1,0].set_ylabel("Ground Truth")
    axes[2,0].set_ylabel("Prediction")

    fig.suptitle("Same Kvasir-SEG test case across five protection trials", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_dir/"prediction_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def protection_inputs(out_dir, sample_index=0):
    fig, axes = plt.subplots(1,5,figsize=(18,4))

    for ax, condition in zip(axes, DISPLAY_ORDER):
        ds = MedicalSegmentationDataset(
            PROCESSED_ROOT/condition/"test"
        )
        image, _, _ = ds[sample_index]

        ax.imshow(image.permute(1,2,0).numpy())
        ax.set_title(LABELS[condition])
        ax.axis("off")

    fig.suptitle("Same RGB medical image at each protection level", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir/"protection_inputs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    out_dir = RUNS_ROOT/"comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results().sort_values("protection_percent")
    df.to_csv(out_dir/"results.csv", index=False)

    metric_curve(df, out_dir)
    dice_drop_plot(df, out_dir)
    protection_inputs(out_dir)
    prediction_comparison(out_dir)

    print("\nMatched-condition results:")
    print(df.to_string(index=False))
    print(f"\nSaved figures to: {out_dir}")

if __name__ == "__main__":
    main()
