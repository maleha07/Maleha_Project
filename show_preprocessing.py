import argparse
import matplotlib.pyplot as plt
from PIL import Image
from config import PROCESSED_ROOT, DISPLAY_ORDER

LABELS = {
    "no_protection": "No Protection",
    "protect_25": "25% Protection",
    "protect_50": "50% Protection",
    "protect_75": "75% Protection",
    "protect_100": "100% Protection",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--split", default="test", choices=["train","val","test"])
    parser.add_argument("--save", default="preprocessing_grid.png")
    args = parser.parse_args()

    name = f"{args.sample_index:04d}.png"

    fig, axes = plt.subplots(2, 5, figsize=(18,7))

    for col, condition in enumerate(DISPLAY_ORDER):
        image = Image.open(
            PROCESSED_ROOT/condition/args.split/"images"/name
        ).convert("RGB")
        mask = Image.open(
            PROCESSED_ROOT/condition/args.split/"masks"/name
        ).convert("L")

        axes[0,col].imshow(image)
        axes[0,col].set_title(LABELS[condition])
        axes[0,col].axis("off")

        axes[1,col].imshow(mask, cmap="gray")
        axes[1,col].axis("off")

    axes[0,0].set_ylabel("RGB input")
    axes[1,0].set_ylabel("Binary mask")

    fig.suptitle("Kvasir-SEG CKKS spatial protection", fontsize=16)
    fig.tight_layout()
    fig.savefig(args.save, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
