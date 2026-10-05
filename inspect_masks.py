import argparse
import numpy as np
from PIL import Image
from config import PROCESSED_ROOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="no_protection")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    mask_dir = PROCESSED_ROOT/args.condition/args.split/"masks"
    ratios = []

    for path in sorted(mask_dir.glob("*.png")):
        mask = np.asarray(
            Image.open(path).convert("L")
        ) >= 128
        ratios.append(mask.mean())

    ratios = np.asarray(ratios)

    print(f"N masks: {len(ratios)}")
    print(f"Mean foreground fraction: {ratios.mean():.4f}")
    print(f"Std foreground fraction:  {ratios.std():.4f}")
    print(f"Min foreground fraction:  {ratios.min():.4f}")
    print(f"Max foreground fraction:  {ratios.max():.4f}")

if __name__ == "__main__":
    main()
