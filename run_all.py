import argparse
import subprocess
import sys

from config import CONDITION_ORDER
from train import train_condition
from compare_results import main as compare_main
from cross_condition_eval import main as cross_main

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--save-ciphertext", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-cross-eval", action="store_true")

    args = parser.parse_args()

    if not args.skip_preprocess:
        cmd = [
            sys.executable,
            "prepare_data.py",
            "--image-size", str(args.image_size),
            "--seed", str(args.seed),
        ]

        if args.save_ciphertext:
            cmd.append("--save-ciphertext")
        if args.overwrite:
            cmd.append("--overwrite")

        subprocess.check_call(cmd)

    for condition in CONDITION_ORDER:
        train_condition(
            condition=condition,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            workers=args.workers,
        )

    compare_main()

    if not args.skip_cross_eval:
        cross_main()

if __name__ == "__main__":
    main()
