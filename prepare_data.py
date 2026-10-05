import argparse
import csv
import json
import random
import shutil
import platform
import hashlib
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import PROCESSED_ROOT, CONDITIONS, RMT_CONDITIONS, RMT_TRANSFORM_SEED
from download_kvasir import download_kvasir
from ckks_protection import create_ckks_context, protect_rgb_image
from rmt_protection import transform_rgb_image, generate_orthogonal_matrices, condition_metadata

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def collect_pairs(dataset_root):
    image_dir = dataset_root / "images"
    mask_dir = dataset_root / "masks"

    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in EXTS
    ])

    pairs = []
    for image_path in image_paths:
        candidates = [
            mask_dir / image_path.name,
            mask_dir / f"{image_path.stem}.jpg",
            mask_dir / f"{image_path.stem}.jpeg",
            mask_dir / f"{image_path.stem}.png",
        ]
        mask_path = next((p for p in candidates if p.exists()), None)
        if mask_path is None:
            raise RuntimeError(f"No mask for {image_path.name}")
        pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError("No image/mask pairs found.")

    return pairs

def make_split(pairs, seed=42):
    pairs = pairs.copy()
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(round(0.70 * n))
    n_val = int(round(0.15 * n))

    return {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

def resize_pair(image_path, mask_path, image_size):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image = image.resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR
    )
    mask = mask.resize(
        (image_size, image_size),
        Image.Resampling.NEAREST
    )

    image_np = np.asarray(image, dtype=np.float32) / 255.0

    mask_np = np.asarray(mask, dtype=np.uint8)
    mask_np = (mask_np >= 128).astype(np.uint8) * 255

    return image_np, mask_np

def save_condition(
    condition_name,
    percent,
    split_name,
    pairs,
    image_size,
    context,
    save_ciphertext,
):
    out_root = PROCESSED_ROOT / condition_name / split_name
    image_dir = out_root / "images"
    mask_dir = out_root / "masks"
    cipher_dir = out_root / "ciphertext"

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    if save_ciphertext and percent > 0:
        cipher_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for i, (image_path, mask_path) in enumerate(
        tqdm(pairs, desc=f"{condition_name}/{split_name}")
    ):
        image_np, mask_np = resize_pair(
            image_path, mask_path, image_size
        )

        sample_id = f"{i:04d}"

        if percent == 0:
            protected = image_np
            meta = {
                "target_percent": 0.0,
                "actual_percent": 0.0,
                "protected_pixels": 0,
                "encrypted_values": 0,
                "chunks": 0,
            }
        else:
            protected, meta = protect_rgb_image(
                image_np,
                percent,
                context,
                save_dir=cipher_dir if save_ciphertext else None,
                sample_id=sample_id,
            )

        Image.fromarray(
            np.clip(protected * 255.0, 0, 255).astype(np.uint8),
            mode="RGB"
        ).save(image_dir / f"{sample_id}.png")

        Image.fromarray(
            mask_np,
            mode="L"
        ).save(mask_dir / f"{sample_id}.png")

        rows.append({
            "sample_id": sample_id,
            "original_image": image_path.name,
            "original_mask": mask_path.name,
            "foreground_fraction": float((mask_np > 0).mean()),
            **meta,
        })

    with open(out_root / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_rmt_condition(condition_name, settings, splits, image_size):
    """Materialize one RMT condition without touching existing conditions."""
    condition_root = PROCESSED_ROOT / condition_name
    if condition_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing RMT data: {condition_root}"
        )

    block_size = settings["block_size"]
    noise_level = settings["noise_level"]
    matrices = generate_orthogonal_matrices(
        image_size, block_size, RMT_TRANSFORM_SEED
    )
    matrix_path = condition_root / "rmt_matrices.npy"
    matrix_bytes = matrices.astype(np.float64, copy=False).tobytes(order="C")
    matrix_sha256 = hashlib.sha256(matrix_bytes).hexdigest()
    metadata = condition_metadata(condition_name, block_size, noise_level)
    metadata.update({
        "image_size": int(image_size),
        "split_seed": 42,
        "split_sizes": {name: len(pairs) for name, pairs in splits.items()},
        "image_interpolation": "bilinear",
        "mask_interpolation": "nearest-neighbor, then threshold at 128",
        "sample_identity_for_noise": "original image filename",
        "matrix_count": int(len(matrices)),
        "matrix_values_saved": True,
        "matrix_file": "rmt_matrices.npy",
        "matrix_sha256": matrix_sha256,
        "matrix_sha256_scope": "raw row-major float64 matrix array bytes before NPY container serialization",
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "reproduction_note": "The exact float64 matrices are saved as well as their deterministic generation procedure; image-specific noise is regenerated from the recorded SHA-256 seed strategy.",
    })

    condition_root.mkdir(parents=True, exist_ok=False)
    np.save(matrix_path, matrices.astype(np.float64, copy=False), allow_pickle=False)
    (condition_root / "rmt_metadata.json").write_text(json.dumps(metadata, indent=2))

    for split_name in ("train", "val", "test"):
        out_root = condition_root / split_name
        image_dir = out_root / "images"
        mask_dir = out_root / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir()
        rows = []

        for i, (image_path, mask_path) in enumerate(
            tqdm(splits[split_name], desc=f"{condition_name}/{split_name}")
        ):
            image_np, mask_np = resize_pair(image_path, mask_path, image_size)
            transformed = transform_rgb_image(
                image_np,
                block_size=block_size,
                noise_level=noise_level,
                sample_id=image_path.name,
                transform_seed=RMT_TRANSFORM_SEED,
                matrices=matrices,
            )
            sample_id = f"{i:04d}"
            # Existing processed data are 8-bit PNGs; this is the sole clipping
            # point and matches the current clean/CKKS serialization convention.
            stored = np.clip(transformed, 0.0, 1.0)
            Image.fromarray((stored * 255.0).astype(np.uint8), mode="RGB").save(
                image_dir / f"{sample_id}.png"
            )
            Image.fromarray(mask_np, mode="L").save(mask_dir / f"{sample_id}.png")
            rows.append({
                "sample_id": sample_id,
                "original_image": image_path.name,
                "original_mask": mask_path.name,
                "foreground_fraction": float((mask_np > 0).mean()),
                "condition": condition_name,
                "block_size": block_size,
                "noise_level": noise_level,
                "transformation_seed": RMT_TRANSFORM_SEED,
            })

        with open(out_root / "manifest.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-ciphertext", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--redownload", action="store_true")
    parser.add_argument(
        "--rmt-only",
        action="store_true",
        help="Create only the approved RMT datasets; refuse to overwrite existing condition folders.",
    )
    args = parser.parse_args()

    if args.rmt_only and args.overwrite:
        parser.error("--overwrite cannot be combined with --rmt-only; existing experiment data must be preserved.")
    if args.rmt_only and args.redownload:
        parser.error("--redownload cannot be combined with --rmt-only.")
    if args.rmt_only and (args.image_size != 256 or args.seed != 42):
        parser.error("The approved RMT preprocessing requires --image-size 256 and --seed 42.")

    dataset_root = download_kvasir(overwrite=args.redownload)
    pairs = collect_pairs(dataset_root)
    splits = make_split(pairs, seed=args.seed)

    print(f"Found {len(pairs)} pairs.")
    print(
        f"Train={len(splits['train'])} "
        f"Val={len(splits['val'])} "
        f"Test={len(splits['test'])}"
    )

    if args.overwrite and PROCESSED_ROOT.exists():
        shutil.rmtree(PROCESSED_ROOT)

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    if args.rmt_only:
        existing = [name for name in RMT_CONDITIONS if (PROCESSED_ROOT / name).exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing RMT condition(s): " + ", ".join(existing)
            )
        for condition_name, settings in RMT_CONDITIONS.items():
            print(f"\n===== {condition_name} =====")
            save_rmt_condition(condition_name, settings, splits, args.image_size)
        print(f"\nSaved RMT processed datasets to: {PROCESSED_ROOT}")
        return

    context = create_ckks_context()

    for condition_name, percent in CONDITIONS.items():
        print(f"\n===== {condition_name} ({percent}%) =====")
        for split_name in ["train", "val", "test"]:
            save_condition(
                condition_name,
                percent,
                split_name,
                splits[split_name],
                args.image_size,
                context,
                args.save_ciphertext,
            )

    print(f"\nSaved processed datasets to: {PROCESSED_ROOT}")

if __name__ == "__main__":
    main()
