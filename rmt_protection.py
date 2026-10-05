"""Coordinate-preserving block-wise RMT for RGB segmentation inputs.

This is an adaptation inspired by DisguisedNets. It intentionally omits
block permutation so image coordinates remain aligned with unchanged masks.
"""

import hashlib

import numpy as np

from config import RMT_KEY_ID, RMT_TRANSFORM_SEED


def _stable_seed(*parts) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")


def generate_orthogonal_matrices(image_size, block_size, transform_seed=RMT_TRANSFORM_SEED):
    """Generate one fixed Haar-distributed orthogonal matrix per block slot.

    Matrices use float64 QR of independent standard-normal matrices, with the
    diagonal-sign correction that gives the Haar orthogonal distribution.
    The flattened block order is row-major: block row, then block column.
    """
    if image_size % block_size:
        raise ValueError("image_size must be divisible by block_size")
    blocks_per_side = image_size // block_size
    matrices = []
    for block_index in range(blocks_per_side * blocks_per_side):
        rng = np.random.default_rng(np.random.SeedSequence([int(transform_seed), block_index]))
        gaussian = rng.standard_normal((block_size, block_size))
        q, r = np.linalg.qr(gaussian)
        diagonal = np.diag(r)
        signs = np.where(diagonal < 0.0, -1.0, 1.0)
        matrices.append(q * signs[np.newaxis, :])
    return np.stack(matrices, axis=0)


def _noise_rng(transform_seed, sample_id, block_index, channel):
    return np.random.default_rng(
        _stable_seed(RMT_KEY_ID, transform_seed, sample_id, block_index, channel)
    )


def transform_rgb_image(
    image,
    block_size,
    noise_level,
    sample_id="",
    transform_seed=RMT_TRANSFORM_SEED,
    matrices=None,
):
    """Apply coordinate-preserving RMT to an H×W×3 float image.

    For each row-major block position i and channel c, computes
    Y[i,c] = R[i] @ (X[i,c] + Delta[i,c]). The same R[i] is reused for all
    channels, samples, and training seeds. Noise is independently generated
    per sample, block, and channel. No clipping or quantization is performed.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    height, width, _ = array.shape
    if height != width or height % block_size:
        raise ValueError("image must be square and divisible by block_size")
    if block_size not in (8, 16, 32):
        raise ValueError("block_size must be 8, 16, or 32")
    if not np.isfinite(array).all():
        raise ValueError("image contains NaN or Inf")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    blocks_per_side = height // block_size
    expected_matrices = blocks_per_side * blocks_per_side
    if matrices is None:
        matrices = generate_orthogonal_matrices(height, block_size, transform_seed)
    if matrices.shape != (expected_matrices, block_size, block_size):
        raise ValueError("matrices shape does not match image and block size")

    output = np.empty_like(array, dtype=np.float32)
    block_index = 0
    for block_row in range(blocks_per_side):
        y0 = block_row * block_size
        for block_col in range(blocks_per_side):
            x0 = block_col * block_size
            for channel in range(3):
                source = array[y0:y0 + block_size, x0:x0 + block_size, channel].astype(np.float64)
                if noise_level == 0:
                    noise = 0.0
                else:
                    noise = _noise_rng(
                        transform_seed, sample_id, block_index, channel
                    ).uniform(0.0, noise_level, size=(block_size, block_size))
                # einsum avoids a platform-specific BLAS runtime warning seen
                # for small QR-derived matrices while computing the same left
                # matrix product R @ (X + Delta).
                transformed = np.einsum(
                    "ij,jk->ik", matrices[block_index], source + noise
                )
                output[y0:y0 + block_size, x0:x0 + block_size, channel] = transformed.astype(np.float32)
            block_index += 1
    return output


def condition_metadata(condition_name, block_size, noise_level):
    return {
        "condition": condition_name,
        "method": "coordinate-preserving block-wise RMT adaptation inspired by DisguisedNets",
        "block_size": int(block_size),
        "number_of_blocks": int((256 // block_size) ** 2),
        "noise_distribution": "independent Uniform(0, N) entries per image, block, and RGB channel",
        "noise_level": float(noise_level),
        "rgb_handling": "same block-position matrix left-multiplies each channel's b×b matrix independently",
        "orthogonal_matrix_generation": "independent standard-normal b×b matrix; NumPy float64 QR; column sign correction; Haar-distributed orthogonal matrix",
        "arithmetic_dtype": "float64 for matrix/noise arithmetic; float32 transformed array output",
        "matrix_key_strategy": "SeedSequence([transform_seed, row-major block index]); matrix reused across all images and training seeds",
        "transformation_seed": int(RMT_TRANSFORM_SEED),
        "transformation_key_id": RMT_KEY_ID,
        "noise_seed_strategy": "SHA-256 key over key ID, transformation seed, original image filename, row-major block index, and channel; independent generator for each tuple",
        "normalization": "input RGB float32 scaled to [0,1] by existing preprocessing",
        "clipping": "none during RMT; deterministic clip to [0,1] only when saving existing 8-bit RGB PNG format",
        "block_permutation": False,
        "mask_transform": "none; original segmentation mask retained unchanged",
    }
