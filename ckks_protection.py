import hashlib
from pathlib import Path
import numpy as np
import tenseal as ts

def region_mask(height, width, percent):
    if percent <= 0:
        return np.zeros((height, width), dtype=bool)
    if percent >= 100:
        return np.ones((height, width), dtype=bool)

    scale = np.sqrt(percent / 100.0)
    ph = max(1, min(height, int(round(height * scale))))
    pw = max(1, min(width, int(round(width * scale))))

    y0 = (height - ph) // 2
    x0 = (width - pw) // 2

    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y0 + ph, x0:x0 + pw] = True
    return mask

def create_ckks_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.global_scale = 2 ** 40
    return context

def ciphertext_noise(serialized_bytes, n_values):
    digest = hashlib.sha256(serialized_bytes).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32 - 1)
    rng = np.random.default_rng(seed)
    return rng.random(n_values, dtype=np.float32)

def protect_rgb_image(
    image_float,
    percent,
    context,
    save_dir=None,
    sample_id=None,
    chunk_size=4096,
):
    h, w, c = image_float.shape
    assert c == 3

    mask = region_mask(h, w, percent)

    if percent <= 0:
        return image_float.copy(), {
            "target_percent": 0.0,
            "actual_percent": 0.0,
            "protected_pixels": 0,
            "encrypted_values": 0,
            "chunks": 0,
        }

    values = image_float[mask].reshape(-1).astype(np.float64)
    replacement = np.empty(len(values), dtype=np.float32)

    ciphertext_dir = None
    if save_dir is not None:
        ciphertext_dir = Path(save_dir)
        ciphertext_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = 0

    for start in range(0, len(values), chunk_size):
        end = min(start + chunk_size, len(values))
        encrypted = ts.ckks_vector(context, values[start:end].tolist())
        serialized = encrypted.serialize()

        replacement[start:end] = ciphertext_noise(
            serialized, end - start
        )

        if ciphertext_dir is not None and sample_id is not None:
            out = ciphertext_dir / f"{sample_id}_chunk{n_chunks:04d}.ckks"
            out.write_bytes(serialized)

        n_chunks += 1

    protected = image_float.copy()
    protected[mask] = replacement.reshape(-1, 3)

    return protected, {
        "target_percent": float(percent),
        "actual_percent": float(100.0 * mask.mean()),
        "protected_pixels": int(mask.sum()),
        "encrypted_values": int(len(values)),
        "chunks": int(n_chunks),
    }
