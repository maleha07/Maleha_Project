"""
PBUNet — Privacy-Branched U-Net
Adapted for the Maleha_Project:
  - Input: RGB (3 channels), 256×256
  - Protection: centred 128×128 region (25% of image area, exact)
  - Ownership mask is constructed from ckks_protection.region_mask —
    NEVER inferred from pixel values.

Architecture (unchanged from original):
  - Branch P ("plaintext"): receives x with the protected region zeroed out.
  - Branch C ("ciphertext"): receives x with the unprotected region zeroed out.
    For protect_25 inputs the protected region contains CKKS-derived noise.
  - Encoders share no weights and have zero cross-branch interaction.
  - ONE cross-branch 1×1 fusion at the bottleneck (32×32), blended only
    inside a halo band straddling the ownership boundary.
  - Decoders use branch-local skip connections only.
  - Final logit at each pixel is taken from whichever branch owns that pixel
    (C owns the protected region; P owns the rest).

Parameter count (base=12, img_size=256, in_channels=3):
  - Per-branch encoder:  165,636  (enc1 differs from 1-ch original by +216)
  - Per-branch decoder:  106,273
  - Fusion (shared):      37,056
  - TOTAL:               580,874

Verified against implementation plan (2026-10-01).
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_dilation

from model import DoubleConv          # same building block as TinyUNet; NOT reimplemented
from ckks_protection import region_mask as _make_region_mask


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def _downsample_ownership(region_full: np.ndarray, target_size: int) -> np.ndarray:
    """Downsample a (H, W) bool ownership mask to (target_size, target_size)
    using 8×8 block-majority vote.  H and W must both be divisible by
    target_size (guaranteed when H == W == img_size and target_size == img_size//8).

    Returns a (target_size, target_size) bool array.
    """
    h, w = region_full.shape
    assert h % target_size == 0 and w % target_size == 0, (
        f"region_full shape {region_full.shape} must divide evenly "
        f"into ({target_size}, {target_size})"
    )
    bh = h // target_size
    bw = w // target_size
    # Reshape exploits C-order (row-major) so [r, c] → [r//bh, r%bh, c//bw, c%bw]
    blocks = region_full.astype(np.float32).reshape(
        target_size, bh, target_size, bw
    )
    means = blocks.mean(axis=(1, 3))          # (target_size, target_size)
    return means >= 0.5


def _halo_band(mask_at_res: np.ndarray, halo_width: int) -> np.ndarray:
    """Boolean band of `halo_width` bottleneck pixels straddling the
    ownership boundary.  halo_width=0 returns all-False."""
    if halo_width <= 0:
        return np.zeros_like(mask_at_res, dtype=bool)
    struct = np.ones((3, 3), dtype=bool)
    dil_in  = binary_dilation(mask_at_res,  structure=struct, iterations=halo_width)
    dil_out = binary_dilation(~mask_at_res, structure=struct, iterations=halo_width)
    return dil_in & dil_out


# ---------------------------------------------------------------------------
# PBUNet
# ---------------------------------------------------------------------------

class PBUNet(nn.Module):
    """Privacy-Branched U-Net for Kvasir-SEG with CKKS protection.

    Args:
        region_percent (int): Protection percentage used to build the
            geometric ownership mask via region_mask().  Default 25.
        base (int): Base channel width.  Default 12 (matches original).
        halo_width (int): Width (in bottleneck pixels) of the fusion halo.
            Default 1.  Each bottleneck pixel covers 8×8 full-res pixels,
            so halo_width=1 corresponds to an ~8-pixel boundary band.
        img_size (int): Spatial side length of the input image.  Default 256.
        in_channels (int): Number of input channels.  Default 3 (RGB).
    """

    def __init__(
        self,
        region_percent: int = 25,
        base: int = 12,
        halo_width: int = 1,
        img_size: int = 256,
        in_channels: int = 3,
    ):
        super().__init__()
        self.img_size    = img_size
        self.halo_width  = halo_width
        self.base        = base

        # ---- Build deterministic ownership mask from geometry (NOT from pixels) ----
        region_full_np = _make_region_mask(img_size, img_size, region_percent)
        # region_full_np: (img_size, img_size) bool, True = protected (C-branch)

        # Bottleneck ownership (img_size // 8 × img_size // 8)
        bn_size = img_size // 8                          # 256//8 = 32
        ownership_bn_np = _downsample_ownership(region_full_np, bn_size)
        halo_bn_np      = _halo_band(ownership_bn_np, halo_width)

        # Register as non-learnable buffers so they move to the correct device
        # automatically with .to(device) and are included in state_dict.
        self.register_buffer("region_full", torch.from_numpy(region_full_np))
        self.register_buffer("halo_16",     torch.from_numpy(halo_bn_np))
        # Note: "halo_16" name is kept for compatibility with the original
        # architecture's state_dict keys.  At 256×256 this buffer is (32×32).

        # ---- Branch P ("plaintext") encoder ----
        self.p_enc1       = DoubleConv(in_channels, base)
        self.p_enc2       = DoubleConv(base,        base * 2)
        self.p_enc3       = DoubleConv(base * 2,    base * 4)
        self.p_bottleneck = DoubleConv(base * 4,    base * 8)

        # ---- Branch C ("ciphertext") encoder ----
        self.c_enc1       = DoubleConv(in_channels, base)
        self.c_enc2       = DoubleConv(base,        base * 2)
        self.c_enc3       = DoubleConv(base * 2,    base * 4)
        self.c_bottleneck = DoubleConv(base * 4,    base * 8)

        self.pool = nn.MaxPool2d(2)

        # ---- The ONE cross-branch operation: 1×1 conv at the bottleneck ----
        # Applied inside the halo band only; outside the halo each branch's
        # bottleneck features remain untouched.
        self.fuse = nn.Conv2d(base * 8 * 2, base * 8 * 2, 1)

        # ---- Branch P decoder (branch-local skip connections only) ----
        self.p_up3  = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.p_dec3 = DoubleConv(base * 8, base * 4)
        self.p_up2  = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.p_dec2 = DoubleConv(base * 4, base * 2)
        self.p_up1  = nn.ConvTranspose2d(base * 2, base,     2, stride=2)
        self.p_dec1 = DoubleConv(base * 2, base)
        self.p_out  = nn.Conv2d(base, 1, 1)

        # ---- Branch C decoder (branch-local skip connections only) ----
        self.c_up3  = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c_dec3 = DoubleConv(base * 8, base * 4)
        self.c_up2  = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c_dec2 = DoubleConv(base * 4, base * 2)
        self.c_up1  = nn.ConvTranspose2d(base * 2, base,     2, stride=2)
        self.c_dec1 = DoubleConv(base * 2, base)
        self.c_out  = nn.Conv2d(base, 1, 1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) float tensor — the pre-processed image for the
               current condition (e.g. protect_25, which has CKKS-noise in
               the protected region).

        Returns:
            logits: (B, 1, H, W) — each pixel's logit from its owning branch.
        """
        # ---- Split into P (plaintext periphery) and C (protected centre) ----
        # Region mask is the deterministic geometric mask, never inferred from x.
        region = self.region_full.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        x_p = x * (1.0 - region)   # real pixels in periphery, 0 in centre
        x_c = x * region           # CKKS-noise values in centre, 0 outside

        # ---- Branch P encoder ----
        p_e1 = self.p_enc1(x_p)                    # (B, base,    H,   W  )
        p_e2 = self.p_enc2(self.pool(p_e1))        # (B, base*2,  H/2, W/2)
        p_e3 = self.p_enc3(self.pool(p_e2))        # (B, base*4,  H/4, W/4)
        p_bn = self.p_bottleneck(self.pool(p_e3))  # (B, base*8,  H/8, W/8)

        # ---- Branch C encoder ----
        c_e1 = self.c_enc1(x_c)
        c_e2 = self.c_enc2(self.pool(c_e1))
        c_e3 = self.c_enc3(self.pool(c_e2))
        c_bn = self.c_bottleneck(self.pool(c_e3))

        # ---- Single cross-branch fusion at bottleneck, inside halo only ----
        fused     = self.fuse(torch.cat([p_bn, c_bn], dim=1))   # (B, base*16, H/8, W/8)
        p_fused, c_fused = fused.split(self.base * 8, dim=1)    # each (B, base*8, …)

        halo = self.halo_16.float().unsqueeze(0).unsqueeze(0)   # (1,1, H/8, W/8)
        p_bn = p_bn * (1.0 - halo) + p_fused * halo            # blend inside halo
        c_bn = c_bn * (1.0 - halo) + c_fused * halo

        # ---- Branch P decoder (branch-local skips) ----
        p_d3 = self.p_up3(p_bn)
        p_d3 = self.p_dec3(torch.cat([p_d3, p_e3], dim=1))
        p_d2 = self.p_up2(p_d3)
        p_d2 = self.p_dec2(torch.cat([p_d2, p_e2], dim=1))
        p_d1 = self.p_up1(p_d2)
        p_d1 = self.p_dec1(torch.cat([p_d1, p_e1], dim=1))
        p_out = self.p_out(p_d1)                                # (B,1,H,W)

        # ---- Branch C decoder (branch-local skips) ----
        c_d3 = self.c_up3(c_bn)
        c_d3 = self.c_dec3(torch.cat([c_d3, c_e3], dim=1))
        c_d2 = self.c_up2(c_d3)
        c_d2 = self.c_dec2(torch.cat([c_d2, c_e2], dim=1))
        c_d1 = self.c_up1(c_d2)
        c_d1 = self.c_dec1(torch.cat([c_d1, c_e1], dim=1))
        c_out = self.c_out(c_d1)                                # (B,1,H,W)

        # ---- Stitch branches: each pixel comes from its owning branch ----
        # C-branch owns the protected region (region_full == True)
        # P-branch owns the unprotected periphery
        owner = self.region_full.unsqueeze(0).unsqueeze(0)     # bool (1,1,H,W)
        return torch.where(owner, c_out, p_out)                # (B,1,H,W)
