"""
Image Overlay and Composition Module
AI-Based Virtual Try-On and Fit Recommendation System

Updated for Phase 2: adds segmentation-aware garment compositing
on top of the existing simple alpha blend functions.
"""

from __future__ import annotations

from typing import Optional, Tuple, List
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# NEW: Segmentation-aware garment compositing (Phase 2)
# ---------------------------------------------------------------------------

def composite_garment_on_person(
    person_image: np.ndarray,
    warped_garment: np.ndarray,
    garment_mask: np.ndarray,
    body_mask: Optional[np.ndarray] = None,
    alpha: float = 0.92,
    edge_feather_px: int = 8,
) -> np.ndarray:
    """
    Composite a warped garment onto a person image.

    Uses a three-layer blending strategy:
        1. Garment mask  — where the warped garment has valid pixels
        2. Body mask     — (optional) restricts blending to upper-body region
        3. Edge feather  — softens garment boundary for a natural look

    Args:
        person_image:    BGR person photo (H, W, 3)
        warped_garment:  TPS-warped garment image, same size as person_image
                         May be (H, W, 3) or (H, W, 4)
        garment_mask:    uint8 binary mask (H, W) — 1 where garment present
        body_mask:       uint8 binary mask (H, W) — 1 where upper body is
                         If None, compositing is not body-restricted
        alpha:           Maximum garment opacity [0..1]
        edge_feather_px: Gaussian blur radius for mask edge softening

    Returns:
        Composite BGR image (H, W, 3), same size as person_image
    """
    if person_image.shape[:2] != warped_garment.shape[:2]:
        raise ValueError(
            f"person_image {person_image.shape[:2]} and "
            f"warped_garment {warped_garment.shape[:2]} must be same size"
        )

    # Ensure garment is BGR (drop alpha channel for blending)
    if warped_garment.ndim == 3 and warped_garment.shape[2] == 4:
        garment_bgr = warped_garment[:, :, :3]
        # Use garment_mask directly — don't multiply by alpha channel
        # because TPS border fill sets alpha=0 in valid areas too
        base_mask = garment_mask.astype(np.float32)
    else:
        garment_bgr = warped_garment
        base_mask = garment_mask.astype(np.float32)

    # Restrict to body region if segmentation mask provided
    if body_mask is not None:
        body_f = body_mask.astype(np.float32)
        blend_mask = base_mask * body_f
    else:
        blend_mask = base_mask

    # Feather edges for natural look
    if edge_feather_px > 0:
        ksize = edge_feather_px * 2 + 1          # must be odd
        blend_mask = cv2.GaussianBlur(
            blend_mask, (ksize, ksize), sigmaX=edge_feather_px / 2
        )

    # Scale by overall alpha
    blend_mask = (blend_mask * alpha).clip(0.0, 1.0)

    # Expand to 3 channels for broadcasting
    blend_mask_3ch = blend_mask[:, :, np.newaxis]

    # Final blend
    person_f  = person_image.astype(np.float32)
    garment_f = garment_bgr.astype(np.float32)

    composite = garment_f * blend_mask_3ch + person_f * (1.0 - blend_mask_3ch)
    return composite.clip(0, 255).astype(np.uint8)


def add_garment_shadow(
    composite: np.ndarray,
    garment_mask: np.ndarray,
    shadow_strength: float = 0.25,
    blur_px: int = 12,
    offset_xy: Tuple[int, int] = (3, 4),
) -> np.ndarray:
    """
    Add a subtle shadow under the garment edges for depth.

    Args:
        composite:        Current composite image (BGR)
        garment_mask:     Binary garment mask (H, W) uint8
        shadow_strength:  How dark the shadow is [0..1]
        blur_px:          Shadow softness in pixels
        offset_xy:        (x, y) pixel offset of shadow

    Returns:
        Composite with shadow applied (BGR)
    """
    result = composite.copy().astype(np.float32)
    h, w = result.shape[:2]

    # Build shadow mask: shift and blur garment mask
    shadow_mask = np.zeros_like(garment_mask, dtype=np.float32)
    ox, oy = offset_xy
    # Shift
    src_y1 = max(0, -oy);  src_y2 = min(h, h - oy)
    dst_y1 = max(0,  oy);  dst_y2 = min(h, h + oy)
    src_x1 = max(0, -ox);  src_x2 = min(w, w - ox)
    dst_x1 = max(0,  ox);  dst_x2 = min(w, w + ox)

    shadow_mask[dst_y1:dst_y2, dst_x1:dst_x2] = \
        garment_mask[src_y1:src_y2, src_x1:src_x2].astype(np.float32)

    # Blur
    ksize = blur_px * 2 + 1
    shadow_mask = cv2.GaussianBlur(shadow_mask, (ksize, ksize), blur_px / 2)

    # Only apply shadow where garment is NOT present (outside garment)
    outside_garment = (1.0 - garment_mask.astype(np.float32))
    shadow_mask = shadow_mask * outside_garment * shadow_strength

    shadow_3ch = shadow_mask[:, :, np.newaxis]
    result = result * (1.0 - shadow_3ch)

    return result.clip(0, 255).astype(np.uint8)


def enhance_garment_edges(
    composite: np.ndarray,
    garment_mask: np.ndarray,
    sharpness: float = 0.4,
) -> np.ndarray:
    """
    Lightly sharpen garment edges to prevent blurry look after warping.

    Args:
        composite:     Current composite image (BGR)
        garment_mask:  Binary garment mask (H, W) uint8
        sharpness:     Sharpening strength [0..1]

    Returns:
        Sharpened composite (BGR)
    """
    # Unsharp mask on the garment region only
    blurred = cv2.GaussianBlur(composite, (0, 0), sigmaX=2.0)
    sharp   = cv2.addWeighted(composite, 1.0 + sharpness, blurred, -sharpness, 0)

    # Apply only inside garment
    mask_3ch = garment_mask[:, :, np.newaxis].astype(np.float32)
    result   = sharp.astype(np.float32) * mask_3ch + \
               composite.astype(np.float32) * (1.0 - mask_3ch)

    return result.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# EXISTING functions (unchanged — preserved for backward compatibility)
# ---------------------------------------------------------------------------

def overlay_garment(
    background_image: np.ndarray,
    garment_image: np.ndarray,
    position: Tuple[int, int],
    alpha: float = 0.8
) -> np.ndarray:
    """
    Overlay garment onto background image at a fixed position.
    (Original simple overlay — kept for backward compatibility)
    """
    if not isinstance(background_image, np.ndarray):
        raise TypeError("Background image must be numpy array")
    if not isinstance(garment_image, np.ndarray):
        raise TypeError("Garment image must be numpy array")
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")

    result = background_image.copy()
    x, y = position
    g_height, g_width = garment_image.shape[:2]

    if x < 0 or y < 0:
        raise ValueError("Position must be non-negative")
    if x + g_width > result.shape[1] or y + g_height > result.shape[0]:
        raise ValueError("Garment extends beyond image bounds")

    roi = result[y:y + g_height, x:x + g_width]

    if len(garment_image.shape) == 3 and garment_image.shape[2] == 4:
        garment_bgr   = garment_image[:, :, :3]
        garment_alpha = garment_image[:, :, 3].astype(float) / 255.0
        blended = (garment_bgr.astype(float) * garment_alpha[:, :, None] +
                   roi.astype(float) * (1 - garment_alpha[:, :, None]))
    else:
        garment_bgr = (garment_image if len(garment_image.shape) == 3
                       else cv2.cvtColor(garment_image, cv2.COLOR_GRAY2BGR))
        blended = (garment_bgr.astype(float) * alpha +
                   roi.astype(float) * (1 - alpha))

    result[y:y + g_height, x:x + g_width] = blended.astype(np.uint8)
    return result


def composite_multiple_garments(
    background_image: np.ndarray,
    garments: list,
    positions: list,
    alphas: Optional[list] = None
) -> np.ndarray:
    """Overlay multiple garments onto background. (Original — unchanged)"""
    if len(garments) != len(positions):
        raise ValueError("Number of garments must match positions")
    if alphas is None:
        alphas = [0.8] * len(garments)
    elif len(alphas) != len(garments):
        raise ValueError("Number of alphas must match garments")

    result = background_image.copy()
    for garment, position, alpha in zip(garments, positions, alphas):
        result = overlay_garment(result, garment, position, alpha)
    return result


def blend_images(
    image1: np.ndarray,
    image2: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """Blend two images together. (Original — unchanged)"""
    if image1.shape != image2.shape:
        raise ValueError("Images must have same dimensions")
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")

    blended = (image1.astype(float) * alpha +
               image2.astype(float) * (1 - alpha))
    return blended.astype(np.uint8)