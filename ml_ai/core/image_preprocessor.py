"""
Image Preprocessor Module
AI-Based Virtual Try-On and Fit Recommendation System

Prepares real-world phone photos for the try-on pipeline:
    - EXIF rotation correction
    - Smart resizing (keep within 2048, upscale if too small)
    - Contrast normalization (CLAHE)
    - Light noise reduction (bilateral filter)

Usage:
    from ml_ai.core.image_preprocessor import preprocess_for_tryon
    preprocessed, info = preprocess_for_tryon(image)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ExifTags


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingInfo:
    """Metadata about what preprocessing was applied."""
    original_size: Tuple[int, int]       # (height, width) before changes
    final_size: Tuple[int, int]          # (height, width) after changes
    was_rotated: bool = False            # True if EXIF rotation was applied
    was_resized: bool = False            # True if image was resized
    contrast_enhanced: bool = False      # True if CLAHE was applied
    noise_reduced: bool = False          # True if bilateral filter applied
    steps_applied: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# EXIF rotation
# ---------------------------------------------------------------------------

def fix_exif_rotation(image_bgr: np.ndarray, image_path: str | None = None) -> Tuple[np.ndarray, bool]:
    """
    Fix image rotation using EXIF orientation tag.

    Args:
        image_bgr:  BGR image (already loaded by OpenCV, which ignores EXIF)
        image_path: Original file path to read EXIF from

    Returns:
        (corrected_image, was_rotated)
    """
    if image_path is None:
        return image_bgr, False

    try:
        pil_img = Image.open(image_path)
        exif_data = pil_img._getexif()
        if exif_data is None:
            return image_bgr, False

        # Find the orientation tag
        orientation_key = None
        for tag_id, tag_name in ExifTags.TAGS.items():
            if tag_name == "Orientation":
                orientation_key = tag_id
                break

        if orientation_key is None or orientation_key not in exif_data:
            return image_bgr, False

        orientation = exif_data[orientation_key]

        # Apply rotation based on EXIF orientation value
        if orientation == 3:
            image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_180)
        elif orientation == 6:
            image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 8:
            image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image_bgr, False

        return image_bgr, True

    except Exception:
        # If EXIF reading fails, return original
        return image_bgr, False


# ---------------------------------------------------------------------------
# Smart resize
# ---------------------------------------------------------------------------

MAX_DIMENSION = 2048
MIN_SHORT_SIDE = 400


def smart_resize(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Resize image if too large or too small.

    - If any dimension > MAX_DIMENSION, scale down proportionally
    - If shortest side < MIN_SHORT_SIDE, scale up proportionally

    Args:
        image: BGR image

    Returns:
        (resized_image, was_resized)
    """
    h, w = image.shape[:2]
    scale = 1.0

    # Downscale if too large
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)

    # Upscale if too small (only if not already downscaling)
    elif min(h, w) < MIN_SHORT_SIDE:
        scale = MIN_SHORT_SIDE / min(h, w)

    if abs(scale - 1.0) < 0.01:
        return image, False

    new_w = int(w * scale)
    new_h = int(h * scale)

    # Use INTER_AREA for downscale, INTER_CUBIC for upscale
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    return resized, True


# ---------------------------------------------------------------------------
# Contrast normalization
# ---------------------------------------------------------------------------

def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    on the L channel of LAB color space.

    This normalizes lighting variations from phone cameras without
    distorting colors.

    Args:
        image:      BGR image
        clip_limit: CLAHE clip limit (higher = more contrast)

    Returns:
        Contrast-enhanced BGR image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return result


# ---------------------------------------------------------------------------
# Noise reduction
# ---------------------------------------------------------------------------

def reduce_noise(image: np.ndarray) -> np.ndarray:
    """
    Apply light bilateral filter to reduce phone camera noise
    while preserving edges.

    Args:
        image: BGR image

    Returns:
        Denoised BGR image
    """
    return cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)


# ---------------------------------------------------------------------------
# Check if image needs contrast enhancement
# ---------------------------------------------------------------------------

def _needs_contrast_enhancement(image: np.ndarray) -> bool:
    """
    Determine if the image has poor contrast and would benefit from CLAHE.

    Checks if the standard deviation of brightness is low (flat histogram)
    or if the image is very dark/bright overall.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))

    # Low contrast (flat histogram)
    if std_brightness < 40:
        return True

    # Very dark or very bright
    if mean_brightness < 50 or mean_brightness > 210:
        return True

    return False


def _needs_noise_reduction(image: np.ndarray) -> bool:
    """
    Estimate if image has noticeable noise by checking high-frequency energy.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_estimate = float(np.std(laplacian))

    # High Laplacian std can indicate noise (threshold tuned for phone photos)
    return noise_estimate > 30


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_for_tryon(
    image: np.ndarray,
    image_path: str | None = None,
    force_contrast: bool = False,
    force_denoise: bool = False,
) -> Tuple[np.ndarray, PreprocessingInfo]:
    """
    Full preprocessing pipeline for real-world phone photos.

    Steps (applied in order):
        1. EXIF rotation correction (if image_path provided)
        2. Smart resize (keep proportions, fit within bounds)
        3. Contrast enhancement (CLAHE, if image needs it or forced)
        4. Noise reduction (bilateral filter, if noisy or forced)

    Args:
        image:           BGR image (from cv2.imread or similar)
        image_path:      Original file path (for EXIF reading). Optional.
        force_contrast:  Force contrast enhancement even if not detected as needed
        force_denoise:   Force noise reduction even if not detected as needed

    Returns:
        (preprocessed_image, PreprocessingInfo)
    """
    info = PreprocessingInfo(
        original_size=(image.shape[0], image.shape[1]),
        final_size=(image.shape[0], image.shape[1]),
    )

    result = image.copy()

    # 1. EXIF rotation
    result, rotated = fix_exif_rotation(result, image_path)
    if rotated:
        info.was_rotated = True
        info.steps_applied.append("EXIF rotation corrected")

    # 2. Smart resize
    result, resized = smart_resize(result)
    if resized:
        info.was_resized = True
        info.steps_applied.append(
            f"Resized from {info.original_size} to {result.shape[:2]}"
        )

    # 3. Contrast enhancement (only if needed or forced)
    if force_contrast or _needs_contrast_enhancement(result):
        result = enhance_contrast(result)
        info.contrast_enhanced = True
        info.steps_applied.append("CLAHE contrast enhancement applied")

    # 4. Noise reduction (only if needed or forced)
    if force_denoise or _needs_noise_reduction(result):
        result = reduce_noise(result)
        info.noise_reduced = True
        info.steps_applied.append("Bilateral noise reduction applied")

    info.final_size = (result.shape[0], result.shape[1])

    return result, info
