"""Garment Normalizer Module
Phase 1 Task 1: Standardize garment images before TPS warping.

Pipeline:
    1. Background removal (rembg)
    2. Mask cleaning (morphological open/close + Gaussian blur)
    3. Mask quality validation
    4. Smart cropping with padding
    5. Centroid-based centering on 512×512 canvas
    6. Clean binary mask generation
    7. Rich metadata output

Outputs per garment:
    garment.png        — RGBA, clean transparent background
    garment_mask.png   — Binary (0/255), smooth edges
    metadata.json      — Updated with normalization info
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MaskQuality:
    """Quality metrics for the garment mask."""
    area_ratio: float          # foreground area / total area
    num_components: int        # connected components (should be 1)
    is_valid: bool             # True if mask passes quality checks

    def to_dict(self) -> dict:
        return {
            "area_ratio": round(self.area_ratio, 4),
            "components": self.num_components,
            "is_valid": self.is_valid,
        }


@dataclass
class NormalizationResult:
    """Output of garment normalization."""
    garment_image: np.ndarray       # RGBA 512×512
    garment_mask: np.ndarray        # Binary 512×512
    original_size: Tuple[int, int]  # (H, W) of input image
    bbox: Tuple[int, int, int, int] # (x1, y1, x2, y2) of content in original
    centroid: Tuple[float, float]   # (cx, cy) on the output canvas
    scale_to_canvas: float          # scale factor applied
    mask_quality: MaskQuality
    success: bool = True
    error: str = ""
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main normalizer class
# ---------------------------------------------------------------------------

class GarmentNormalizer:
    """
    Full garment normalization pipeline.

    Usage:
        normalizer = GarmentNormalizer(canvas_size=512)
        result = normalizer.normalize(image_bgr_or_bgra)
        # result.garment_image  → RGBA 512×512
        # result.garment_mask   → binary 512×512
    """

    def __init__(self, canvas_size: int = 512):
        self.canvas_size = canvas_size
        self._rembg_session = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, image: np.ndarray) -> NormalizationResult:
        """
        Run the full normalization pipeline on a garment image.

        Args:
            image: BGR or BGRA garment image (numpy array)

        Returns:
            NormalizationResult with all outputs and metadata
        """
        warnings: List[str] = []
        orig_h, orig_w = image.shape[:2]

        # ── Step 1: Background removal ────────────────────────────────
        try:
            rgba = self._remove_background(image)
        except Exception as e:
            return NormalizationResult(
                garment_image=np.zeros((self.canvas_size, self.canvas_size, 4), dtype=np.uint8),
                garment_mask=np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8),
                original_size=(orig_h, orig_w),
                bbox=(0, 0, 0, 0),
                centroid=(0, 0),
                scale_to_canvas=0.0,
                mask_quality=MaskQuality(0, 0, False),
                success=False,
                error=f"Background removal failed: {e}",
            )

        # ── Step 2: Clean the mask ────────────────────────────────────
        # Make a writable copy (rembg may return read-only arrays)
        rgba = rgba.copy()
        raw_mask = rgba[:, :, 3].copy()
        clean_mask = self._clean_mask(raw_mask)

        # Apply cleaned mask back to RGBA
        rgba[:, :, 3] = clean_mask

        # ── Step 3: Validate mask quality ─────────────────────────────
        mask_quality = self._validate_mask(clean_mask)
        if not mask_quality.is_valid:
            warnings.append(
                f"Mask quality issue: area_ratio={mask_quality.area_ratio:.2f}, "
                f"components={mask_quality.num_components}"
            )

        # ── Step 4: Smart crop with padding ───────────────────────────
        cropped_rgba, bbox = self._smart_crop(rgba, clean_mask)
        if cropped_rgba is None:
            return NormalizationResult(
                garment_image=np.zeros((self.canvas_size, self.canvas_size, 4), dtype=np.uint8),
                garment_mask=np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8),
                original_size=(orig_h, orig_w),
                bbox=(0, 0, 0, 0),
                centroid=(0, 0),
                scale_to_canvas=0.0,
                mask_quality=mask_quality,
                success=False,
                error="No foreground content found after background removal",
            )

        # ── Step 5: Center on canvas using centroid ───────────────────
        canvas_rgba, scale, centroid = self._center_on_canvas(cropped_rgba)

        # ── Step 6: Generate final clean binary mask ──────────────────
        final_mask = self._generate_binary_mask(canvas_rgba[:, :, 3])

        # Ensure perfect transparency in the output image using the clean mask
        # 1. Zero out original RGB background (prevents grey polygon bug during fallback)
        bgr = canvas_rgba[:, :, :3]
        mask_3ch = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
        bgr_clean = cv2.bitwise_and(bgr, mask_3ch)

        # 2. Dilate the clean RGB outwards (prevents dark fringes when feathering mask later)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bgr_dilated = cv2.dilate(bgr_clean, kernel, iterations=3)

        # 3. Apply the dilated RGB and final alpha mask
        canvas_rgba[:, :, :3] = np.where(mask_3ch > 0, bgr_clean, bgr_dilated)
        canvas_rgba[:, :, 3] = final_mask

        return NormalizationResult(
            garment_image=canvas_rgba,
            garment_mask=final_mask,
            original_size=(orig_h, orig_w),
            bbox=bbox,
            centroid=centroid,
            scale_to_canvas=round(scale, 4),
            mask_quality=mask_quality,
            success=True,
            warnings=warnings,
        )

    def normalize_and_save(
        self,
        input_image_path: str | Path,
        output_dir: str | Path,
        update_metadata: bool = True,
    ) -> NormalizationResult:
        """
        Load a garment image, normalize it, and save all outputs.

        Args:
            input_image_path: Path to the raw garment image
            output_dir: Directory to write garment.png, garment_mask.png
            update_metadata: If True, update metadata.json in output_dir

        Returns:
            NormalizationResult
        """
        input_path = Path(input_image_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load image (with alpha if present)
        image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return NormalizationResult(
                garment_image=np.zeros((self.canvas_size, self.canvas_size, 4), dtype=np.uint8),
                garment_mask=np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8),
                original_size=(0, 0),
                bbox=(0, 0, 0, 0),
                centroid=(0, 0),
                scale_to_canvas=0.0,
                mask_quality=MaskQuality(0, 0, False),
                success=False,
                error=f"Failed to load image: {input_path}",
            )

        result = self.normalize(image)
        if not result.success:
            return result

        # ── Save outputs ──────────────────────────────────────────────
        garment_path = out_dir / "garment.png"
        mask_path = out_dir / "garment_mask.png"

        cv2.imwrite(str(garment_path), result.garment_image)
        cv2.imwrite(str(mask_path), result.garment_mask)

        logger.info(f"Saved: {garment_path}")
        logger.info(f"Saved: {mask_path}")

        # ── Update metadata.json ──────────────────────────────────────
        if update_metadata:
            self._update_metadata(out_dir, result)

        return result

    # ------------------------------------------------------------------
    # Step 1: Background removal (rembg)
    # ------------------------------------------------------------------

    def _remove_background(self, image: np.ndarray) -> np.ndarray:
        """Remove background using rembg. Returns BGRA image."""
        from rembg import remove, new_session

        # Lazy-load rembg session (u2net is the default, good quality)
        if self._rembg_session is None:
            self._rembg_session = new_session("u2net")

        # Ensure BGR input (rembg expects BGR or RGB, returns BGRA)
        if image.shape[2] == 4:
            bgr = image[:, :, :3]
        else:
            bgr = image

        # rembg returns BGRA with transparent background
        result = remove(bgr, session=self._rembg_session)

        # Ensure 4-channel output
        if result.shape[2] == 3:
            alpha = np.full(result.shape[:2], 255, dtype=np.uint8)
            result = np.dstack([result, alpha])

        return result

    # ------------------------------------------------------------------
    # Step 2: Mask cleaning (morphological ops + blur)
    # ------------------------------------------------------------------

    def _clean_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        """
        Clean the alpha mask using morphological operations.

        Pipeline:
            1. Threshold to binary
            2. Morphological CLOSE (fill holes)
            3. Morphological OPEN (remove noise)
            4. Gaussian blur for smooth edges
            5. Re-threshold to binary
        """
        # Threshold to binary
        _, binary = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)

        # Close: fill small holes inside the garment
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # Open: remove small noise blobs outside the garment
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # Gaussian blur for smooth edges
        smoothed = cv2.GaussianBlur(opened, (5, 5), sigmaX=1.5)

        # Re-threshold to strict binary
        _, final = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

        return final

    # ------------------------------------------------------------------
    # Step 3: Mask quality validation
    # ------------------------------------------------------------------

    def _validate_mask(self, mask: np.ndarray) -> MaskQuality:
        """
        Validate the garment mask quality.

        Checks:
            - Foreground area ratio should be between 10% and 80%
            - Should have exactly 1 main connected component
        """
        total_pixels = mask.shape[0] * mask.shape[1]
        fg_pixels = np.count_nonzero(mask)
        area_ratio = fg_pixels / total_pixels if total_pixels > 0 else 0.0

        # Connected components (ignoring background component 0)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # num_labels includes background, so foreground components = num_labels - 1
        num_fg_components = num_labels - 1

        # Validity: area in range and ideally 1 main blob
        area_ok = 0.10 <= area_ratio <= 0.80
        components_ok = num_fg_components >= 1  # at least some foreground

        # If multiple components, keep only the largest one
        if num_fg_components > 1:
            # Find the largest foreground component (stats[:, cv2.CC_STAT_AREA])
            # Skip label 0 (background)
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1  # +1 because we skipped bg
            # Keep only the largest component
            mask[labels != largest_label] = 0
            num_fg_components = 1  # now cleaned to 1

        is_valid = area_ok and (num_fg_components == 1)

        return MaskQuality(
            area_ratio=area_ratio,
            num_components=num_fg_components,
            is_valid=is_valid,
        )

    # ------------------------------------------------------------------
    # Step 4: Smart cropping with padding
    # ------------------------------------------------------------------

    def _smart_crop(
        self, rgba: np.ndarray, mask: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
        """
        Crop to content bounding box with proportional padding.

        Padding = 10% of max(content_width, content_height) on each side.
        This prevents sleeve/collar clipping.

        Returns:
            (cropped_rgba, (x1, y1, x2, y2)) or (None, (0,0,0,0)) if empty
        """
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None, (0, 0, 0, 0)

        x, y, w, h = cv2.boundingRect(coords)

        # Add padding: 10% of max dimension on each side
        pad = int(0.10 * max(w, h))
        img_h, img_w = rgba.shape[:2]

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)

        cropped = rgba[y1:y2, x1:x2].copy()
        return cropped, (x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # Step 5: Centroid-based centering on canvas
    # ------------------------------------------------------------------

    def _center_on_canvas(
        self, cropped_rgba: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Place the cropped garment on a canvas_size×canvas_size transparent canvas,
        centered by the mask centroid (NOT bounding box center).

        Returns:
            (canvas_rgba, scale_factor, (centroid_x, centroid_y) on canvas)
        """
        cs = self.canvas_size
        crop_h, crop_w = cropped_rgba.shape[:2]

        # Scale to fit within canvas, preserving aspect ratio
        scale = cs / max(crop_h, crop_w)
        if scale > 1.0:
            # Don't upscale, only downscale
            scale = min(scale, 1.0)
            # Actually, if the garment is smaller than canvas, we still
            # want to fill most of it. Allow up to 0.9 * canvas.
            scale = (cs * 0.9) / max(crop_h, crop_w)

        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)

        # Resize with high-quality interpolation
        if scale < 1.0:
            interp = cv2.INTER_AREA  # Best for downscaling
        else:
            interp = cv2.INTER_LANCZOS4  # Best for upscaling

        resized = cv2.resize(cropped_rgba, (new_w, new_h), interpolation=interp)

        # Compute centroid of the mask in the resized image
        mask_resized = resized[:, :, 3]
        moments = cv2.moments(mask_resized)
        if moments["m00"] > 0:
            cx_local = moments["m10"] / moments["m00"]
            cy_local = moments["m01"] / moments["m00"]
        else:
            # Fallback to bounding box center
            cx_local = new_w / 2
            cy_local = new_h / 2

        # Place on canvas so centroid maps to canvas center
        canvas_cx = cs / 2
        canvas_cy = cs / 2

        offset_x = int(canvas_cx - cx_local)
        offset_y = int(canvas_cy - cy_local)

        # Create transparent canvas
        canvas = np.zeros((cs, cs, 4), dtype=np.uint8)

        # Compute paste region (clamp to canvas bounds)
        src_x1 = max(0, -offset_x)
        src_y1 = max(0, -offset_y)
        src_x2 = min(new_w, cs - offset_x)
        src_y2 = min(new_h, cs - offset_y)

        dst_x1 = max(0, offset_x)
        dst_y1 = max(0, offset_y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]

        # Actual centroid position on canvas
        centroid_on_canvas = (canvas_cx, canvas_cy)

        return canvas, scale, centroid_on_canvas

    # ------------------------------------------------------------------
    # Step 6: Generate clean binary mask
    # ------------------------------------------------------------------

    def _generate_binary_mask(self, alpha_channel: np.ndarray) -> np.ndarray:
        """
        Generate a strictly binary mask (0 or 255) from the alpha channel.
        Ensures smooth edges, no holes.
        """
        _, binary = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)

        # Final cleanup: close any tiny holes that survived
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        return binary

    # ------------------------------------------------------------------
    # Step 7: Update metadata
    # ------------------------------------------------------------------

    def _update_metadata(self, garment_dir: Path, result: NormalizationResult):
        """Update metadata.json with normalization info."""
        meta_path = garment_dir / "metadata.json"

        metadata = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read existing metadata: {e}")

        # Add normalization data
        metadata["normalized"] = True
        metadata["normalization"] = {
            "original_size": list(result.original_size),
            "bbox": list(result.bbox),
            "scale_to_canvas": result.scale_to_canvas,
            "centroid": [round(result.centroid[0], 2), round(result.centroid[1], 2)],
            "canvas_size": self.canvas_size,
            "mask_quality": result.mask_quality.to_dict(),
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Updated metadata: {meta_path}")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_normalizer: Optional[GarmentNormalizer] = None


def get_normalizer(canvas_size: int = 512) -> GarmentNormalizer:
    """Get or create a global normalizer (reuses rembg session)."""
    global _normalizer
    if _normalizer is None or _normalizer.canvas_size != canvas_size:
        _normalizer = GarmentNormalizer(canvas_size=canvas_size)
    return _normalizer


def normalize_garment_image(
    input_path: str | Path,
    output_dir: str | Path,
    canvas_size: int = 512,
) -> NormalizationResult:
    """Convenience function to normalize a single garment."""
    normalizer = get_normalizer(canvas_size)
    return normalizer.normalize_and_save(input_path, output_dir)
