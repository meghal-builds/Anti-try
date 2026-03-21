"""
Virtual Try-On Engine
AI-Based Virtual Try-On and Fit Recommendation System

Main pipeline that wires together:
    pose detection → keypoint resolution → TPS warp → segmentation-aware composite

Usage:
    engine  = TryOnEngine()
    result  = engine.run(person_image, garment_image, garment_category)
    display = result.composite_image   # final try-on image
    mask    = result.garment_mask      # where garment was placed
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from ml_ai.core.tps_warp import tps_warp_with_mask, compute_warp_quality
from ml_ai.core.garment_keypoints import get_garment_schema, resolve_points
from ml_ai.core.pose_detection import detect_pose
from ml_ai.core.segmentation import segment_body
from ml_ai.core.model_layer import load_models
from ml_ai.core.overlay import composite_garment_on_person
from ml_ai.core.image_preprocessor import preprocess_for_tryon


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TryOnResult:
    """
    Output of a single try-on run.

    Attributes:
        composite_image:   Final image with garment on person (BGR)
        warped_garment:    Warped garment before compositing
        garment_mask:      Binary mask showing where garment was placed
        person_image:      Original person image (unmodified)
        warnings:          Non-fatal issues encountered during processing
        processing_time_s: Total wall-clock time in seconds
        success:           True if compositing completed without fatal errors
        error:             Error message if success=False
    """
    composite_image: np.ndarray | None
    warped_garment: np.ndarray | None
    garment_mask: np.ndarray | None
    person_image: np.ndarray
    warnings: List[str] = field(default_factory=list)
    processing_time_s: float = 0.0
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TryOnEngine:
    """
    Full virtual try-on pipeline.

    Lifecycle:
        engine = TryOnEngine()          # loads models once
        result = engine.run(...)        # can be called many times
        engine.release()                # frees MediaPipe resources
    """

    def __init__(self, config_path: str = "database/config/models.json"):
        """
        Initialize engine and load segmentation + pose models.

        Args:
            config_path: Path to models.json config
        """
        self._seg_model, self._pose_model = load_models(config_path)
        self._pose_detector = None   # lazy-loaded real MediaPipe instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        person_image: np.ndarray,
        garment_image: np.ndarray,
        garment_category: str,
        blend_alpha: float = 0.92,
        shoulder_scale: float = 1.0,
        use_segmentation_mask: bool = True,
    ) -> TryOnResult:
        """
        Run full virtual try-on pipeline.

        Args:
            person_image:           BGR image of the person (H, W, 3)
            garment_image:          BGR or BGRA garment image
            garment_category:       'tshirt', 'shirt', or 'jacket'
            blend_alpha:            Garment opacity in final composite [0..1]
            shoulder_scale:         Scale garment width relative to body
                                    (1.0 = exact fit, 1.05 = slightly loose)
            use_segmentation_mask:  If True, blend only over torso region

        Returns:
            TryOnResult with composite image and diagnostics
        """
        t_start = time.perf_counter()
        warnings: List[str] = []

        # ── 1. Validate inputs ──────────────────────────────────────────
        valid, err = self._validate_inputs(person_image, garment_image, garment_category)
        if not valid:
            return TryOnResult(
                composite_image=None, warped_garment=None, garment_mask=None,
                person_image=person_image, success=False, error=err
            )

        person_h, person_w = person_image.shape[:2]

        # ── 1½. Preprocess person image for real-world photos ────────
        try:
            person_image, preprocess_info = preprocess_for_tryon(person_image)
            if preprocess_info.steps_applied:
                warnings.append(
                    f"Image preprocessed: {', '.join(preprocess_info.steps_applied)}"
                )
            # Update dimensions after preprocessing
            person_h, person_w = person_image.shape[:2]
        except Exception as e:
            warnings.append(f"Preprocessing skipped: {e}")

        # ── 2. Pose detection ───────────────────────────────────────────
        try:
            pose_result = self._detect_pose(person_image)
        except RuntimeError as e:
            return TryOnResult(
                composite_image=None, warped_garment=None, garment_mask=None,
                person_image=person_image, success=False,
                error=f"Pose detection failed: {e}"
            )
        warnings.extend(pose_result.warnings)

        # ── 3. Segmentation ─────────────────────────────────────────────
        seg_result = None
        if use_segmentation_mask:
            try:
                seg_result = segment_body(person_image, self._seg_model)
                warnings.extend(seg_result.warnings)
            except Exception as e:
                warnings.append(f"Segmentation skipped: {e}")
                seg_result = None

        # ── 4. Resolve TPS control points ───────────────────────────────
        schema = get_garment_schema(garment_category)
        src_pts, dst_pts = resolve_points(
            schema, garment_image,
            pose_result.keypoints,
            person_w, person_h,
            shoulder_scale=shoulder_scale
        )

        if src_pts is None or dst_pts is None:
            return TryOnResult(
                composite_image=None, warped_garment=None, garment_mask=None,
                person_image=person_image, success=False,
                error=(
                    "Required body keypoints missing for try-on. "
                    "Ensure the full upper body is visible."
                )
            )

        # ── 5. Warp quality check ────────────────────────────────────────
        quality = compute_warp_quality(src_pts, dst_pts)
        warnings.extend(quality["warnings"])
        if not quality["is_valid"]:
            return TryOnResult(
                composite_image=None, warped_garment=None, garment_mask=None,
                person_image=person_image, success=False,
                error="Warp configuration is degenerate — cannot proceed."
            )

        # ── 6. TPS warp garment ──────────────────────────────────────────
        try:
            warped_garment, garment_mask = tps_warp_with_mask(
                garment_image,
                src_pts,
                dst_pts,
                output_size=(person_h, person_w)
            )
        except Exception as e:
            return TryOnResult(
                composite_image=None, warped_garment=None, garment_mask=None,
                person_image=person_image, success=False,
                error=f"TPS warping failed: {e}"
            )

        # ── 7. Composite ─────────────────────────────────────────────────
        body_mask = None
        if seg_result is not None:
            body_mask = _build_upper_body_mask(seg_result)

        composite = composite_garment_on_person(
            person_image=person_image,
            warped_garment=warped_garment,
            garment_mask=garment_mask,
            body_mask=body_mask,
            alpha=blend_alpha
        )

        elapsed = time.perf_counter() - t_start

        return TryOnResult(
            composite_image=composite,
            warped_garment=warped_garment,
            garment_mask=garment_mask,
            person_image=person_image,
            warnings=warnings,
            processing_time_s=round(elapsed, 3),
            success=True
        )

    def release(self) -> None:
        """Release MediaPipe and other held resources."""
        if self._pose_detector is not None:
            try:
                self._pose_detector.release()
            except Exception:
                pass
            self._pose_detector = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_pose(self, image: np.ndarray):
        """Use real MediaPipe detector (lazy singleton)."""
        from ml_ai.core.mediapipe_real import create_real_pose_detector
        if self._pose_detector is None:
            self._pose_detector = create_real_pose_detector()
        return self._pose_detector.detect_pose(image)

    @staticmethod
    def _validate_inputs(
        person_image: np.ndarray,
        garment_image: np.ndarray,
        garment_category: str
    ) -> Tuple[bool, str]:
        """Basic input sanity checks."""
        if not isinstance(person_image, np.ndarray):
            return False, "person_image must be a numpy array"
        if not isinstance(garment_image, np.ndarray):
            return False, "garment_image must be a numpy array"
        if len(person_image.shape) != 3 or person_image.shape[2] not in (3, 4):
            return False, "person_image must be (H, W, 3) or (H, W, 4)"
        if len(garment_image.shape) != 3 or garment_image.shape[2] not in (3, 4):
            return False, "garment_image must be (H, W, 3) or (H, W, 4)"

        # Convert BGRA person to BGR
        supported = {"tshirt", "shirt", "jacket", "t-shirt", "t_shirt"}
        if garment_category.lower().strip() not in supported:
            return False, f"Unsupported garment category: '{garment_category}'"
        return True, ""


# ---------------------------------------------------------------------------
# Mask helper
# ---------------------------------------------------------------------------

def _build_upper_body_mask(seg_result) -> np.ndarray:
    """
    Build a combined upper-body mask from segmentation result.
    Combines torso + arms so the garment blends naturally over all of them.
    """
    parts = seg_result.body_parts
    mask = np.zeros_like(list(parts.values())[0], dtype=np.uint8)

    for part_name in ("torso", "left_arm", "right_arm", "neck"):
        if part_name in parts:
            mask = np.maximum(mask, parts[part_name])

    # Dilate slightly so garment edges don't clip at body boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# ---------------------------------------------------------------------------
# Convenience function (stateless, for simple one-shot use)
# ---------------------------------------------------------------------------

def run_tryon(
    person_image: np.ndarray,
    garment_image: np.ndarray,
    garment_category: str,
    blend_alpha: float = 0.92,
    shoulder_scale: float = 1.0,
) -> TryOnResult:
    """
    One-shot try-on without managing an engine instance.
    Creates and immediately releases the engine.
    Prefer TryOnEngine() directly if processing multiple images.

    Args:
        person_image:     BGR person photo
        garment_image:    BGR/BGRA garment image
        garment_category: 'tshirt', 'shirt', or 'jacket'
        blend_alpha:      Garment opacity [0..1]
        shoulder_scale:   Fit width multiplier

    Returns:
        TryOnResult
    """
    engine = TryOnEngine()
    try:
        return engine.run(
            person_image, garment_image, garment_category,
            blend_alpha=blend_alpha,
            shoulder_scale=shoulder_scale
        )
    finally:
        engine.release()