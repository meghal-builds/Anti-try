"""Measurement Inference Module - HEIGHT-BASED CALIBRATION

Replaces the old circular logic (AVERAGE_SHOULDER_CM = 40) with real
per-person calibration using the user's actual height.

How it works:
  1. User provides their real height in cm (e.g., 175)
  2. MediaPipe detects nose (head proxy) and ankles (feet proxy)
  3. pixels_per_cm = body_height_px / user_height_cm
  4. All measurements (shoulder, chest, torso) use this *real* ratio
"""

from typing import Tuple, List
import numpy as np

from src.models import Measurements, SegmentationResult, PoseResult, Keypoint

# Anatomical ratio: flat chest width → chest circumference
# This is a valid medical approximation (circumference ≈ 2.35× front width)
SHOULDER_TO_CIRCUMFERENCE = 2.35

# Torso multiplier to account for spine curvature (front view underestimates depth)
TORSO_MULTIPLIER = 1.13


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def infer_measurements(
    pose_result: PoseResult,
    seg_result: SegmentationResult,
    image_height: int = 0,
    user_height_cm: float = 0.0
) -> Measurements:
    """Infer body measurements from pose and segmentation.

    Args:
        pose_result:    Pose detection result with keypoints
        seg_result:     Body segmentation result
        image_height:   Height of the image in pixels (legacy, used for fallback)
        user_height_cm: User's real height in cm. When provided (> 0), enables
                        accurate calibration. Without it, falls back to rough
                        estimation with low confidence.

    Returns:
        Measurements with shoulder, chest, torso, and calibration info
    """
    if not pose_result or not seg_result:
        raise ValueError("Both pose and segmentation results required")

    # ── Pixel measurements from keypoints (always real) ──────────────
    shoulder_width_px = pose_result.shoulder_width_px
    torso_length_px = calculate_torso_length(pose_result.keypoints)
    chest_width_px = shoulder_width_px * 1.15  # ribcage slightly wider than shoulders

    # ── Calibrate pixels_per_cm ──────────────────────────────────────
    if user_height_cm > 0:
        pixels_per_cm, calibration_method = _calibrate_from_height(
            user_height_cm, pose_result.keypoints, image_height
        )
    else:
        # Fallback: rough estimation (the old "fake" mode, kept for backward compat)
        pixels_per_cm = _estimate_pixels_per_cm_fallback(shoulder_width_px, image_height)
        calibration_method = 'estimated'

    # ── Convert pixels → cm ──────────────────────────────────────────
    shoulder_width_cm = shoulder_width_px / pixels_per_cm
    torso_length_cm = (torso_length_px / pixels_per_cm) * TORSO_MULTIPLIER
    chest_circumference_cm = chest_width_px / pixels_per_cm * SHOULDER_TO_CIRCUMFERENCE

    # ── Confidence ───────────────────────────────────────────────────
    confidence = _calculate_confidence(
        pose_result, shoulder_width_cm, chest_circumference_cm, calibration_method
    )

    measurements = Measurements(
        shoulder_width_cm=round(shoulder_width_cm, 2),
        chest_circumference_cm=round(chest_circumference_cm, 2),
        torso_length_cm=round(torso_length_cm, 2),
        source='calibrated' if calibration_method == 'height' else 'estimated',
        confidence=round(confidence, 2),
        calibration_method=calibration_method,
        user_height_cm=user_height_cm
    )

    return measurements


# ============================================================================
# HEIGHT-BASED CALIBRATION (THE REAL FIX)
# ============================================================================

def _calibrate_from_height(
    user_height_cm: float,
    keypoints: List[Keypoint],
    image_height: int = 0
) -> Tuple[float, str]:
    """Calculate pixels_per_cm from the user's real height.

    Strategy:
      - Measure body height in pixels: from nose (top) to ankles (bottom)
      - Note: nose is ~6% below the top of the head, so we adjust upward
      - pixels_per_cm = adjusted_body_height_px / user_height_cm

    Returns:
        (pixels_per_cm, calibration_method)
    """
    # Find the highest point (nose or top of head proxy)
    nose = next((kp for kp in keypoints if kp.name == 'nose'), None)

    # Find the lowest points (ankles)
    left_ankle = next((kp for kp in keypoints if kp.name == 'left_ankle'), None)
    right_ankle = next((kp for kp in keypoints if kp.name == 'right_ankle'), None)

    # We need at least nose + one ankle for height calibration
    if nose is None or (left_ankle is None and right_ankle is None):
        # Can't calibrate from height — missing keypoints, use shoulder-based fallback
        # but still try to get something reasonable from height + shoulders
        return _calibrate_from_height_shoulder_fallback(
            user_height_cm, keypoints, image_height
        )

    # Calculate body height in pixels
    top_y = nose.y  # nose y-coordinate

    if left_ankle is not None and right_ankle is not None:
        bottom_y = (left_ankle.y + right_ankle.y) / 2
    elif left_ankle is not None:
        bottom_y = left_ankle.y
    else:
        bottom_y = right_ankle.y

    nose_to_ankle_px = abs(bottom_y - top_y)

    if nose_to_ankle_px < 10:
        # Something is very wrong — keypoints are overlapping
        return _estimate_pixels_per_cm_fallback(0, image_height), 'estimated'

    # Nose is roughly 6% below the actual top of the head
    # nose-to-ankle ≈ 94% of full height
    # Also ankles are slightly above the ground (sole of foot ≈ 2% of height)
    # So nose-to-ankle ≈ 92% of full standing height
    full_body_height_px = nose_to_ankle_px / 0.92

    pixels_per_cm = full_body_height_px / user_height_cm

    # Sanity check: pixels_per_cm should be reasonable (1-30 range for typical photos)
    if pixels_per_cm < 1.0 or pixels_per_cm > 30.0:
        return _estimate_pixels_per_cm_fallback(0, image_height), 'estimated'

    return pixels_per_cm, 'height'


def _calibrate_from_height_shoulder_fallback(
    user_height_cm: float,
    keypoints: List[Keypoint],
    image_height: int
) -> Tuple[float, str]:
    """Fallback when ankles not visible: use shoulder-hip span + height ratio.

    Average human proportions:
      - Shoulder to hip ≈ 30% of total height
      - If we can detect shoulder-to-hip pixels, we can calibrate from that
    """
    left_shoulder = next((kp for kp in keypoints if kp.name == 'left_shoulder'), None)
    right_shoulder = next((kp for kp in keypoints if kp.name == 'right_shoulder'), None)
    left_hip = next((kp for kp in keypoints if kp.name == 'left_hip'), None)
    right_hip = next((kp for kp in keypoints if kp.name == 'right_hip'), None)

    if left_shoulder and right_shoulder and left_hip and right_hip:
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        torso_px = abs(hip_y - shoulder_y)

        if torso_px > 10:
            # Shoulder-to-hip is ~30% of total height
            estimated_full_height_px = torso_px / 0.30
            pixels_per_cm = estimated_full_height_px / user_height_cm

            if 1.0 <= pixels_per_cm <= 30.0:
                return pixels_per_cm, 'height'

    # Final fallback
    return _estimate_pixels_per_cm_fallback(0, image_height), 'estimated'


# ============================================================================
# LEGACY FALLBACK (low confidence, marked as "estimated")
# ============================================================================

def _estimate_pixels_per_cm_fallback(
    shoulder_width_px: float,
    image_height: int = 0
) -> float:
    """Legacy estimation when no user height is provided.

    This is the OLD method — kept only for backward compatibility.
    Results from this path are marked as 'estimated' with low confidence.
    """
    if image_height > 0 and shoulder_width_px > 0:
        expected_shoulder_px = image_height * 0.22
        expected_px_per_cm = expected_shoulder_px / 40.0
        actual_ratio = shoulder_width_px / expected_shoulder_px
        pixels_per_cm = expected_px_per_cm * actual_ratio
        return max(3.0, min(15.0, pixels_per_cm))

    EXPECTED_SHOULDER_PX = 280
    EXPECTED_PIXELS_PER_CM = 7.0

    if shoulder_width_px > 0:
        ratio = EXPECTED_SHOULDER_PX / shoulder_width_px
        adjusted = EXPECTED_PIXELS_PER_CM * ratio
        return max(3.0, min(15.0, adjusted))

    return EXPECTED_PIXELS_PER_CM


# ============================================================================
# TORSO LENGTH
# ============================================================================

def calculate_torso_length(keypoints: List[Keypoint]) -> float:
    """Calculate torso length from keypoints (always real — measured from photo)."""
    left_shoulder = next((kp for kp in keypoints if kp.name == 'left_shoulder'), None)
    right_shoulder = next((kp for kp in keypoints if kp.name == 'right_shoulder'), None)
    left_hip = next((kp for kp in keypoints if kp.name == 'left_hip'), None)
    right_hip = next((kp for kp in keypoints if kp.name == 'right_hip'), None)

    if not (left_shoulder and right_shoulder and left_hip and right_hip):
        return 0.0

    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_y = (left_hip.y + right_hip.y) / 2
    return abs(hip_y - shoulder_y)


# ============================================================================
# CONFIDENCE
# ============================================================================

def _calculate_confidence(
    pose_result: PoseResult,
    shoulder_width_cm: float,
    chest_circumference_cm: float,
    calibration_method: str = 'estimated'
) -> float:
    """Calculate measurement confidence.

    Height-calibrated measurements get much higher confidence than estimated ones.
    """
    if calibration_method == 'height':
        # Start higher for calibrated measurements
        confidence = 0.70
    else:
        # Estimated mode: start low, cap at 0.40
        confidence = 0.25

    # Keypoint detection quality bonus
    keypoint_bonus = min(len(pose_result.keypoints) / 20.0, 0.15)
    confidence += keypoint_bonus

    # Frontal pose bonus
    if pose_result.is_frontal:
        confidence += 0.10

    # Sanity check: measurements in reasonable adult range
    if 30 <= shoulder_width_cm <= 55:
        confidence += 0.05
    if 70 <= chest_circumference_cm <= 130:
        confidence += 0.05

    # Cap based on calibration method
    if calibration_method == 'height':
        return min(confidence, 0.95)
    else:
        return min(confidence, 0.40)


# ============================================================================
# UTILITIES
# ============================================================================

def calculate_measurement_confidence(measurements: Measurements) -> float:
    """Get confidence from measurements."""
    return measurements.confidence


def calculate_measurement_fit(
    user_value: float, size_value: float, tolerance_percent: float = 5.0
) -> float:
    """Calculate fit score for a single measurement."""
    if size_value == 0:
        return 0.0
    difference_percent = abs(user_value - size_value) / size_value * 100
    if difference_percent == 0:
        return 1.0
    max_difference = tolerance_percent * 2
    if difference_percent >= max_difference:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (difference_percent / max_difference)))


def validate_measurements(measurements: Measurements) -> Tuple[bool, str]:
    """Validate that measurements are in reasonable range."""
    errors = []

    if not (25 <= measurements.shoulder_width_cm <= 60):
        errors.append(f"Invalid shoulder width: {measurements.shoulder_width_cm}cm (expected 25-60cm)")

    if not (60 <= measurements.chest_circumference_cm <= 140):
        errors.append(f"Invalid chest circumference: {measurements.chest_circumference_cm}cm (expected 60-140cm)")

    if not (30 <= measurements.torso_length_cm <= 90):
        errors.append(f"Invalid torso length: {measurements.torso_length_cm}cm (expected 30-90cm)")

    is_valid = len(errors) == 0
    error_message = "; ".join(errors) if errors else ""
    return is_valid, error_message


def recalibrate_pixels_per_cm(reference_width_cm: float, measured_width_px: float) -> float:
    """Recalibrate pixels-per-cm ratio from a reference object."""
    return measured_width_px / reference_width_cm


def print_measurement_debug_info(pose_result: PoseResult, measurements: Measurements):
    """Print debug information about measurements."""
    print("\n" + "=" * 70)
    print("MEASUREMENT DEBUG INFO")
    print("=" * 70)
    print(f"Calibration Method: {measurements.calibration_method}")
    if measurements.user_height_cm > 0:
        print(f"User Height: {measurements.user_height_cm} cm")
    print(f"Shoulder Width (px): {pose_result.shoulder_width_px:.2f}")
    print(f"Shoulder Width (cm): {measurements.shoulder_width_cm:.2f}")
    if measurements.shoulder_width_cm > 0:
        print(f"Pixels per cm: {pose_result.shoulder_width_px / measurements.shoulder_width_cm:.2f}")
    print(f"\nChest Circumference (cm): {measurements.chest_circumference_cm:.2f}")
    print(f"Torso Length (cm): {measurements.torso_length_cm:.2f}")
    print(f"Confidence: {measurements.confidence * 100:.1f}%")
    print("=" * 70 + "\n")