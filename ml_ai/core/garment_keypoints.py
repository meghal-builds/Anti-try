"""
Garment Keypoints Module
AI-Based Virtual Try-On and Fit Recommendation System

Defines anchor points on garment images that correspond to body keypoints.
These are used as TPS control points to deform the garment onto the body.

Anchor point coordinate system:
    - Normalized [0.0, 1.0] relative to garment image dimensions
    - (0, 0) = top-left corner of garment image
    - (1, 1) = bottom-right corner

Body keypoint names match MediaPipe output from mediapipe_real.py:
    nose, left_shoulder, right_shoulder,
    left_elbow, right_elbow,
    left_hip, right_hip,
    left_wrist, right_wrist

Usage:
    schema   = get_garment_schema("tshirt")
    src_pts  = schema.get_src_points(garment_w, garment_h)
    dst_pts  = schema.get_dst_points(keypoints, person_image)
    warped   = tps_warp(garment_img, src_pts, dst_pts, output_size)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GarmentAnchor:
    """
    A single anchor point linking a garment pixel location to a body landmark.

    Attributes:
        name:           Human-readable label (e.g. 'left_shoulder')
        garment_uv:     Normalized (u, v) in garment image [0..1, 0..1]
        body_landmark:  Corresponding MediaPipe keypoint name, or None if
                        the anchor is inferred from multiple landmarks
        offset_ratio:   (dx, dy) fractional offset applied to the landmark
                        position — useful to nudge anchor away from exact joint
    """
    name: str
    garment_uv: Tuple[float, float]
    body_landmark: str | None = None
    offset_ratio: Tuple[float, float] = (0.0, 0.0)


@dataclass
class GarmentKeypointSchema:
    """
    Full set of anchor points for one garment category.

    Attributes:
        category:  'tshirt', 'shirt', or 'jacket'
        anchors:   Ordered list of GarmentAnchor definitions
    """
    category: str
    anchors: List[GarmentAnchor] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Point extraction helpers
    # ------------------------------------------------------------------

    def get_src_points(
        self,
        garment_w: int,
        garment_h: int
    ) -> np.ndarray:
        """
        Convert normalized garment UV coords to pixel coordinates.

        Args:
            garment_w: Garment image width in pixels
            garment_h: Garment image height in pixels

        Returns:
            (N, 2) float32 array of [x, y] pixel positions
        """
        pts = []
        for anchor in self.anchors:
            u, v = anchor.garment_uv
            x = u * garment_w
            y = v * garment_h
            pts.append([x, y])
        return np.array(pts, dtype=np.float32)

    def get_dst_points(
        self,
        keypoints: list,
        person_w: int,
        person_h: int,
        shoulder_scale: float = 1.0
    ) -> np.ndarray | None:
        """
        Resolve destination positions from detected body keypoints.

        Args:
            keypoints:      List of Keypoint objects from pose detection
            person_w:       Person image width in pixels
            person_h:       Person image height in pixels
            shoulder_scale: Optional scale multiplier (for fit adjustment)

        Returns:
            (N, 2) float32 array of [x, y] pixel positions,
            or None if a required landmark is missing.
        """
        kp_map = {kp.name: kp for kp in keypoints}

        # Derive useful composite landmarks
        derived = _derive_landmarks(kp_map, person_w, person_h)
        all_landmarks = {**{k: (v.x, v.y) for k, v in kp_map.items()}, **derived}

        pts = []
        for anchor in self.anchors:
            lm_name = anchor.body_landmark
            if lm_name is None or lm_name not in all_landmarks:
                return None  # Required landmark missing

            base_x, base_y = all_landmarks[lm_name]
            dx = anchor.offset_ratio[0] * person_w
            dy = anchor.offset_ratio[1] * person_h

            # Apply shoulder scale around body center
            center_x = person_w / 2
            x = center_x + (base_x - center_x) * shoulder_scale + dx
            y = base_y + dy

            pts.append([x, y])

        return np.array(pts, dtype=np.float32)

    def anchor_names(self) -> List[str]:
        """Return ordered list of anchor names."""
        return [a.name for a in self.anchors]


# ---------------------------------------------------------------------------
# Derived landmark helpers
# ---------------------------------------------------------------------------

def _derive_landmarks(
    kp_map: dict,
    person_w: int,
    person_h: int
) -> Dict[str, Tuple[float, float]]:
    """
    Compute composite/derived body positions from raw keypoints.

    Derived landmarks:
        neck_center     — midpoint between shoulders, shifted up slightly
        torso_center    — midpoint of shoulders and hips
        left_hem_ref    — below left hip, estimating shirt hem
        right_hem_ref   — below right hip
        left_sleeve_end — midpoint of left elbow and wrist
        right_sleeve_end— midpoint of right elbow and wrist
    """
    derived: Dict[str, Tuple[float, float]] = {}

    def _get(name) -> Tuple[float, float] | None:
        kp = kp_map.get(name)
        return (kp.x, kp.y) if kp else None

    ls = _get("left_shoulder")
    rs = _get("right_shoulder")
    lh = _get("left_hip")
    rh = _get("right_hip")
    le = _get("left_elbow")
    re = _get("right_elbow")
    lw = _get("left_wrist")
    rw = _get("right_wrist")

    if ls and rs:
        # Neck center: midpoint of shoulders, nudged upward
        neck_offset = (ls[1] + rs[1]) / 2 * 0.06
        derived["neck_center"] = (
            (ls[0] + rs[0]) / 2,
            (ls[1] + rs[1]) / 2 - neck_offset
        )

    if ls and rs and lh and rh:
        derived["torso_center"] = (
            (ls[0] + rs[0] + lh[0] + rh[0]) / 4,
            (ls[1] + rs[1] + lh[1] + rh[1]) / 4
        )

    if lh:
        # Hem reference: below hip by ~15% of person height
        derived["left_hem_ref"] = (lh[0], lh[1] + person_h * 0.15)

    if rh:
        derived["right_hem_ref"] = (rh[0], rh[1] + person_h * 0.15)

    if le and lw:
        derived["left_sleeve_end"] = (
            (le[0] + lw[0]) / 2,
            (le[1] + lw[1]) / 2
        )

    if re and rw:
        derived["right_sleeve_end"] = (
            (re[0] + rw[0]) / 2,
            (re[1] + rw[1]) / 2
        )

    return derived


# ---------------------------------------------------------------------------
# Garment schemas
# ---------------------------------------------------------------------------

# ── T-SHIRT ─────────────────────────────────────────────────────────────────
TSHIRT_SCHEMA = GarmentKeypointSchema(
    category="tshirt",
    anchors=[
        # Collar / neckline
        GarmentAnchor(
            name="collar_center",
            garment_uv=(0.50, 0.08),
            body_landmark="neck_center"
        ),
        # Shoulders
        GarmentAnchor(
            name="left_shoulder",
            garment_uv=(0.20, 0.12),
            body_landmark="left_shoulder",
            offset_ratio=(-0.01, 0.0)
        ),
        GarmentAnchor(
            name="right_shoulder",
            garment_uv=(0.80, 0.12),
            body_landmark="right_shoulder",
            offset_ratio=(0.01, 0.0)
        ),
        # Sleeve ends (short sleeves — midway down upper arm)
        GarmentAnchor(
            name="left_sleeve_end",
            garment_uv=(0.10, 0.32),
            body_landmark="left_sleeve_end"
        ),
        GarmentAnchor(
            name="right_sleeve_end",
            garment_uv=(0.90, 0.32),
            body_landmark="right_sleeve_end"
        ),
        # Side seams at waist
        GarmentAnchor(
            name="left_side_waist",
            garment_uv=(0.18, 0.65),
            body_landmark="left_hip",
            offset_ratio=(0.0, -0.05)
        ),
        GarmentAnchor(
            name="right_side_waist",
            garment_uv=(0.82, 0.65),
            body_landmark="right_hip",
            offset_ratio=(0.0, -0.05)
        ),
        # Hem corners
        GarmentAnchor(
            name="left_hem",
            garment_uv=(0.18, 0.95),
            body_landmark="left_hem_ref"
        ),
        GarmentAnchor(
            name="right_hem",
            garment_uv=(0.82, 0.95),
            body_landmark="right_hem_ref"
        ),
    ]
)


# ── SHIRT (button-down, longer sleeves) ─────────────────────────────────────
SHIRT_SCHEMA = GarmentKeypointSchema(
    category="shirt",
    anchors=[
        GarmentAnchor(
            name="collar_center",
            garment_uv=(0.50, 0.06),
            body_landmark="neck_center"
        ),
        GarmentAnchor(
            name="left_shoulder",
            garment_uv=(0.18, 0.11),
            body_landmark="left_shoulder",
            offset_ratio=(-0.01, 0.0)
        ),
        GarmentAnchor(
            name="right_shoulder",
            garment_uv=(0.82, 0.11),
            body_landmark="right_shoulder",
            offset_ratio=(0.01, 0.0)
        ),
        # Full sleeves — map to wrist
        GarmentAnchor(
            name="left_cuff",
            garment_uv=(0.04, 0.72),
            body_landmark="left_wrist"
        ),
        GarmentAnchor(
            name="right_cuff",
            garment_uv=(0.96, 0.72),
            body_landmark="right_wrist"
        ),
        # Elbow reference for sleeve curve
        GarmentAnchor(
            name="left_elbow",
            garment_uv=(0.08, 0.44),
            body_landmark="left_elbow"
        ),
        GarmentAnchor(
            name="right_elbow",
            garment_uv=(0.92, 0.44),
            body_landmark="right_elbow"
        ),
        # Side seams
        GarmentAnchor(
            name="left_side_waist",
            garment_uv=(0.16, 0.65),
            body_landmark="left_hip",
            offset_ratio=(0.0, -0.04)
        ),
        GarmentAnchor(
            name="right_side_waist",
            garment_uv=(0.84, 0.65),
            body_landmark="right_hip",
            offset_ratio=(0.0, -0.04)
        ),
        # Hem
        GarmentAnchor(
            name="left_hem",
            garment_uv=(0.16, 0.96),
            body_landmark="left_hem_ref"
        ),
        GarmentAnchor(
            name="right_hem",
            garment_uv=(0.84, 0.96),
            body_landmark="right_hem_ref"
        ),
    ]
)


# ── JACKET ───────────────────────────────────────────────────────────────────
JACKET_SCHEMA = GarmentKeypointSchema(
    category="jacket",
    anchors=[
        # Collar/lapel
        GarmentAnchor(
            name="collar_left",
            garment_uv=(0.42, 0.08),
            body_landmark="neck_center",
            offset_ratio=(-0.02, 0.0)
        ),
        GarmentAnchor(
            name="collar_right",
            garment_uv=(0.58, 0.08),
            body_landmark="neck_center",
            offset_ratio=(0.02, 0.0)
        ),
        # Shoulders (jacket sits wider)
        GarmentAnchor(
            name="left_shoulder",
            garment_uv=(0.15, 0.12),
            body_landmark="left_shoulder",
            offset_ratio=(-0.02, 0.0)
        ),
        GarmentAnchor(
            name="right_shoulder",
            garment_uv=(0.85, 0.12),
            body_landmark="right_shoulder",
            offset_ratio=(0.02, 0.0)
        ),
        # Lapels
        GarmentAnchor(
            name="left_lapel",
            garment_uv=(0.38, 0.22),
            body_landmark="left_shoulder",
            offset_ratio=(0.04, 0.08)
        ),
        GarmentAnchor(
            name="right_lapel",
            garment_uv=(0.62, 0.22),
            body_landmark="right_shoulder",
            offset_ratio=(-0.04, 0.08)
        ),
        # Sleeves — full length
        GarmentAnchor(
            name="left_elbow",
            garment_uv=(0.06, 0.46),
            body_landmark="left_elbow"
        ),
        GarmentAnchor(
            name="right_elbow",
            garment_uv=(0.94, 0.46),
            body_landmark="right_elbow"
        ),
        GarmentAnchor(
            name="left_cuff",
            garment_uv=(0.03, 0.74),
            body_landmark="left_wrist"
        ),
        GarmentAnchor(
            name="right_cuff",
            garment_uv=(0.97, 0.74),
            body_landmark="right_wrist"
        ),
        # Side seams
        GarmentAnchor(
            name="left_side_waist",
            garment_uv=(0.14, 0.62),
            body_landmark="left_hip",
            offset_ratio=(0.0, -0.04)
        ),
        GarmentAnchor(
            name="right_side_waist",
            garment_uv=(0.86, 0.62),
            body_landmark="right_hip",
            offset_ratio=(0.0, -0.04)
        ),
        # Hem
        GarmentAnchor(
            name="left_hem",
            garment_uv=(0.14, 0.95),
            body_landmark="left_hem_ref"
        ),
        GarmentAnchor(
            name="right_hem",
            garment_uv=(0.86, 0.95),
            body_landmark="right_hem_ref"
        ),
    ]
)


# ---------------------------------------------------------------------------
# Registry and public API
# ---------------------------------------------------------------------------

_SCHEMA_REGISTRY: Dict[str, GarmentKeypointSchema] = {
    "tshirt":  TSHIRT_SCHEMA,
    "shirt":   SHIRT_SCHEMA,
    "jacket":  JACKET_SCHEMA,
    # Aliases
    "t-shirt": TSHIRT_SCHEMA,
    "t_shirt": TSHIRT_SCHEMA,
}


def get_garment_schema(category: str) -> GarmentKeypointSchema:
    """
    Return the keypoint schema for a garment category.

    Args:
        category: One of 'tshirt', 'shirt', 'jacket' (case-insensitive)

    Returns:
        GarmentKeypointSchema for that category

    Raises:
        ValueError: if category is not recognised
    """
    key = category.lower().strip()
    if key not in _SCHEMA_REGISTRY:
        supported = sorted({k for k in _SCHEMA_REGISTRY if "-" not in k and "_" not in k[1:]})
        raise ValueError(
            f"Unknown garment category: '{category}'. "
            f"Supported: {supported}"
        )
    return _SCHEMA_REGISTRY[key]


def list_supported_categories() -> List[str]:
    """Return canonical category names (no aliases)."""
    return ["tshirt", "shirt", "jacket"]


def resolve_points(
    schema: GarmentKeypointSchema,
    garment_image: np.ndarray,
    keypoints: list,
    person_w: int,
    person_h: int,
    shoulder_scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    """
    Convenience function: resolve both src and dst points in one call.

    Args:
        schema:         GarmentKeypointSchema for the garment type
        garment_image:  Garment image numpy array
        keypoints:      Detected body keypoints from pose detection
        person_w:       Person image width
        person_h:       Person image height
        shoulder_scale: Scale multiplier for fit adjustment

    Returns:
        (src_points, dst_points) as (N, 2) float32 arrays,
        or (None, None) if required keypoints are missing.
    """
    g_h, g_w = garment_image.shape[:2]
    src_pts = schema.get_src_points(g_w, g_h)
    dst_pts = schema.get_dst_points(keypoints, person_w, person_h, shoulder_scale)

    if dst_pts is None:
        return None, None

    return src_pts, dst_pts