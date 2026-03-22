"""
Garment Keypoints Module
AI-Based Virtual Try-On and Fit Recommendation System

Calibrated using real MediaPipe keypoint data:
    SW=278px, torso=404px (1.46xSW), image=1028x1370
    Left shoulder:(659,394), Right:(381,387)
    Left hip:(596,794), Right:(435,796)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GarmentAnchor:
    name: str
    garment_uv: Tuple[float, float]
    body_landmark: str | None = None
    offset_ratio: Tuple[float, float] = (0.0, 0.0)


@dataclass
class GarmentKeypointSchema:
    category: str
    anchors: List[GarmentAnchor] = field(default_factory=list)

    def get_src_points(self, garment_w: int, garment_h: int) -> np.ndarray:
        pts = []
        for anchor in self.anchors:
            u, v = anchor.garment_uv
            pts.append([u * garment_w, v * garment_h])
        return np.array(pts, dtype=np.float32)

    def get_dst_points(
        self,
        keypoints: list,
        person_w: int,
        person_h: int,
        shoulder_scale: float = 1.0
    ) -> np.ndarray | None:
        kp_map = {kp.name: kp for kp in keypoints}
        derived = _derive_all_landmarks(kp_map, person_w, person_h)
        all_lm  = {**{k: (float(v.x), float(v.y)) for k, v in kp_map.items()}, **derived}

        pts = []
        for anchor in self.anchors:
            lm_name = anchor.body_landmark
            if lm_name is None or lm_name not in all_lm:
                return None

            base_x, base_y = all_lm[lm_name]
            dx = anchor.offset_ratio[0] * person_w
            dy = anchor.offset_ratio[1] * person_h

            cx = person_w / 2
            x  = cx + (base_x - cx) * shoulder_scale + dx
            y  = base_y + dy
            pts.append([x, y])

        return np.array(pts, dtype=np.float32)

    def anchor_names(self) -> List[str]:
        return [a.name for a in self.anchors]


# ---------------------------------------------------------------------------
# Landmark derivation — calibrated to real body proportions
# Torso height ≈ 1.46 × SW (verified from keypoint data)
# ---------------------------------------------------------------------------

def _derive_all_landmarks(
    kp_map: dict,
    person_w: int,
    person_h: int
) -> Dict[str, Tuple[float, float]]:
    """
    Derive body landmarks using shoulder width (SW) as base unit.
    All proportions verified against real MediaPipe keypoint coordinates.
    """
    d: Dict[str, Tuple[float, float]] = {}

    def _get(name):
        kp = kp_map.get(name)
        return (float(kp.x), float(kp.y)) if kp else None

    ls   = _get("left_shoulder")
    rs   = _get("right_shoulder")
    lh   = _get("left_hip")
    rh   = _get("right_hip")
    le   = _get("left_elbow")
    re   = _get("right_elbow")
    lw   = _get("left_wrist")
    rw   = _get("right_wrist")

    if not (ls and rs):
        return d

    # ── Base metrics ──────────────────────────────────────────────────
    SW  = abs(ls[0] - rs[0])           # shoulder width px
    MX  = (ls[0] + rs[0]) / 2         # body center x
    SY  = (ls[1] + rs[1]) / 2         # shoulder y

    # ── Neck: at actual collar position ──────────────────────────────
    # Collar should sit at ~75% between nose and shoulder
    # nose=228, shoulder=390 → collar = 228 + (390-228)*0.55 = 317
    # This is ABOVE shoulder Y, so garment top covers the neck/collar area
    nose = _get("nose")
    if nose:
        neck_y = nose[1] + (SY - nose[1]) * 0.55
    else:
        neck_y = SY - SW * 0.26
    d["neck_center"]  = (MX, neck_y)
    d["collar_left"]  = (MX - SW * 0.06, neck_y)
    d["collar_right"] = (MX + SW * 0.06, neck_y)

    # ── Shoulder tops: wider ─────────────────────────────────────────
    d["left_shoulder_dst"]  = (ls[0] + SW * 0.45, SY)
    d["right_shoulder_dst"] = (rs[0] - SW * 0.45, SY)

    # ── Sleeve tips: wider and lower ─────────────────────────────────
    d["left_sleeve_dst"]  = (ls[0] + SW * 0.47, SY + SW * 0.30)
    d["right_sleeve_dst"] = (rs[0] - SW * 0.47, SY + SW * 0.30)

    # ── Sleeve ends: 40% toward elbow + 10% SW outward nudge ─────────
    if le:
        sx = ls[0] + (le[0] - ls[0]) * 0.40
        sy = ls[1] + (le[1] - ls[1]) * 0.40
        d["left_sleeve_end"] = (sx + SW * 0.10, sy)
    else:
        d["left_sleeve_end"] = (ls[0] + SW * 0.35, ls[1] + SW * 0.25)

    if re:
        sx = rs[0] + (re[0] - rs[0]) * 0.40
        sy = rs[1] + (re[1] - rs[1]) * 0.40
        d["right_sleeve_end"] = (sx - SW * 0.10, sy)
    else:
        d["right_sleeve_end"] = (rs[0] - SW * 0.35, rs[1] + SW * 0.25)

    # ── Elbows ────────────────────────────────────────────────────────
    d["left_elbow"]  = le if le else (ls[0] - SW * 0.15, ls[1] + SW * 0.77)
    d["right_elbow"] = re if re else (rs[0] + SW * 0.15, rs[1] + SW * 0.77)

    # ── Cuffs ─────────────────────────────────────────────────────────
    d["left_cuff"]  = lw if lw else (ls[0] - SW * 0.18, ls[1] + SW * 1.50)
    d["right_cuff"] = rw if rw else (rs[0] + SW * 0.18, rs[1] + SW * 1.50)

    # ── Lapels ────────────────────────────────────────────────────────
    d["left_lapel"]  = (MX - SW * 0.20, SY + SW * 0.35)
    d["right_lapel"] = (MX + SW * 0.20, SY + SW * 0.35)

    # ── Hips / waist / hem ───────────────────────────────────────────
    # Verified: torso = 1.46 × SW for this image
    # Use detected hips when available, else estimate at SY + 1.46*SW

    if lh and rh:
        hip_y = (lh[1] + rh[1]) / 2
        hip_lx, hip_rx = lh[0], rh[0]
    elif lh:
        hip_y  = lh[1]
        hip_lx = lh[0]
        hip_rx = rs[0]
    elif rh:
        hip_y  = rh[1]
        hip_lx = ls[0]
        hip_rx = rh[0]
    else:
        # Estimate: 1.46 SW below shoulder line
        hip_y  = SY + SW * 1.46
        hip_lx = ls[0]
        hip_rx = rs[0]

    # Side waist: 0.10 SW above hip, same x as shoulders
    d["left_side_waist"]  = (ls[0], hip_y - SW * 0.10)
    d["right_side_waist"] = (rs[0], hip_y - SW * 0.10)

    # Hem: wide — matches shoulder_dst x positions
    d["left_hem_ref"]  = (ls[0] + SW * 0.45, hip_y + SW * 0.30)
    d["right_hem_ref"] = (rs[0] - SW * 0.45, hip_y + SW * 0.30)

    return d


# ---------------------------------------------------------------------------
# Garment schemas — anchor UVs match standard flat-lay garment proportions
# ---------------------------------------------------------------------------

TSHIRT_SCHEMA = GarmentKeypointSchema(
    category="tshirt",
    anchors=[
        # Collar (v=0.08 on garment) → neck position on body (above shoulders)
        GarmentAnchor("collar_center",    (0.50, 0.08), "neck_center"),
        # Shoulder seams → outer shoulder positions at shoulder Y
        GarmentAnchor("left_shoulder",    (0.20, 0.15), "left_shoulder_dst"),
        GarmentAnchor("right_shoulder",   (0.80, 0.15), "right_shoulder_dst"),
        # Sleeve tips → below and outside shoulder
        GarmentAnchor("left_sleeve_end",  (0.05, 0.32), "left_sleeve_dst"),
        GarmentAnchor("right_sleeve_end", (0.95, 0.32), "right_sleeve_dst"),
        # Hem corners → wide at bottom
        GarmentAnchor("left_hem",         (0.15, 0.94), "left_hem_ref"),
        GarmentAnchor("right_hem",        (0.85, 0.94), "right_hem_ref"),
    ]
)

SHIRT_SCHEMA = GarmentKeypointSchema(
    category="shirt",
    anchors=[
        GarmentAnchor("collar_center",    (0.50, 0.06), "neck_center"),
        GarmentAnchor("left_shoulder",    (0.20, 0.12), "left_shoulder"),
        GarmentAnchor("right_shoulder",   (0.80, 0.12), "right_shoulder"),
        GarmentAnchor("left_elbow",       (0.08, 0.42), "left_elbow"),
        GarmentAnchor("right_elbow",      (0.92, 0.42), "right_elbow"),
        GarmentAnchor("left_cuff",        (0.05, 0.72), "left_cuff"),
        GarmentAnchor("right_cuff",       (0.95, 0.72), "right_cuff"),
        GarmentAnchor("left_side_waist",  (0.18, 0.63), "left_side_waist"),
        GarmentAnchor("right_side_waist", (0.82, 0.63), "right_side_waist"),
        GarmentAnchor("left_hem",         (0.20, 0.93), "left_hem_ref"),
        GarmentAnchor("right_hem",        (0.80, 0.93), "right_hem_ref"),
    ]
)

JACKET_SCHEMA = GarmentKeypointSchema(
    category="jacket",
    anchors=[
        GarmentAnchor("collar_left",      (0.44, 0.07), "collar_left"),
        GarmentAnchor("collar_right",     (0.56, 0.07), "collar_right"),
        GarmentAnchor("left_shoulder",    (0.17, 0.12), "left_shoulder"),
        GarmentAnchor("right_shoulder",   (0.83, 0.12), "right_shoulder"),
        GarmentAnchor("left_lapel",       (0.38, 0.22), "left_lapel"),
        GarmentAnchor("right_lapel",      (0.62, 0.22), "right_lapel"),
        GarmentAnchor("left_elbow",       (0.07, 0.44), "left_elbow"),
        GarmentAnchor("right_elbow",      (0.93, 0.44), "right_elbow"),
        GarmentAnchor("left_cuff",        (0.04, 0.72), "left_cuff"),
        GarmentAnchor("right_cuff",       (0.96, 0.72), "right_cuff"),
        GarmentAnchor("left_side_waist",  (0.16, 0.62), "left_side_waist"),
        GarmentAnchor("right_side_waist", (0.84, 0.62), "right_side_waist"),
        GarmentAnchor("left_hem",         (0.18, 0.93), "left_hem_ref"),
        GarmentAnchor("right_hem",        (0.82, 0.93), "right_hem_ref"),
    ]
)


# ---------------------------------------------------------------------------
# Registry and public API
# ---------------------------------------------------------------------------

_SCHEMA_REGISTRY: Dict[str, GarmentKeypointSchema] = {
    "tshirt":  TSHIRT_SCHEMA,
    "shirt":   SHIRT_SCHEMA,
    "jacket":  JACKET_SCHEMA,
    "t-shirt": TSHIRT_SCHEMA,
    "t_shirt": TSHIRT_SCHEMA,
    "tops":    TSHIRT_SCHEMA,
}


def get_garment_schema(category: str) -> GarmentKeypointSchema:
    key = category.lower().strip()
    if key not in _SCHEMA_REGISTRY:
        raise ValueError(
            f"Unknown garment category: '{category}'. "
            f"Supported: {list_supported_categories()}"
        )
    return _SCHEMA_REGISTRY[key]


def list_supported_categories() -> List[str]:
    return ["tshirt", "shirt", "jacket"]


def resolve_points(
    schema: GarmentKeypointSchema,
    garment_image: np.ndarray,
    keypoints: list,
    person_w: int,
    person_h: int,
    shoulder_scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    g_h, g_w = garment_image.shape[:2]
    src_pts  = schema.get_src_points(g_w, g_h)
    dst_pts  = schema.get_dst_points(keypoints, person_w, person_h, shoulder_scale)
    if dst_pts is None:
        return None, None
    return src_pts, dst_pts