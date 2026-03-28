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
    weight: int = 1


@dataclass
class GarmentKeypointSchema:
    category: str
    anchors: List[GarmentAnchor] = field(default_factory=list)

    def get_src_points(self, garment_w: int, garment_h: int) -> np.ndarray:
        pts = []
        for anchor in self.anchors:
            u, v = anchor.garment_uv
            for _ in range(anchor.weight):
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
            
            for _ in range(anchor.weight):
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

    # ── Guarantee viewer-relative coordinates ─────────────────────────
    # For a front-facing image, person's RIGHT shoulder is on the LEFT side of the screen.
    # We must map viewer-left UVs (u < 0.5) to viewer-left body coordinates.
    if ls[0] > rs[0]:
        vr_s, vl_s = ls, rs     # Viewer-Right Shoulder, Viewer-Left Shoulder
        vr_h, vl_h = lh, rh     # Viewer-Right Hip, Viewer-Left Hip
        vr_e, vl_e = le, re     # Viewer-Right Elbow, Viewer-Left Elbow
        vr_w, vl_w = lw, rw     # Viewer-Right Wrist, Viewer-Left Wrist
    else:
        vr_s, vl_s = rs, ls
        vr_h, vl_h = rh, lh
        vr_e, vl_e = re, le
        vr_w, vl_w = rw, lw

    # ── Base metrics ──────────────────────────────────────────────────
    SW  = vr_s[0] - vl_s[0]       # shoulder width px (positive distance)
    MX  = (vr_s[0] + vl_s[0]) / 2 # body center x
    SY  = (vr_s[1] + vl_s[1]) / 2 # shoulder y

    # ── Hips / Waist foundation ──────────────────────────────────────
    if vl_h and vr_h:
        hip_y = (vl_h[1] + vr_h[1]) / 2
        hip_lx, hip_rx = vl_h[0], vr_h[0]
    elif vl_h:
        hip_y  = vl_h[1]
        hip_lx = vl_h[0]
        hip_rx = vr_s[0]
    elif vr_h:
        hip_y  = vr_h[1]
        hip_lx = vl_s[0]
        hip_rx = vr_h[0]
    else:
        hip_y  = SY + SW * 1.46
        hip_lx = vl_s[0]
        hip_rx = vr_s[0]

    # ── Neck / Collar (3-point curve) ────────────────────────────────
    nose = _get("nose")
    neck_y = nose[1] + (SY - nose[1]) * 0.55 if nose else SY - SW * 0.26
    
    d["neck_center"]   = (MX, neck_y)
    d["collar_left"]   = (MX - SW * 0.15, neck_y)  # viewer-left
    d["collar_right"]  = (MX + SW * 0.15, neck_y)  # viewer-right
    d["collar_bottom"] = (MX, neck_y + SW * 0.10)

    # ── Shoulder Tops (Natural slope & reduced width) ────────────────
    # Move INWARD from shoulders toward neck
    d["left_shoulder_dst"]  = (vl_s[0] + SW * 0.25, SY - SW * 0.05)
    d["right_shoulder_dst"] = (vr_s[0] - SW * 0.25, SY - SW * 0.05)
    
    # ── Chest Anchors ────────────────────────────────────────────────
    chest_y = SY + (hip_y - SY) * 0.35
    # Move INWARD from shoulders
    d["left_chest"]   = (vl_s[0] + SW * 0.15, chest_y)
    d["right_chest"]  = (vr_s[0] - SW * 0.15, chest_y)
    d["center_chest"] = (MX, chest_y)

    # ── Constraint Anchors (Anti-Fold) ───────────────────────────────
    armpit_y = SY + (chest_y - SY) * 0.5
    # Armpits sit slightly INWARD from the exact shoulder joint
    d["left_armpit"]  = (vl_s[0] + SW * 0.10, armpit_y)
    d["right_armpit"] = (vr_s[0] - SW * 0.10, armpit_y)
    d["upper_chest_center"] = (MX, neck_y + SW * 0.25)

    upper_side_y = chest_y + (hip_y - chest_y) * 0.15
    # Sides sit INSIDE the shoulder line
    d["upper_side_left"]  = (vl_s[0] + SW * 0.15, upper_side_y)
    d["upper_side_right"] = (vr_s[0] - SW * 0.15, upper_side_y)

    # ── Mid-Edge Anchors (Distributes tension) ───────────────────────
    waist_y = hip_y - SW * 0.10
    dy = (waist_y - upper_side_y) / 3.0
    d["mid_side_left_1"]  = (vl_s[0] + SW * 0.15, upper_side_y + dy)
    d["mid_side_left_2"]  = (vl_s[0] + SW * 0.15, upper_side_y + 2*dy)
    d["mid_side_right_1"] = (vr_s[0] - SW * 0.15, upper_side_y + dy)
    d["mid_side_right_2"] = (vr_s[0] - SW * 0.15, upper_side_y + 2*dy)

    # ── Sleeve Tips ──────────────────────────────────────────────────
    # Sleeves extend OUTWARD from shoulders
    d["left_sleeve_dst"]  = (vl_s[0] - SW * 0.47, SY + SW * 0.30)
    d["right_sleeve_dst"] = (vr_s[0] + SW * 0.47, SY + SW * 0.30)
    
    d["left_sleeve_mid"]  = (vl_s[0] - SW * 0.25, SY + SW * 0.15)
    d["right_sleeve_mid"] = (vr_s[0] + SW * 0.25, SY + SW * 0.15)

    # ── Sleeve Ends (for long sleeves) ───────────────────────────────
    if vl_e:
        sx = vl_s[0] + (vl_e[0] - vl_s[0]) * 0.40
        sy = vl_s[1] + (vl_e[1] - vl_s[1]) * 0.40
        # Nudge outward from line
        d["left_sleeve_end"] = (sx - SW * 0.10, sy)
    else:
        d["left_sleeve_end"] = (vl_s[0] - SW * 0.35, vl_s[1] + SW * 0.25)

    if vr_e:
        sx = vr_s[0] + (vr_e[0] - vr_s[0]) * 0.40
        sy = vr_s[1] + (vr_e[1] - vr_s[1]) * 0.40
        # Nudge outward from line
        d["right_sleeve_end"] = (sx + SW * 0.10, sy)
    else:
        d["right_sleeve_end"] = (vr_s[0] + SW * 0.35, vr_s[1] + SW * 0.25)

    # ── Elbows & Cuffs ───────────────────────────────────────────────
    d["left_elbow"]  = vl_e if vl_e else (vl_s[0] - SW * 0.15, vl_s[1] + SW * 0.77)
    d["right_elbow"] = vr_e if vr_e else (vr_s[0] + SW * 0.15, vr_s[1] + SW * 0.77)
    d["left_cuff"]   = vl_w if vl_w else (vl_s[0] - SW * 0.18, vl_s[1] + SW * 1.50)
    d["right_cuff"]  = vr_w if vr_w else (vr_s[0] + SW * 0.18, vr_s[1] + SW * 1.50)

    # ── Lapels ───────────────────────────────────────────────────────
    d["left_lapel"]  = (MX - SW * 0.20, SY + SW * 0.35)
    d["right_lapel"] = (MX + SW * 0.20, SY + SW * 0.35)

    # ── Waist / Hem (Curve & Contour) ────────────────────────────────
    # Side waist: nudged INWARD by 0.18 SW for contour
    d["left_side_waist"]  = (vl_s[0] + SW * 0.18, hip_y - SW * 0.10)
    d["right_side_waist"] = (vr_s[0] - SW * 0.18, hip_y - SW * 0.10)

    # Hem: Matches shoulder width visually, center drops for vertical curve
    # Left/Right hem should align similarly to the waist but slightly outward
    d["left_hem_ref"]  = (vl_s[0] + SW * 0.12, hip_y + SW * 0.30)
    d["right_hem_ref"] = (vr_s[0] - SW * 0.12, hip_y + SW * 0.30)
    d["center_hem"]    = (MX, hip_y + SW * 0.35)

    return d


# ---------------------------------------------------------------------------
# Garment schemas — anchor UVs match standard flat-lay garment proportions
# ---------------------------------------------------------------------------

TSHIRT_SCHEMA = GarmentKeypointSchema(
    category="tshirt",
    anchors=[
        # Collar curve (HIGH weight) + Collar Lock (HIGH weight)
        GarmentAnchor("collar_left",      (0.38, 0.05), "collar_left", weight=3),
        GarmentAnchor("collar_bottom",    (0.50, 0.12), "collar_bottom", weight=3),
        GarmentAnchor("collar_right",     (0.62, 0.05), "collar_right", weight=3),
        GarmentAnchor("upper_chest_center", (0.50, 0.20), "upper_chest_center", weight=3),
        
        # Shoulder seams (HIGH weight)
        GarmentAnchor("left_shoulder",    (0.18, 0.15), "left_shoulder_dst", weight=3),
        GarmentAnchor("right_shoulder",   (0.82, 0.15), "right_shoulder_dst", weight=3),
        
        # Armpit Anti-Fold Constraint (LOW weight)
        GarmentAnchor("left_armpit",      (0.20, 0.25), "left_armpit", weight=1),
        GarmentAnchor("right_armpit",     (0.80, 0.25), "right_armpit", weight=1),
        
        # Chest anchors (LOW weight - prevents flat sticker look)
        GarmentAnchor("left_chest",       (0.25, 0.40), "left_chest", weight=1),
        GarmentAnchor("center_chest",     (0.50, 0.40), "center_chest", weight=1),
        GarmentAnchor("right_chest",      (0.75, 0.40), "right_chest", weight=1),

        # Side edges & Waist (MEDIUM weight - structural tension)
        GarmentAnchor("upper_side_left",  (0.20, 0.35), "upper_side_left", weight=2),
        GarmentAnchor("upper_side_right", (0.80, 0.35), "upper_side_right", weight=2),
        GarmentAnchor("mid_side_left_1",  (0.20, 0.45), "mid_side_left_1", weight=2),
        GarmentAnchor("mid_side_right_1", (0.80, 0.45), "mid_side_right_1", weight=2),
        GarmentAnchor("mid_side_left_2",  (0.20, 0.55), "mid_side_left_2", weight=2),
        GarmentAnchor("mid_side_right_2", (0.80, 0.55), "mid_side_right_2", weight=2),
        GarmentAnchor("left_side_waist",  (0.20, 0.65), "left_side_waist", weight=2),
        GarmentAnchor("right_side_waist", (0.80, 0.65), "right_side_waist", weight=2),
        
        # Sleeve tips (LOW weight)
        GarmentAnchor("left_sleeve_end",  (0.05, 0.32), "left_sleeve_dst", weight=1),
        GarmentAnchor("left_sleeve_mid",  (0.12, 0.25), "left_sleeve_mid", weight=1),
        GarmentAnchor("right_sleeve_end", (0.95, 0.32), "right_sleeve_dst", weight=1),
        GarmentAnchor("right_sleeve_mid", (0.88, 0.25), "right_sleeve_mid", weight=1),
        
        # Hem corners & center curve (MEDIUM weight)
        GarmentAnchor("left_hem",         (0.15, 0.94), "left_hem_ref", weight=2),
        GarmentAnchor("center_hem",       (0.50, 0.97), "center_hem", weight=2),
        GarmentAnchor("right_hem",        (0.85, 0.94), "right_hem_ref", weight=2),
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