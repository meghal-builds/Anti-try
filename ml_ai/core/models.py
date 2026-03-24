"""Data models for Virtual Try-On System"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ValidationResult:
    """Result of image validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class Keypoint:
    """A detected keypoint (e.g., shoulder, elbow)"""
    name: str
    x: float
    y: float
    confidence: float


@dataclass
class PoseResult:
    """Result of pose detection"""
    keypoints: List[Keypoint]
    shoulder_width_px: float
    shoulder_angle_degrees: float
    is_frontal: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class SegmentationResult:
    """Result of body segmentation"""
    mask: 'np.ndarray'  # Binary mask
    body_parts: Dict[str, 'np.ndarray']  # Masks for each body part
    confidence: float
    torso_percentage: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class Measurements:
    """Body measurements"""
    shoulder_width_cm: float
    chest_circumference_cm: float
    torso_length_cm: float
    source: str  # "calibrated", "estimated", or "manual"
    confidence: float = 0.0
    calibration_method: str = 'none'  # 'height', 'estimated', or 'none'
    user_height_cm: float = 0.0      # user's input height (0 if not provided)


@dataclass
class GarmentMetadata:
    """Garment metadata"""
    id: str
    name: str
    category: str
    brand: str
    size_chart: Dict[str, Dict[str, float]]
    material: str = ""
    price_usd: float = 0.0


@dataclass
class SizeRecommendation:
    """Size recommendation result"""
    size: str
    confidence: float
    fit_scores: Dict[str, float]
    recommended_sizes: List[str] = field(default_factory=list)