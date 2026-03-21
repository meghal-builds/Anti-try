"""Model Layer - Abstract interfaces and implementations for AI models"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import cv2
import numpy as np

from src.models import PoseResult, SegmentationResult, Keypoint


# ============================================================================
# ABSTRACT BASE CLASSES
# ============================================================================

class SegmentationModel(ABC):
    """Abstract base class for body segmentation models"""
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> SegmentationResult:
        """
        Segment body in image
        
        Args:
            image: Input image (H x W x 3) in BGR format
            
        Returns:
            SegmentationResult with body mask and confidence
        """
        pass


class PoseModel(ABC):
    """Abstract base class for pose detection models"""
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> PoseResult:
        """
        Detect pose in image
        
        Args:
            image: Input image (H x W x 3) in BGR format
            
        Returns:
            PoseResult with keypoints and pose info
        """
        pass


# ============================================================================
# SEGMENTATION MODEL IMPLEMENTATIONS
# ============================================================================

class UNetSegmentationModel(SegmentationModel):
    """U-Net based body segmentation — enhanced with MediaPipe Selfie Segmentation."""
    
    def __init__(self, weights_path: Optional[str] = None, input_size: tuple = (512, 512)):
        """
        Initialize segmentation model.
        
        Uses MediaPipe Selfie Segmentation for real body detection,
        with fallback to placeholder rectangles if MediaPipe is unavailable.
        
        Args:
            weights_path: Path to pretrained weights (unused, kept for API compat)
            input_size: Input image size (unused, kept for API compat)
        """
        self.weights_path = weights_path
        self.input_size = input_size
        self.model_name = "UNet"
        self._selfie_seg = None
        self._init_selfie_segmentation()
    
    def _init_selfie_segmentation(self):
        """Try to initialize MediaPipe Selfie Segmentation."""
        try:
            import mediapipe as mp
            self._selfie_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1  # Landscape model — better for full body
            )
        except Exception:
            self._selfie_seg = None
    
    def predict(self, image: np.ndarray) -> SegmentationResult:
        """Predict body segmentation using real MediaPipe or fallback."""
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be numpy array")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be (H x W x 3)")
        
        height, width = image.shape[:2]
        
        # Try real MediaPipe Selfie Segmentation
        if self._selfie_seg is not None:
            try:
                return self._predict_with_mediapipe(image, height, width)
            except Exception:
                pass
        
        # Fallback to placeholder
        return self._predict_placeholder(image, height, width)
    
    def _predict_with_mediapipe(self, image: np.ndarray, height: int, width: int) -> SegmentationResult:
        """Run real MediaPipe Selfie Segmentation."""
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._selfie_seg.process(image_rgb)
        
        if results.segmentation_mask is None:
            return self._predict_placeholder(image, height, width)
        
        # MediaPipe returns float32 mask [0..1]
        raw_mask = results.segmentation_mask
        body_binary = (raw_mask > 0.5).astype(np.uint8)
        
        # If body is too small, likely a bad detection
        body_fraction = np.sum(body_binary) / (height * width)
        if body_fraction < 0.02:
            return self._predict_placeholder(image, height, width)
        
        # Derive body-part masks using pose keypoints for region splitting
        body_parts = self._derive_body_parts_from_mask(body_binary, height, width)
        
        # Calculate torso percentage
        torso_mask = body_parts.get('torso', np.zeros_like(body_binary))
        total_body_pixels = max(np.sum(body_binary > 0), 1)
        torso_pixels = np.sum(torso_mask > 0)
        torso_percentage = (torso_pixels / total_body_pixels * 100)
        
        # Confidence based on body area fraction (good range: 10-60% of image)
        if 0.08 <= body_fraction <= 0.65:
            confidence = 0.90
        elif 0.04 <= body_fraction <= 0.80:
            confidence = 0.80
        else:
            confidence = 0.70
        
        return SegmentationResult(
            mask=body_binary,
            body_parts=body_parts,
            confidence=confidence,
            torso_percentage=torso_percentage,
            warnings=[]
        )
    
    def _derive_body_parts_from_mask(
        self, body_mask: np.ndarray, height: int, width: int
    ) -> Dict:
        """
        Derive torso/arm/neck regions from the full body mask.
        
        Uses geometric heuristics based on body proportions:
            - Neck:  top 15% of body bounding box
            - Torso: middle 50% of body bbox, central 60% width
            - Arms:  sides of the torso region
        """
        
        # Find body bounding box
        coords = np.argwhere(body_mask > 0)
        if len(coords) == 0:
            return self._empty_parts(height, width)
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        body_h = y_max - y_min
        body_w = x_max - x_min
        body_cx = (x_min + x_max) // 2
        
        if body_h < 10 or body_w < 10:
            return self._empty_parts(height, width)
        
        # Region boundaries (as fractions of body bbox height)
        neck_bottom = y_min + int(body_h * 0.15)
        torso_top = neck_bottom
        torso_bottom = y_min + int(body_h * 0.60)
        
        # Torso width: central 60% of body width
        torso_half_w = int(body_w * 0.30)
        torso_left = max(0, body_cx - torso_half_w)
        torso_right = min(width, body_cx + torso_half_w)
        
        # Build part masks (intersected with body mask)
        neck_mask = np.zeros((height, width), dtype=np.uint8)
        neck_mask[y_min:neck_bottom, :] = 1
        neck_mask = (neck_mask & body_mask).astype(np.uint8)
        
        torso_mask = np.zeros((height, width), dtype=np.uint8)
        torso_mask[torso_top:torso_bottom, torso_left:torso_right] = 1
        torso_mask = (torso_mask & body_mask).astype(np.uint8)
        
        left_arm_mask = np.zeros((height, width), dtype=np.uint8)
        left_arm_mask[torso_top:torso_bottom, :torso_left] = 1
        left_arm_mask = (left_arm_mask & body_mask).astype(np.uint8)
        
        right_arm_mask = np.zeros((height, width), dtype=np.uint8)
        right_arm_mask[torso_top:torso_bottom, torso_right:] = 1
        right_arm_mask = (right_arm_mask & body_mask).astype(np.uint8)
        
        return {
            'torso': torso_mask,
            'left_arm': left_arm_mask,
            'right_arm': right_arm_mask,
            'neck': neck_mask,
        }
    
    def _empty_parts(self, height: int, width: int) -> Dict:
        """Return empty body-part masks."""
        empty = np.zeros((height, width), dtype=np.uint8)
        return {
            'torso': empty.copy(),
            'left_arm': empty.copy(),
            'right_arm': empty.copy(),
            'neck': empty.copy(),
        }
    
    def _predict_placeholder(self, image: np.ndarray, height: int, width: int) -> SegmentationResult:
        """Fallback: generate placeholder segmentation mask (original method)."""
        mask = self._generate_placeholder_mask(height, width)
        body_parts = self._extract_body_parts(mask, height, width)
        
        torso_mask = body_parts.get('torso', np.zeros_like(mask))
        total_pixels = max(np.sum(mask > 0), 1)
        torso_pixels = np.sum(torso_mask > 0)
        torso_percentage = (torso_pixels / total_pixels * 100)
        
        return SegmentationResult(
            mask=mask,
            body_parts=body_parts,
            confidence=0.70,
            torso_percentage=torso_percentage,
            warnings=["Using placeholder segmentation — MediaPipe unavailable"]
        )
    
    def _generate_placeholder_mask(self, height: int, width: int) -> np.ndarray:
        """Generate placeholder segmentation mask"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Torso
        torso_top = height // 4
        torso_bottom = 3 * height // 4
        torso_left = width // 3
        torso_right = 2 * width // 3
        mask[torso_top:torso_bottom, torso_left:torso_right] = 1
        
        # Arms
        arm_top = height // 3
        arm_bottom = 2 * height // 3
        mask[arm_top:arm_bottom, torso_left - width // 8:torso_left] = 2
        mask[arm_top:arm_bottom, torso_right:torso_right + width // 8] = 3
        
        # Neck
        neck_bottom = torso_top
        neck_top = torso_top - height // 10
        mask[neck_top:neck_bottom, torso_left:torso_right] = 4
        
        return mask
    
    def _extract_body_parts(self, mask: np.ndarray, height: int, width: int) -> Dict:
        """Extract body part masks"""
        return {
            'torso': (mask == 1).astype(np.uint8),
            'left_arm': (mask == 2).astype(np.uint8),
            'right_arm': (mask == 3).astype(np.uint8),
            'neck': (mask == 4).astype(np.uint8),
        }


class DeepLabSegmentationModel(SegmentationModel):
    """DeepLab based body segmentation (more detailed than U-Net)"""
    
    def __init__(self, weights_path: Optional[str] = None, input_size: tuple = (512, 512)):
        """
        Initialize DeepLab model
        
        Args:
            weights_path: Path to pretrained weights (optional)
            input_size: Input image size (height, width)
        """
        self.weights_path = weights_path
        self.input_size = input_size
        self.model_name = "DeepLab"
    
    def predict(self, image: np.ndarray) -> SegmentationResult:
        """Predict body segmentation with more detail"""
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be numpy array")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be (H x W x 3)")
        
        # Generate placeholder mask with more body parts
        height, width = image.shape[:2]
        mask = self._generate_placeholder_mask(height, width)
        
        # Extract body parts
        body_parts = self._extract_body_parts(mask, height, width)
        
        # Calculate torso percentage
        torso_mask = body_parts.get('torso', np.zeros_like(mask))
        total_pixels = np.sum(mask > 0)
        torso_pixels = np.sum(torso_mask > 0)
        torso_percentage = (torso_pixels / total_pixels * 100) if total_pixels > 0 else 0
        
        return SegmentationResult(
            mask=mask,
            body_parts=body_parts,
            confidence=0.90,
            torso_percentage=torso_percentage,
            warnings=[]
        )
    
    def _generate_placeholder_mask(self, height: int, width: int) -> np.ndarray:
        """Generate detailed placeholder mask"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Torso
        torso_top = height // 4
        torso_bottom = 3 * height // 4
        torso_left = width // 3
        torso_right = 2 * width // 3
        mask[torso_top:torso_bottom, torso_left:torso_right] = 1
        
        # Arms
        arm_top = height // 3
        arm_bottom = 2 * height // 3
        mask[arm_top:arm_bottom, torso_left - width // 8:torso_left] = 2
        mask[arm_top:arm_bottom, torso_right:torso_right + width // 8] = 3
        
        # Neck
        neck_bottom = torso_top
        neck_top = torso_top - height // 10
        mask[neck_top:neck_bottom, torso_left:torso_right] = 4
        
        # Head
        head_bottom = neck_top
        head_top = max(0, neck_top - height // 8)
        head_left = torso_left + (torso_right - torso_left) // 4
        head_right = torso_right - (torso_right - torso_left) // 4
        mask[head_top:head_bottom, head_left:head_right] = 5
        
        return mask
    
    def _extract_body_parts(self, mask: np.ndarray, height: int, width: int) -> Dict:
        """Extract body part masks"""
        return {
            'torso': (mask == 1).astype(np.uint8),
            'left_arm': (mask == 2).astype(np.uint8),
            'right_arm': (mask == 3).astype(np.uint8),
            'neck': (mask == 4).astype(np.uint8),
            'head': (mask == 5).astype(np.uint8),
        }


# ============================================================================
# POSE MODEL IMPLEMENTATIONS
# ============================================================================

class MediaPipePoseModel(PoseModel):
    """MediaPipe based pose detection (9 keypoints)"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize MediaPipe pose model
        
        Args:
            confidence_threshold: Minimum confidence for keypoints
        """
        self.confidence_threshold = confidence_threshold
        self.model_name = "MediaPipe"
    
    def predict(self, image: np.ndarray) -> PoseResult:
        """Detect pose using MediaPipe"""
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be numpy array")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be (H x W x 3)")
        
        # Generate placeholder keypoints
        height, width = image.shape[:2]
        keypoints = self._generate_keypoints(height, width)
        
        # Calculate shoulder width
        left_shoulder = next((k for k in keypoints if k.name == 'left_shoulder'), None)
        right_shoulder = next((k for k in keypoints if k.name == 'right_shoulder'), None)
        
        if left_shoulder and right_shoulder:
            shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        else:
            shoulder_width = width // 3
        
        # Calculate shoulder angle
        shoulder_angle = 0.0
        
        return PoseResult(
            keypoints=keypoints,
            shoulder_width_px=shoulder_width,
            shoulder_angle_degrees=shoulder_angle,
            is_frontal=True,
            warnings=[]
        )
    
    def _generate_keypoints(self, height: int, width: int):
        """Generate placeholder keypoints"""
        return [
            Keypoint('nose', width // 2, height // 6, 0.9),
            Keypoint('neck', width // 2, height // 4, 0.88),
            Keypoint('left_shoulder', width // 3, height // 3, 0.85),
            Keypoint('right_shoulder', 2 * width // 3, height // 3, 0.85),
            Keypoint('left_elbow', width // 4, height // 2, 0.8),
            Keypoint('right_elbow', 3 * width // 4, height // 2, 0.8),
            Keypoint('left_wrist', width // 8, 2 * height // 3, 0.75),
            Keypoint('right_wrist', 7 * width // 8, 2 * height // 3, 0.75),
            Keypoint('left_hip', width // 3, 2 * height // 3, 0.8),
            Keypoint('right_hip', 2 * width // 3, 2 * height // 3, 0.8),
        ]


class OpenPosePoseModel(PoseModel):
    """OpenPose based pose detection (11 keypoints)"""
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize OpenPose model
        
        Args:
            model_path: Path to OpenPose model files
            confidence_threshold: Minimum confidence for keypoints
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model_name = "OpenPose"
    
    def predict(self, image: np.ndarray) -> PoseResult:
        """Detect pose using OpenPose"""
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be numpy array")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be (H x W x 3)")
        
        # Generate placeholder keypoints (more than MediaPipe)
        height, width = image.shape[:2]
        keypoints = self._generate_keypoints(height, width)
        
        # Calculate shoulder width
        left_shoulder = next((k for k in keypoints if k.name == 'left_shoulder'), None)
        right_shoulder = next((k for k in keypoints if k.name == 'right_shoulder'), None)
        
        if left_shoulder and right_shoulder:
            shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        else:
            shoulder_width = width // 3
        
        # Calculate shoulder angle
        shoulder_angle = 0.0
        
        return PoseResult(
            keypoints=keypoints,
            shoulder_width_px=shoulder_width,
            shoulder_angle_degrees=shoulder_angle,
            is_frontal=True,
            warnings=[]
        )
    
    def _generate_keypoints(self, height: int, width: int):
        """Generate placeholder keypoints (11 total)"""
        return [
            Keypoint('nose', width // 2, height // 6, 0.9),
            Keypoint('neck', width // 2, height // 4, 0.88),
            Keypoint('left_shoulder', width // 3, height // 3, 0.85),
            Keypoint('right_shoulder', 2 * width // 3, height // 3, 0.85),
            Keypoint('left_elbow', width // 4, height // 2, 0.8),
            Keypoint('right_elbow', 3 * width // 4, height // 2, 0.8),
            Keypoint('left_wrist', width // 8, 2 * height // 3, 0.75),
            Keypoint('right_wrist', 7 * width // 8, 2 * height // 3, 0.75),
            Keypoint('left_hip', width // 3, 2 * height // 3, 0.8),
            Keypoint('right_hip', 2 * width // 3, 2 * height // 3, 0.8),
            Keypoint('pelvis', width // 2, 3 * height // 4, 0.82),
        ]


# ============================================================================
# MODEL LOADING FUNCTION
# ============================================================================

def load_models(config_path: str = "database/config/models.json"):
    """
    Load models based on configuration
    
    Args:
        config_path: Path to model configuration JSON
        
    Returns:
        Tuple of (segmentation_model, pose_model)
    """
    import json
    import os
    
    if not os.path.exists(config_path):
        if config_path == "database/config/models.json" and os.path.exists("config/models.json"):
            config_path = "config/models.json"
        elif config_path == "config/models.json" and os.path.exists("database/config/models.json"):
            config_path = "database/config/models.json"
        else:
            raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load segmentation model
    seg_type = config['segmentation']['model_type'].lower()
    
    if seg_type == 'unet':
        seg_model = UNetSegmentationModel()
    elif seg_type == 'deeplab':
        seg_model = DeepLabSegmentationModel()
    else:
        raise ValueError(f"Unknown segmentation model: {seg_type}")
    
    # Load pose model
    pose_type = config['pose']['model_type'].lower()
    
    if pose_type == 'mediapipe':
        pose_model = MediaPipePoseModel()
    elif pose_type == 'openpose':
        pose_model = OpenPosePoseModel()
    else:
        raise ValueError(f"Unknown pose model: {pose_type}")
    
    return seg_model, pose_model
