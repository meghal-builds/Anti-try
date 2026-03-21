"""Real MediaPipe Implementation for Pose Detection"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List

from src.models import PoseResult, Keypoint


class RealMediaPipePose:
    """Real MediaPipe Pose Detection"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        self.KEYPOINT_NAMES = {
            0: 'nose',
            11: 'left_shoulder',
            12: 'right_shoulder',
            13: 'left_elbow',
            14: 'right_elbow',
            23: 'left_hip',
            24: 'right_hip',
            25: 'left_knee',
            26: 'right_knee',
            27: 'left_ankle',
            28: 'right_ankle',
        }
    
    def detect_pose(self, image: np.ndarray) -> PoseResult:
        """Detect pose using real MediaPipe with retry on failure."""
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be numpy array")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        results = self.pose.process(image_rgb)
        
        # Retry with contrast enhancement if first attempt fails
        if results.pose_landmarks is None:
            enhanced = self._enhance_for_retry(image)
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            results = self.pose.process(enhanced_rgb)
        
        if results.pose_landmarks is None:
            raise RuntimeError("No pose detected in image")
        
        keypoints = self._extract_keypoints(results, height, width)
        left_shoulder = next((kp for kp in keypoints if kp.name == 'left_shoulder'), None)
        right_shoulder = next((kp for kp in keypoints if kp.name == 'right_shoulder'), None)
        
        shoulder_width_px = 0
        if left_shoulder and right_shoulder:
            shoulder_width_px = abs(right_shoulder.x - left_shoulder.x)
        
        is_frontal = self._is_frontal_pose(keypoints)
        warnings = self._check_pose_quality(keypoints, is_frontal)
        shoulder_angle = self._calculate_shoulder_angle(keypoints)
        
        # Use all required fields
        return PoseResult(
            keypoints=keypoints,
            shoulder_width_px=shoulder_width_px,
            is_frontal=is_frontal,
            shoulder_angle_degrees=shoulder_angle,
            warnings=warnings
        )
    
    def _extract_keypoints(self, results, height: int, width: int) -> List[Keypoint]:
        """Extract keypoints from MediaPipe results"""
        keypoints = []
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx not in self.KEYPOINT_NAMES:
                continue
            
            name = self.KEYPOINT_NAMES[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            confidence = landmark.visibility
            
            if confidence > 0.15:
                keypoint = Keypoint(
                    name=name,
                    x=x,
                    y=y,
                    confidence=float(confidence)
                )
                keypoints.append(keypoint)
        
        return keypoints
    
    def _calculate_shoulder_angle(self, keypoints: List[Keypoint]) -> float:
        """Calculate shoulder angle"""
        left_shoulder = next((kp for kp in keypoints if kp.name == 'left_shoulder'), None)
        right_shoulder = next((kp for kp in keypoints if kp.name == 'right_shoulder'), None)
        
        if not (left_shoulder and right_shoulder):
            return 0.0
        
        height_diff = abs(left_shoulder.y - right_shoulder.y)
        width_diff = abs(right_shoulder.x - left_shoulder.x)
        
        if width_diff == 0:
            return 90.0 if height_diff > 0 else 0.0
        
        angle = np.degrees(np.arctan(height_diff / width_diff))
        return float(angle)
    
    def _is_frontal_pose(self, keypoints: List[Keypoint]) -> bool:
        """Check if pose is frontal"""
        left_shoulder = next((kp for kp in keypoints if kp.name == 'left_shoulder'), None)
        right_shoulder = next((kp for kp in keypoints if kp.name == 'right_shoulder'), None)
        nose = next((kp for kp in keypoints if kp.name == 'nose'), None)
        
        if not (left_shoulder and right_shoulder and nose):
            return False
        
        shoulder_height_diff = abs(left_shoulder.y - right_shoulder.y)
        shoulder_distance = abs(right_shoulder.x - left_shoulder.x)
        is_level = shoulder_height_diff < shoulder_distance * 0.15
        nose_between = (left_shoulder.x < nose.x < right_shoulder.x) or \
                      (right_shoulder.x < nose.x < left_shoulder.x)
        
        return is_level and nose_between
    
    def _check_pose_quality(self, keypoints: List[Keypoint], is_frontal: bool) -> List[str]:
        """Check pose quality"""
        warnings = []
        critical = ['left_shoulder', 'right_shoulder', 'nose']
        visible = sum(1 for kp in keypoints if kp.name in critical)
        
        if visible < 3:
            warnings.append(f"Only {visible}/3 critical keypoints visible")
        if not is_frontal:
            warnings.append("Person not facing camera")
        return warnings
    
    def _enhance_for_retry(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast enhancement before retrying pose detection."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        merged = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    def release(self):
        """Release resources"""
        self.pose.close()


def create_real_pose_detector() -> RealMediaPipePose:
    """Factory function"""
    return RealMediaPipePose()