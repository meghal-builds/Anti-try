"""Image validation module"""

import os
from typing import Tuple

import cv2
import numpy as np

from src.models import ValidationResult


SUPPORTED_FORMATS = {'jpg', 'jpeg', 'png', 'webp', 'heic', 'heif'}
MAX_FILE_SIZE_MB = 10
MIN_RESOLUTION = (256, 256)
BRIGHTNESS_MIN = 15
BRIGHTNESS_MAX = 240


def validate_image(image_path: str) -> ValidationResult:
    """Validate image"""
    errors = []
    warnings = []
    
    # Check file exists
    if not os.path.exists(image_path):
        errors.append(f"File not found: {image_path}")
        return ValidationResult(is_valid=False, errors=errors)
    
    # Check format
    if not validate_format(image_path):
        errors.append(f"Unsupported format")
    
    # Check file size
    if not validate_file_size(image_path):
        errors.append(f"File too large (max {MAX_FILE_SIZE_MB}MB)")
    
    # Check resolution
    if not validate_resolution(image_path):
        errors.append(f"Resolution too small (min {MIN_RESOLUTION[0]}x{MIN_RESOLUTION[1]})")
    
    # Check aspect ratio
    ar_warning = validate_aspect_ratio(image_path)
    if ar_warning:
        warnings.append(ar_warning)
    
    # Check lighting
    if not validate_lighting(image_path):
        warnings.append("Poor lighting detected")
    
    # Check corruption
    if is_image_corrupted(image_path):
        errors.append("Image appears to be corrupted")
    
    is_valid = len(errors) == 0
    
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings
    )


def validate_format(image_path: str) -> bool:
    """Validate image format"""
    ext = image_path.split('.')[-1].lower()
    return ext in SUPPORTED_FORMATS


def validate_file_size(image_path: str) -> bool:
    """Validate file size"""
    size_mb = os.path.getsize(image_path) / (1024 * 1024)
    return size_mb <= MAX_FILE_SIZE_MB


def validate_resolution(image_path: str) -> bool:
    """Validate image resolution"""
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    height, width = image.shape[:2]
    return width >= MIN_RESOLUTION[0] and height >= MIN_RESOLUTION[1]


def validate_lighting(image_path: str) -> bool:
    """Check if lighting is adequate"""
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate average brightness
    brightness = np.mean(gray)
    
    return BRIGHTNESS_MIN <= brightness <= BRIGHTNESS_MAX


def validate_aspect_ratio(image_path: str) -> str:
    """Check if aspect ratio is extreme (panoramic, etc.). Returns warning or empty string."""
    image = cv2.imread(image_path)
    if image is None:
        return ""
    
    height, width = image.shape[:2]
    ratio = max(width, height) / max(min(width, height), 1)
    
    if ratio > 3.0:
        return f"Extreme aspect ratio ({ratio:.1f}:1) — consider cropping to a standard photo"
    return ""


def is_image_corrupted(image_path: str) -> bool:
    """Check if image is corrupted"""
    try:
        image = cv2.imread(image_path)
        return image is None
    except Exception:
        return True