"""Image utility functions"""

import os
from pathlib import Path
from typing import Optional, Tuple
import hashlib
from datetime import datetime
import random
import string

import cv2
import numpy as np
from PIL import Image


def generate_unique_filename(
    prefix: str = "image",
    extension: str = "png"
) -> str:
    """Generate unique filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{prefix}_{timestamp}_{random_suffix}.{extension}"


def load_image(image_path: str, fix_orientation: bool = True) -> Optional[np.ndarray]:
    """Load image using OpenCV, with optional EXIF orientation fix."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    if fix_orientation:
        try:
            from ml_ai.core.image_preprocessor import fix_exif_rotation
            image, _ = fix_exif_rotation(image, image_path)
        except Exception:
            pass  # If EXIF fix fails, continue with unrotated image
    
    return image


def save_image(image: np.ndarray, output_path: str, quality: int = 95) -> str:
    """Save image to file"""
    # Create directories if they don't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.lower().endswith(('.jpg', '.jpeg')):
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(output_path, image)
    
    return output_path


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get image height and width"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = load_image(image_path)
    height, width = image.shape[:2]
    
    return height, width


def get_file_size(file_path: str) -> float:
    """Get file size in MB"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb


def compute_image_hash(image_path: str) -> str:
    """Compute SHA256 hash of image file"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    
    with open(image_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    return file_hash


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    maintain_aspect_ratio: bool = True
) -> np.ndarray:
    """Resize image"""
    if maintain_aspect_ratio:
        height, width = image.shape[:2]
        target_height, target_width = size
        
        aspect_ratio = width / height
        target_aspect = target_width / target_height
        
        if aspect_ratio > target_aspect:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        resized = cv2.resize(image, (new_width, new_height))
    else:
        resized = cv2.resize(image, size)
    
    return resized


def convert_color_space(image: np.ndarray, conversion: str) -> np.ndarray:
    """Convert image color space"""
    conversions = {
        'BGR2RGB': cv2.COLOR_BGR2RGB,
        'BGR2GRAY': cv2.COLOR_BGR2GRAY,
        'RGB2BGR': cv2.COLOR_RGB2BGR,
    }
    
    if conversion not in conversions:
        raise ValueError(f"Unsupported conversion: {conversion}")
    
    return cv2.cvtColor(image, conversions[conversion])