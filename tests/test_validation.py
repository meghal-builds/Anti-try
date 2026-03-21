"""Tests for validation module"""

import pytest
import tempfile
import numpy as np
import cv2
import os

from src.validation import (
    validate_image,
    validate_format,
    validate_file_size,
    validate_resolution,
    validate_lighting,
)


class TestValidateImage:
    """Test image validation"""
    
    def test_nonexistent_file(self):
        """Test validation of nonexistent file"""
        result = validate_image("nonexistent.png")
        
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_unsupported_format(self):
        """Test validation of unsupported format"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a text file
            path = os.path.join(tmpdir, "test.txt")
            with open(path, 'w') as f:
                f.write("test")
            
            result = validate_image(path)
            
            assert not result.is_valid


class TestValidateFormat:
    """Test format validation"""
    
    def test_supported_formats(self):
        """Test supported formats"""
        assert validate_format("image.jpg")
        assert validate_format("image.png")
        assert validate_format("image.webp")
    
    def test_unsupported_formats(self):
        """Test unsupported formats"""
        assert not validate_format("image.txt")
        assert not validate_format("image.bmp")


class TestValidateResolution:
    """Test resolution validation"""
    
    def test_resolution_meets_minimum(self):
        """Test image with minimum resolution"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 512x512 image
            image = np.ones((512, 512, 3), dtype=np.uint8)
            path = os.path.join(tmpdir, "test.png")
            cv2.imwrite(path, image)
            
            assert validate_resolution(path)
    
    def test_resolution_below_minimum(self):
        """Test image below minimum resolution"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 100x100 image (below new 256x256 minimum)
            image = np.ones((100, 100, 3), dtype=np.uint8)
            path = os.path.join(tmpdir, "test.png")
            cv2.imwrite(path, image)
            
            assert not validate_resolution(path)