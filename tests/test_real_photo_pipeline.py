"""Tests for Real-World Photo Pipeline

Verifies the new preprocessing, relaxed validation, and resolution-agnostic
measurement inference that make the system work with any phone photo.
"""

import os
import tempfile
import pytest
import numpy as np
import cv2

from ml_ai.core.image_preprocessor import (
    preprocess_for_tryon,
    fix_exif_rotation,
    smart_resize,
    enhance_contrast,
    reduce_noise,
    PreprocessingInfo,
)
from src.validation import (
    validate_image,
    validate_format,
    validate_resolution,
    validate_lighting,
)
from src.measurement_inference import infer_measurements
from src.model_layer import UNetSegmentationModel, MediaPipePoseModel


# ────────────────────────────────────────────────────────────────────
# Image Preprocessor Tests
# ────────────────────────────────────────────────────────────────────

class TestPreprocessor:
    """Test image preprocessing pipeline."""

    def test_preprocess_returns_info(self):
        """preprocess_for_tryon returns image + PreprocessingInfo"""
        image = np.ones((600, 400, 3), dtype=np.uint8) * 128
        result, info = preprocess_for_tryon(image)

        assert isinstance(result, np.ndarray)
        assert isinstance(info, PreprocessingInfo)
        assert info.original_size == (600, 400)

    def test_smart_resize_downscale(self):
        """Images larger than 2048px are downscaled proportionally"""
        big = np.zeros((4096, 3072, 3), dtype=np.uint8)
        resized, was_resized = smart_resize(big)

        assert was_resized
        assert max(resized.shape[:2]) <= 2048

    def test_smart_resize_upscale(self):
        """Images smaller than 400px on shortest side are upscaled"""
        tiny = np.zeros((200, 300, 3), dtype=np.uint8)
        resized, was_resized = smart_resize(tiny)

        assert was_resized
        assert min(resized.shape[:2]) >= 400

    def test_smart_resize_noop(self):
        """Images within bounds are not resized"""
        normal = np.zeros((800, 600, 3), dtype=np.uint8)
        resized, was_resized = smart_resize(normal)

        assert not was_resized
        assert resized.shape == normal.shape

    def test_contrast_enhancement(self):
        """CLAHE enhancement produces valid output"""
        dark = np.ones((100, 100, 3), dtype=np.uint8) * 30
        enhanced = enhance_contrast(dark)

        assert enhanced.shape == dark.shape
        assert enhanced.dtype == np.uint8

    def test_noise_reduction(self):
        """Bilateral filter produces valid output"""
        noisy = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        denoised = reduce_noise(noisy)

        assert denoised.shape == noisy.shape

    def test_exif_rotation_no_path(self):
        """fix_exif_rotation returns original when no path given"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result, rotated = fix_exif_rotation(image, None)

        assert not rotated
        assert result.shape == image.shape

    def test_dark_image_gets_contrast(self):
        """Very dark image triggers contrast enhancement"""
        dark = np.ones((600, 400, 3), dtype=np.uint8) * 20
        result, info = preprocess_for_tryon(dark)

        assert info.contrast_enhanced


# ────────────────────────────────────────────────────────────────────
# Relaxed Validation Tests
# ────────────────────────────────────────────────────────────────────

class TestRelaxedValidation:
    """Test that validation now accepts more image types."""

    def test_256x256_passes_resolution(self):
        """256x256 images now pass resolution validation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image = np.ones((256, 256, 3), dtype=np.uint8) * 128
            path = os.path.join(tmpdir, "small.png")
            cv2.imwrite(path, image)
            assert validate_resolution(path)

    def test_300x400_passes_validation(self):
        """Typical small phone photo passes full validation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image = np.ones((400, 300, 3), dtype=np.uint8) * 128
            path = os.path.join(tmpdir, "phone.jpg")
            cv2.imwrite(path, image)
            result = validate_image(path)
            assert result.is_valid

    def test_heic_format_supported(self):
        """HEIC/HEIF formats are now accepted"""
        assert validate_format("photo.heic")
        assert validate_format("photo.heif")

    def test_wider_brightness_range(self):
        """Darker images (brightness ~20) now pass lighting"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image = np.ones((300, 300, 3), dtype=np.uint8) * 20
            path = os.path.join(tmpdir, "dark.png")
            cv2.imwrite(path, image)
            assert validate_lighting(path)


# ────────────────────────────────────────────────────────────────────
# Resolution-Agnostic Measurements
# ────────────────────────────────────────────────────────────────────

class TestMeasurementsAtMultipleResolutions:
    """Test that measurements work for different image sizes."""

    def test_measurement_inference_512(self):
        """Original 512px size still works"""
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        seg_model = UNetSegmentationModel()
        pose_model = MediaPipePoseModel()

        seg_result = seg_model.predict(image)
        pose_result = pose_model.predict(image)
        measurements = infer_measurements(pose_result, seg_result, image_height=512)

        assert measurements.shoulder_width_cm > 0
        assert measurements.chest_circumference_cm > 0
        assert measurements.torso_length_cm > 0

    def test_measurement_inference_1080(self):
        """Full HD phone resolution works"""
        image = np.ones((1080, 720, 3), dtype=np.uint8) * 128
        seg_model = UNetSegmentationModel()
        pose_model = MediaPipePoseModel()

        seg_result = seg_model.predict(image)
        pose_result = pose_model.predict(image)
        measurements = infer_measurements(pose_result, seg_result, image_height=1080)

        assert measurements.shoulder_width_cm > 0
        assert measurements.chest_circumference_cm > 0

    def test_measurement_backward_compat(self):
        """Measurements still work without image_height (backward compat)"""
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        seg_model = UNetSegmentationModel()
        pose_model = MediaPipePoseModel()

        seg_result = seg_model.predict(image)
        pose_result = pose_model.predict(image)

        # No image_height — uses fallback
        measurements = infer_measurements(pose_result, seg_result)

        assert measurements.shoulder_width_cm > 0


# ────────────────────────────────────────────────────────────────────
# Segmentation Model Tests
# ────────────────────────────────────────────────────────────────────

class TestSegmentationModel:
    """Test that UNetSegmentationModel works with MediaPipe or fallback."""

    def test_model_produces_mask(self):
        """Model always produces a segmentation mask"""
        model = UNetSegmentationModel()
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        result = model.predict(image)

        assert result.mask is not None
        assert result.mask.shape == (512, 512)
        assert result.confidence > 0

    def test_model_body_parts_exist(self):
        """Model always returns torso/arm/neck body-part masks"""
        model = UNetSegmentationModel()
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        result = model.predict(image)

        assert 'torso' in result.body_parts
        assert 'left_arm' in result.body_parts
        assert 'right_arm' in result.body_parts
        assert 'neck' in result.body_parts
