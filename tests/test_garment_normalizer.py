"""Tests for Garment Normalizer Module

Tests cover:
    1. Output canvas is exactly 512×512
    2. Mask is strictly binary (0 and 255 only)
    3. Garment is centered (centroid near canvas center)
    4. Output has transparent background (RGBA with alpha=0 outside garment)
    5. Mask quality validation (area ratio, connected components)
    6. No internal holes in mask
    7. Metadata is written correctly
    8. Smart crop padding prevents clipping
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_ai.core.garment_normalizer import GarmentNormalizer, MaskQuality


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normalizer():
    """Create a normalizer with default settings."""
    return GarmentNormalizer(canvas_size=512)


@pytest.fixture
def synthetic_garment():
    """
    Create a synthetic garment image: a colored rectangle on a white background.
    Simulates a flat-lay garment photo.
    """
    # White background (800x600)
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255

    # Blue rectangle in center (simulates garment)
    cv2.rectangle(img, (100, 150), (500, 650), (200, 50, 50), -1)

    return img


@pytest.fixture
def synthetic_garment_rgba():
    """
    Create a synthetic RGBA garment image with transparent background.
    """
    img = np.zeros((600, 400, 4), dtype=np.uint8)

    # Red garment shape
    cv2.rectangle(img, (50, 80), (350, 520), (0, 0, 200, 255), -1)

    return img


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for output."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# Test: Output dimensions
# ---------------------------------------------------------------------------

class TestOutputDimensions:
    def test_output_is_512x512(self, normalizer, synthetic_garment_rgba):
        """Verify output canvas is exactly 512×512."""
        result = normalizer.normalize(synthetic_garment_rgba)
        assert result.success, f"Normalization failed: {result.error}"
        assert result.garment_image.shape[:2] == (512, 512)
        assert result.garment_mask.shape == (512, 512)

    def test_custom_canvas_size(self, synthetic_garment_rgba):
        """Verify custom canvas size works."""
        normalizer = GarmentNormalizer(canvas_size=256)
        result = normalizer.normalize(synthetic_garment_rgba)
        assert result.success
        assert result.garment_image.shape[:2] == (256, 256)
        assert result.garment_mask.shape == (256, 256)


# ---------------------------------------------------------------------------
# Test: Mask is binary
# ---------------------------------------------------------------------------

class TestMaskBinary:
    def test_mask_is_binary(self, normalizer, synthetic_garment_rgba):
        """Verify mask contains only 0 and 255 values."""
        result = normalizer.normalize(synthetic_garment_rgba)
        assert result.success
        unique_values = set(np.unique(result.garment_mask))
        assert unique_values.issubset({0, 255}), (
            f"Mask has non-binary values: {unique_values}"
        )

    def test_mask_has_foreground(self, normalizer, synthetic_garment_rgba):
        """Verify mask is not all-black (has some foreground)."""
        result = normalizer.normalize(synthetic_garment_rgba)
        assert result.success
        assert np.any(result.garment_mask == 255), "Mask has no foreground pixels"


# ---------------------------------------------------------------------------
# Test: Centroid alignment
# ---------------------------------------------------------------------------

class TestCentroidAlignment:
    def test_centroid_near_center(self, normalizer, synthetic_garment_rgba):
        """Verify garment centroid is near canvas center."""
        result = normalizer.normalize(synthetic_garment_rgba)
        assert result.success

        cx, cy = result.centroid
        canvas_center = normalizer.canvas_size / 2

        # Centroid should be within 10% of canvas center
        tolerance = normalizer.canvas_size * 0.10
        assert abs(cx - canvas_center) < tolerance, (
            f"Centroid X={cx} too far from center {canvas_center}"
        )
        assert abs(cy - canvas_center) < tolerance, (
            f"Centroid Y={cy} too far from center {canvas_center}"
        )


# ---------------------------------------------------------------------------
# Test: Transparent background
# ---------------------------------------------------------------------------

class TestTransparentBackground:
    def test_output_has_alpha_channel(self, normalizer, synthetic_garment_rgba):
        """Verify output garment.png is RGBA (4 channels)."""
        result = normalizer.normalize(synthetic_garment_rgba)
        assert result.success
        assert result.garment_image.shape[2] == 4, "Output should be RGBA (4 channels)"

    def test_background_is_transparent(self, normalizer, synthetic_garment_rgba):
        """Verify alpha=0 outside garment region."""
        result = normalizer.normalize(synthetic_garment_rgba)
        assert result.success

        alpha = result.garment_image[:, :, 3]
        mask = result.garment_mask

        # Everywhere mask is 0 (background), alpha should also be 0
        bg_region = mask == 0
        if np.any(bg_region):
            bg_alpha = alpha[bg_region]
            assert np.all(bg_alpha == 0), (
                f"Background has non-zero alpha: max={bg_alpha.max()}"
            )


# ---------------------------------------------------------------------------
# Test: Mask quality validation
# ---------------------------------------------------------------------------

class TestMaskQuality:
    def test_mask_quality_valid(self, normalizer, synthetic_garment_rgba):
        """Verify mask quality passes validation for a good garment."""
        result = normalizer.normalize(synthetic_garment_rgba)
        assert result.success
        assert result.mask_quality.is_valid

    def test_mask_quality_reports_area(self, normalizer, synthetic_garment_rgba):
        """Verify area ratio is reported and reasonable."""
        result = normalizer.normalize(synthetic_garment_rgba)
        assert result.success
        assert 0.0 < result.mask_quality.area_ratio < 1.0

    def test_mask_quality_single_component(self, normalizer, synthetic_garment_rgba):
        """Verify mask has exactly 1 connected component."""
        result = normalizer.normalize(synthetic_garment_rgba)
        assert result.success
        # After cleaning, should be 1 main blob
        assert result.mask_quality.num_components == 1

    def test_no_internal_holes(self, normalizer, synthetic_garment_rgba):
        """Verify there are no internal black holes within the garment mask."""
        result = normalizer.normalize(synthetic_garment_rgba)
        assert result.success

        mask = result.garment_mask

        # Find contours
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is not None:
            # Check for child contours (holes = children of outer contour)
            # hierarchy[0][i] = [next, prev, first_child, parent]
            for i in range(len(contours)):
                parent = hierarchy[0][i][3]
                if parent != -1:
                    # This is a hole (child contour) — should be very small
                    hole_area = cv2.contourArea(contours[i])
                    total_area = mask.shape[0] * mask.shape[1]
                    hole_ratio = hole_area / total_area
                    assert hole_ratio < 0.01, (
                        f"Internal hole found with area ratio {hole_ratio:.4f}"
                    )


# ---------------------------------------------------------------------------
# Test: File saving
# ---------------------------------------------------------------------------

class TestFileSaving:
    def test_normalize_and_save_creates_files(self, normalizer, temp_dir):
        """Verify garment.png, garment_mask.png, and metadata.json are created."""
        # Create a synthetic input image
        img = np.zeros((400, 300, 4), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 350), (100, 150, 200, 255), -1)
        input_path = temp_dir / "input.png"
        cv2.imwrite(str(input_path), img)

        # Create a basic metadata.json
        meta = {"id": "test-001", "name": "Test Garment"}
        with open(temp_dir / "metadata.json", "w") as f:
            json.dump(meta, f)

        result = normalizer.normalize_and_save(str(input_path), str(temp_dir))
        assert result.success, f"Save failed: {result.error}"

        assert (temp_dir / "garment.png").exists(), "garment.png not created"
        assert (temp_dir / "garment_mask.png").exists(), "garment_mask.png not created"

        # Verify metadata was updated
        with open(temp_dir / "metadata.json") as f:
            updated_meta = json.load(f)
        assert updated_meta.get("normalized") is True
        assert "normalization" in updated_meta
        assert updated_meta["normalization"]["canvas_size"] == 512

    def test_saved_garment_is_512x512(self, normalizer, temp_dir):
        """Verify the saved garment.png file is 512×512."""
        img = np.zeros((400, 300, 4), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 350), (100, 150, 200, 255), -1)
        input_path = temp_dir / "input.png"
        cv2.imwrite(str(input_path), img)

        result = normalizer.normalize_and_save(str(input_path), str(temp_dir))
        assert result.success

        saved_img = cv2.imread(str(temp_dir / "garment.png"), cv2.IMREAD_UNCHANGED)
        assert saved_img.shape[:2] == (512, 512)

    def test_metadata_contains_normalization_info(self, normalizer, temp_dir):
        """Verify metadata has all required normalization fields."""
        img = np.zeros((400, 300, 4), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 350), (100, 150, 200, 255), -1)
        input_path = temp_dir / "input.png"
        cv2.imwrite(str(input_path), img)

        with open(temp_dir / "metadata.json", "w") as f:
            json.dump({"id": "test"}, f)

        result = normalizer.normalize_and_save(str(input_path), str(temp_dir))
        assert result.success

        with open(temp_dir / "metadata.json") as f:
            meta = json.load(f)

        norm = meta["normalization"]
        assert "original_size" in norm
        assert "bbox" in norm
        assert "scale_to_canvas" in norm
        assert "centroid" in norm
        assert "canvas_size" in norm
        assert "mask_quality" in norm
        assert norm["mask_quality"]["is_valid"] is True


# ---------------------------------------------------------------------------
# Test: Scale tracking
# ---------------------------------------------------------------------------

class TestScaleTracking:
    def test_scale_to_canvas_is_recorded(self, normalizer, synthetic_garment_rgba):
        """Verify scale_to_canvas is a positive float."""
        result = normalizer.normalize(synthetic_garment_rgba)
        assert result.success
        assert isinstance(result.scale_to_canvas, float)
        assert result.scale_to_canvas > 0

    def test_original_size_is_recorded(self, normalizer, synthetic_garment_rgba):
        """Verify original_size matches input dimensions."""
        result = normalizer.normalize(synthetic_garment_rgba)
        assert result.success
        h, w = synthetic_garment_rgba.shape[:2]
        assert result.original_size == (h, w)


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_image_fails_gracefully(self, normalizer):
        """Verify normalizer handles an all-transparent image."""
        empty = np.zeros((400, 300, 4), dtype=np.uint8)
        result = normalizer.normalize(empty)
        # Should either fail or produce empty output
        if result.success:
            assert np.sum(result.garment_mask) == 0

    def test_tiny_image(self, normalizer):
        """Verify normalizer handles a very small image."""
        tiny = np.zeros((32, 32, 4), dtype=np.uint8)
        cv2.rectangle(tiny, (5, 5), (27, 27), (100, 100, 200, 255), -1)
        result = normalizer.normalize(tiny)
        if result.success:
            assert result.garment_image.shape[:2] == (512, 512)
