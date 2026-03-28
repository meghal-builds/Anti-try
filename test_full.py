============================= test session starts =============================
collecting ... collected 18 items

tests/test_garment_normalizer.py::TestOutputDimensions::test_output_is_512x512 PASSED [  5%]
tests/test_garment_normalizer.py::TestOutputDimensions::test_custom_canvas_size PASSED [ 11%]
tests/test_garment_normalizer.py::TestMaskBinary::test_mask_is_binary PASSED [ 16%]
tests/test_garment_normalizer.py::TestMaskBinary::test_mask_has_foreground PASSED [ 22%]
tests/test_garment_normalizer.py::TestCentroidAlignment::test_centroid_near_center PASSED [ 27%]
tests/test_garment_normalizer.py::TestTransparentBackground::test_output_has_alpha_channel PASSED [ 33%]
tests/test_garment_normalizer.py::TestTransparentBackground::test_background_is_transparent PASSED [ 38%]
tests/test_garment_normalizer.py::TestMaskQuality::test_mask_quality_valid PASSED [ 44%]
tests/test_garment_normalizer.py::TestMaskQuality::test_mask_quality_reports_area PASSED [ 50%]
tests/test_garment_normalizer.py::TestMaskQuality::test_mask_quality_single_component PASSED [ 55%]
tests/test_garment_normalizer.py::TestMaskQuality::test_no_internal_holes PASSED [ 61%]
tests/test_garment_normalizer.py::TestFileSaving::test_normalize_and_save_creates_files PASSED [ 66%]
tests/test_garment_normalizer.py::TestFileSaving::test_saved_garment_is_512x512 PASSED [ 72%]
tests/test_garment_normalizer.py::TestFileSaving::test_metadata_contains_normalization_info PASSED [ 77%]
tests/test_garment_normalizer.py::TestScaleTracking::test_scale_to_canvas_is_recorded PASSED [ 83%]
tests/test_garment_normalizer.py::TestScaleTracking::test_original_size_is_recorded PASSED [ 88%]
tests/test_garment_normalizer.py::TestEdgeCases::test_empty_image_fails_gracefully PASSED [ 94%]
tests/test_garment_normalizer.py::TestEdgeCases::test_tiny_image PASSED  [100%]

============================= 18 passed in 29.57s =============================

