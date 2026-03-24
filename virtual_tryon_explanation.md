# Virtual Try-On and Size Recommendation System Overview

This document provides a deep, technical explanation of the Anti-Tryon project's two core systems: **1) Size Recommendation** and **2) Virtual Try-On Engine**, specifically designed to be ingested by another AI agent for problem-solving.

---

## 1. Size Recommendation System

The Size Recommendation system takes a user's image, securely infers their body measurements (using height calibration), and matches them against a garment's size chart to recommend the best-fitting size.

### 1.1 Body Measurement Inference ([measurement_inference.py](file:///c:/Users/megha/Desktop/Anti-tryon/ml_ai/core/measurement_inference.py))
This step is critical for deriving realistic physical measurements from a 2D image.

- **Height-Based Calibration (Real Fix):** 
  - Instead of estimating measurements using assumed body proportions, the system uses the user's real height (`user_height_cm`) to calculate a [pixels_per_cm](file:///c:/Users/megha/Desktop/Anti-tryon/ml_ai/core/measurement_inference.py#322-325) ratio.
  - It uses a MediaPipe pose detection to find the top of the body (nose, acting as head proxy, adjusted +6%) and the bottom (ankles, adjusted for ground clearance), providing a precise anchor for actual scale.
- **Anatomical Adjustments:**
  - **Chest Circumference:** Extracted from pose keypoints (`shoulder_width_px * 1.15` for ribcage), converted to cm, and multiplied by `2.35` to transition from flat front-width to full circumference.
  - **Torso Length:** Measured from shoulder to hip span, adjusted by `1.13x` to account for spinal curvature that is typically missing in front views.
- **Fallback Mechanisms:** If the user doesn't provide height, or ankles are not visible:
  - Backs off to a shoulder-to-hip torso scale (assuming torso is ~30% of total height).
  - Falls further back to an old "estimated" method with very low confidence scores (capped at 0.40).
- **Confidence Scoring:** Outputs a confidence rating (0.0 to 0.95+) depending on whether calibration was "height" based vs. "estimated", overall keypoint visibility, and frontal pose validity.

### 1.2 Size Recommendation & Matching ([size_recommendation.py](file:///c:/Users/megha/Desktop/Anti-tryon/ml_ai/core/size_recommendation.py))
Matches the inferred body measurements against a specific garment's `size_chart` metadata.

- **Fit Score Calculation ([calculate_fit_score](file:///c:/Users/megha/Desktop/Anti-tryon/ml_ai/core/size_recommendation.py#39-81)):**
  - Iterates over available sizes in the size chart (e.g., S, M, L, XL).
  - Uses a **Weighted Scoring System**:
    - **Shoulder Width:** 50% weight (Tolerance: 5%)
    - **Chest Circumference:** 35% weight (Tolerance: 7%)
    - **Torso Length:** 15% weight (Tolerance: 5%)
- **Match Mechanism ([calculate_measurement_fit](file:///c:/Users/megha/Desktop/Anti-tryon/ml_ai/core/measurement_inference.py#289-302)):**
  - For each measurement (e.g. shoulder), calculates the difference between `user_value` and `size_value` as a percentage.
  - Linearly degrades the score from 1.0 (perfect match) down to 0.0 at the maximum allowed tolerance (tolerance `* 2`).
- **Output:** Returns a [SizeRecommendation](file:///c:/Users/megha/Desktop/Anti-tryon/ml_ai/core/models.py#68-75) object with the `best_size` (highest total fit score), an aggregated confidence score, a list of alternative `recommended_sizes` (scoring > 0.7), and granular fit scores.

---

## 2. Virtual Try-On Engine ([tryon_engine.py](file:///c:/Users/megha/Desktop/Anti-tryon/ml_ai/core/tryon_engine.py))

The Virtual Try-On Engine handles visually compositing a 2D garment onto a person's photo using a physics-like Thin Plate Spline (TPS) warp, guided by semantic segmentation and pose detection.

### 2.1 Preprocessing and Feature Extraction
When `/api/tryon` is hit, the engine triggers the following vision models:

- **Preprocessing (`preprocess_for_tryon`):** Checks and prepares real-world images (auto-orient, contrast, clarity).
- **Pose Detection ([detect_pose](file:///c:/Users/megha/Desktop/Anti-tryon/ml_ai/core/tryon_engine.py#242-248)):** Leverages a real MediaPipe Pose instance to detect 33 skeletal keypoints (shoulders, elbows, hips, nose, ankles, etc.).
- **Segmentation (`segment_body`):** Identifies distinct body parts ([torso](file:///c:/Users/megha/Desktop/Anti-tryon/ml_ai/core/measurement_inference.py#223-236), `left_arm`, `right_arm`, `neck`, etc.) outputting binary bitmasks for layered compositing later.

### 2.2 Garment Warping (The TPS Pipeline)
The engine utilizes a Thin Plate Spline (TPS) transformation to organically deform the garment image to fit the wearer's shape.

- **Control Points Resolution (`resolve_points`):**
  - Fetches a "schema" for the given `garment_category` (e.g., tshirt, jacket) which loosely defines where the garment should sit anatomically (shoulders, armpits, hem).
  - Maps source points (garment image borders/keypoints) to destination points (the person's body keypoints, dynamically scaled by a `shoulder_scale` parameter acting as a fit width multiplier).
- **Warp Quality Check (`compute_warp_quality`):**
  - Runs a topological non-degeneracy check. Validates that the mapped points don't cause impossible distortions (e.g., left and right points crossing over).
- **TPS Warp (`tps_warp_with_mask`):**
  - Calculates the TPS transformation matrix and warps both the garment's RGB values and an generated Alpha/mask layer to the exact dimensions of the person's image output frame [(person_h, person_w)](file:///c:/Users/megha/Desktop/Anti-tryon/ml_ai/core/tryon_engine.py#90-228).

### 2.3 Semantic Compositing (`composite_garment_on_person`)
After the garment is warped into the correct spatial coordinates, it is smoothly blended over the user's base photo.

- **Upper Body Masking ([_build_upper_body_mask](file:///c:/Users/megha/Desktop/Anti-tryon/ml_ai/core/tryon_engine.py#276-293)):**
  - Combines segmentation outputs ([torso](file:///c:/Users/megha/Desktop/Anti-tryon/ml_ai/core/measurement_inference.py#223-236) + `neck` + `left_arm` + `right_arm`) into a monolithic "upper body" mask.
  - Slightly dilates the mask using `cv2.dilate` (morphological ellipse) so that the edges of the garment are allowed to roll naturally over the edges of the person's silhouette without hard clipping at the strict body boundary.
- **Alpha Blending:**
  - Uses the `garment_mask` and the `blend_alpha` parameter (default 0.92) to seamlessly mix the warped garment layer onto the body. High-opacity keeps the garment solid, but blending prevents a "pasted on" look by slightly preserving environmental light/shadow logic.

### 2.4 Try-On Pipeline Lifecycle
- Validation → Preprocessing → Pose Detection → Segmentation → Resolve TPS Points → Quality Check → TPS Warp → Segmented Composite → Return Base64 or Image Response. 

---
*For any AI using this data for problem-solving: Note the distinction between [tryon_engine.py](file:///c:/Users/megha/Desktop/Anti-tryon/ml_ai/core/tryon_engine.py) (which exclusively handles visual presentation & TPS transforms) and [measurement_inference.py](file:///c:/Users/megha/Desktop/Anti-tryon/ml_ai/core/measurement_inference.py)/[size_recommendation.py](file:///c:/Users/megha/Desktop/Anti-tryon/ml_ai/core/size_recommendation.py) (which handle the abstract math & logic of proper sizing).*
