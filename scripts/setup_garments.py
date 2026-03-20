"""
Garment Database Setup Script
AI-Based Virtual Try-On and Fit Recommendation System

Processes raw garment images from raw_garments/ folder:
    1. Removes background using rembg
    2. Resizes to standard 768x1024
    3. Creates database/data/garments/ folder structure
    4. Generates metadata.json with anchor points + size charts

Usage:
    python scripts/setup_garments.py

Requirements:
    pip install rembg onnxruntime
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from rembg import remove
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_GARMENTS_DIR  = Path("raw_garments")
OUTPUT_BASE_DIR   = Path("database/data/garments")
STANDARD_SIZE     = (768, 1024)   # (width, height)

# Supported input formats
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp"}

# ---------------------------------------------------------------------------
# Garment definitions
# Map filename stem → garment metadata + anchor points
# ---------------------------------------------------------------------------

GARMENT_DEFINITIONS = {
    "tshirt": {
        "id":          "tshirt-001",
        "name":        "Classic Men's T-Shirt",
        "category":    "tshirt",
        "brand":       "BasicWear",
        "description": "Classic fit men's crew neck t-shirt",
        "material":    "100% Cotton",
        "price_usd":   19.99,
        "available_colors": ["white", "black", "navy", "grey"],
        "image_filename": "image.png",
        "size_chart": {
            "S":  {"shoulder_width_cm": 42.0, "chest_circumference_cm": 91.0,  "torso_length_cm": 68.0},
            "M":  {"shoulder_width_cm": 44.5, "chest_circumference_cm": 96.0,  "torso_length_cm": 70.0},
            "L":  {"shoulder_width_cm": 47.0, "chest_circumference_cm": 101.0, "torso_length_cm": 72.0},
            "XL": {"shoulder_width_cm": 49.5, "chest_circumference_cm": 106.0, "torso_length_cm": 74.0},
            "XXL":{"shoulder_width_cm": 52.0, "chest_circumference_cm": 111.0, "torso_length_cm": 76.0},
        },
        # Anchor points: normalized UV coords on the 768x1024 garment image
        # These match the TSHIRT_SCHEMA in garment_keypoints.py
        "anchor_points": {
            "collar_center":   [0.50, 0.08],
            "left_shoulder":   [0.20, 0.12],
            "right_shoulder":  [0.80, 0.12],
            "left_sleeve_end": [0.10, 0.32],
            "right_sleeve_end":[0.90, 0.32],
            "left_side_waist": [0.18, 0.65],
            "right_side_waist":[0.82, 0.65],
            "left_hem":        [0.18, 0.95],
            "right_hem":       [0.82, 0.95],
        }
    },

    "shirt": {
        "id":          "shirt-001",
        "name":        "Men's Formal Shirt",
        "category":    "shirt",
        "brand":       "FormalEdge",
        "description": "Men's slim fit formal button-down shirt",
        "material":    "60% Cotton, 40% Polyester",
        "price_usd":   39.99,
        "available_colors": ["white", "light blue", "grey", "black"],
        "image_filename": "image.png",
        "size_chart": {
            "S":  {"shoulder_width_cm": 42.0, "chest_circumference_cm": 90.0,  "torso_length_cm": 72.0},
            "M":  {"shoulder_width_cm": 44.5, "chest_circumference_cm": 95.0,  "torso_length_cm": 74.0},
            "L":  {"shoulder_width_cm": 47.0, "chest_circumference_cm": 100.0, "torso_length_cm": 76.0},
            "XL": {"shoulder_width_cm": 49.5, "chest_circumference_cm": 105.0, "torso_length_cm": 78.0},
            "XXL":{"shoulder_width_cm": 52.0, "chest_circumference_cm": 110.0, "torso_length_cm": 80.0},
        },
        "anchor_points": {
            "collar_center":   [0.50, 0.06],
            "left_shoulder":   [0.18, 0.11],
            "right_shoulder":  [0.82, 0.11],
            "left_cuff":       [0.04, 0.72],
            "right_cuff":      [0.96, 0.72],
            "left_elbow":      [0.08, 0.44],
            "right_elbow":     [0.92, 0.44],
            "left_side_waist": [0.16, 0.65],
            "right_side_waist":[0.84, 0.65],
            "left_hem":        [0.16, 0.96],
            "right_hem":       [0.84, 0.96],
        }
    },

    "jacket": {
        "id":          "jacket-001",
        "name":        "Men's Casual Jacket",
        "category":    "jacket",
        "brand":       "UrbanLayer",
        "description": "Men's slim fit casual jacket",
        "material":    "100% Polyester",
        "price_usd":   79.99,
        "available_colors": ["black", "navy", "olive", "grey"],
        "image_filename": "image.png",
        "size_chart": {
            "S":  {"shoulder_width_cm": 43.0, "chest_circumference_cm": 92.0,  "torso_length_cm": 68.0},
            "M":  {"shoulder_width_cm": 45.5, "chest_circumference_cm": 97.0,  "torso_length_cm": 70.0},
            "L":  {"shoulder_width_cm": 48.0, "chest_circumference_cm": 102.0, "torso_length_cm": 72.0},
            "XL": {"shoulder_width_cm": 50.5, "chest_circumference_cm": 107.0, "torso_length_cm": 74.0},
            "XXL":{"shoulder_width_cm": 53.0, "chest_circumference_cm": 112.0, "torso_length_cm": 76.0},
        },
        "anchor_points": {
            "collar_left":     [0.42, 0.08],
            "collar_right":    [0.58, 0.08],
            "left_shoulder":   [0.15, 0.12],
            "right_shoulder":  [0.85, 0.12],
            "left_lapel":      [0.38, 0.22],
            "right_lapel":     [0.62, 0.22],
            "left_elbow":      [0.06, 0.46],
            "right_elbow":     [0.94, 0.46],
            "left_cuff":       [0.03, 0.74],
            "right_cuff":      [0.97, 0.74],
            "left_side_waist": [0.14, 0.62],
            "right_side_waist":[0.86, 0.62],
            "left_hem":        [0.14, 0.95],
            "right_hem":       [0.86, 0.95],
        }
    }
}


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def find_raw_images() -> dict[str, Path]:
    """
    Scan raw_garments/ and match files to garment definitions.
    Returns {garment_key: image_path}
    """
    found = {}
    if not RAW_GARMENTS_DIR.exists():
        print(f"❌ raw_garments/ folder not found at {RAW_GARMENTS_DIR.resolve()}")
        return found

    for path in RAW_GARMENTS_DIR.iterdir():
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            continue
        stem = path.stem.lower()

        # Match filename stem to garment definition
        # e.g. tshirt.png → tshirt, shirt.jpg → shirt
        matched_key = None
        for key in GARMENT_DEFINITIONS:
            if key in stem:
                matched_key = key
                break

        if matched_key:
            found[matched_key] = path
            print(f"  ✅ Found: {path.name} → {matched_key}")
        else:
            print(f"  ⚠️  Skipped: {path.name} (no matching garment key)")

    return found


def remove_background(image_path: Path) -> Image.Image:
    """Remove background from garment image using rembg."""
    print(f"  🔄 Removing background from {image_path.name}...")
    with open(image_path, "rb") as f:
        input_bytes = f.read()

    output_bytes = remove(input_bytes)
    from io import BytesIO
    result = Image.open(BytesIO(output_bytes)).convert("RGBA")
    print(f"  ✅ Background removed")
    return result


def resize_garment(pil_image: Image.Image, size: tuple) -> Image.Image:
    """
    Resize garment image maintaining aspect ratio with padding.
    Pads with transparent pixels to reach exact target size.
    """
    target_w, target_h = size

    # Resize maintaining aspect ratio
    pil_image.thumbnail((target_w, target_h), Image.LANCZOS)

    # Create transparent canvas
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))

    # Center the garment on canvas
    offset_x = (target_w - pil_image.width)  // 2
    offset_y = (target_h - pil_image.height) // 2
    canvas.paste(pil_image, (offset_x, offset_y), pil_image)

    print(f"  ✅ Resized to {target_w}x{target_h} (centered with transparency)")
    return canvas


def save_garment_to_database(
    garment_key: str,
    processed_image: Image.Image,
    definition: dict
) -> Path:
    """
    Save processed garment image and metadata.json to database.
    Returns the garment output directory.
    """
    garment_id  = definition["id"]
    output_dir  = OUTPUT_BASE_DIR / garment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save image as PNG (preserves transparency)
    image_path = output_dir / "image.png"
    processed_image.save(str(image_path), "PNG")
    print(f"  ✅ Image saved → {image_path}")

    # Build metadata (exclude anchor_points from top level — embed inside)
    metadata = {k: v for k, v in definition.items()}
    metadata["id"] = garment_id

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ Metadata saved → {metadata_path}")

    return output_dir


def verify_garment(garment_id: str) -> bool:
    """Verify saved garment has all required files and fields."""
    garment_dir   = OUTPUT_BASE_DIR / garment_id
    image_path    = garment_dir / "image.png"
    metadata_path = garment_dir / "metadata.json"

    if not image_path.exists():
        print(f"  ❌ Missing image: {image_path}")
        return False

    if not metadata_path.exists():
        print(f"  ❌ Missing metadata: {metadata_path}")
        return False

    with open(metadata_path) as f:
        meta = json.load(f)

    required_fields = ["id", "name", "category", "brand", "image_filename", "size_chart"]
    missing = [f for f in required_fields if f not in meta]
    if missing:
        print(f"  ❌ Missing metadata fields: {missing}")
        return False

    # Check image is readable and has transparency
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  ❌ Cannot read image")
        return False

    has_alpha = img.shape[2] == 4 if len(img.shape) == 3 else False
    if not has_alpha:
        print(f"  ⚠️  Image has no alpha channel — background removal may have failed")

    print(f"  ✅ Verification passed ({img.shape[1]}x{img.shape[0]}, alpha={has_alpha})")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Garment Database Setup")
    print("  AI-Based Virtual Try-On and Fit Recommendation System")
    print("=" * 60)

    # ── Scan raw_garments/ ────────────────────────────────────────────
    print("\n📂 Scanning raw_garments/ folder...")
    raw_images = find_raw_images()

    if not raw_images:
        print("\n❌ No matching images found in raw_garments/")
        print("   Make sure files are named: tshirt.png, shirt.png, jacket.png")
        return

    print(f"\n  Found {len(raw_images)} garment(s) to process\n")

    # ── Process each garment ──────────────────────────────────────────
    success_count = 0
    for garment_key, image_path in raw_images.items():
        definition = GARMENT_DEFINITIONS[garment_key]
        garment_id = definition["id"]

        print(f"\n{'─' * 50}")
        print(f"  Processing: {garment_key} → {garment_id}")
        print(f"{'─' * 50}")

        try:
            # Step 1: Remove background
            processed = remove_background(image_path)

            # Step 2: Resize to standard size
            processed = resize_garment(processed, STANDARD_SIZE)

            # Step 3: Save to database
            save_garment_to_database(garment_key, processed, definition)

            # Step 4: Verify
            print(f"\n  🔍 Verifying...")
            if verify_garment(garment_id):
                success_count += 1
                print(f"  🎉 {garment_id} ready!")
            else:
                print(f"  ❌ Verification failed for {garment_id}")

        except Exception as e:
            print(f"  ❌ Failed to process {garment_key}: {e}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  ✅ Done! {success_count}/{len(raw_images)} garments processed")
    print(f"  📁 Output: {OUTPUT_BASE_DIR.resolve()}")
    print(f"{'=' * 60}\n")

    if success_count > 0:
        print("Next steps:")
        print("  1. Run your Streamlit app")
        print("  2. Upload a person photo")
        print("  3. Go to Virtual Try-On section")
        print("  4. Select a garment and press 'Try It On'\n")


if __name__ == "__main__":
    main()