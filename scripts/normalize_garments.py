#!/usr/bin/env python
"""Batch normalize all garment images.

Iterates over all garments in database/data/garments/,
runs background removal + centering + 512×512 canvas normalization,
and saves garment.png + garment_mask.png + updates metadata.json.

Usage:
    cd c:\\Users\\megha\\Desktop\\Anti-tryon
    python scripts/normalize_garments.py

    # Force re-normalize even if already done:
    python scripts/normalize_garments.py --force

    # Use a different canvas size:
    python scripts/normalize_garments.py --canvas-size 768
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ml_ai.core.garment_normalizer import GarmentNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Normalize all garment images")
    parser.add_argument(
        "--garments-dir",
        default="database/data/garments",
        help="Path to garments directory (default: database/data/garments)",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=512,
        help="Canvas size for normalized output (default: 512)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-normalize even if garment.png already exists",
    )
    args = parser.parse_args()

    garments_dir = Path(args.garments_dir)
    if not garments_dir.exists():
        # Try from project root
        garments_dir = project_root / args.garments_dir
    if not garments_dir.exists():
        logger.error(f"Garments directory not found: {args.garments_dir}")
        sys.exit(1)

    normalizer = GarmentNormalizer(canvas_size=args.canvas_size)

    # Find all garment directories (must contain metadata.json)
    garment_dirs = sorted(
        d for d in garments_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    )

    if not garment_dirs:
        logger.warning(f"No garments found in {garments_dir}")
        sys.exit(0)

    logger.info(f"Found {len(garment_dirs)} garments in {garments_dir}")
    logger.info(f"Canvas size: {args.canvas_size}×{args.canvas_size}")
    logger.info("")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for garment_dir in garment_dirs:
        garment_id = garment_dir.name

        # Skip if already normalized (unless --force)
        if not args.force and (garment_dir / "garment.png").exists():
            logger.info(f"  ⏭  {garment_id} — already normalized (use --force to redo)")
            skip_count += 1
            continue

        # Load metadata to find the raw image filename
        try:
            with open(garment_dir / "metadata.json", "r") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"  ✗  {garment_id} — failed to read metadata: {e}")
            fail_count += 1
            continue

        image_filename = metadata.get("image_filename", "image.png")
        raw_image_path = garment_dir / image_filename

        if not raw_image_path.exists():
            logger.error(f"  ✗  {garment_id} — raw image not found: {image_filename}")
            fail_count += 1
            continue

        # Run normalization
        logger.info(f"  ⚙  {garment_id} — normalizing...")
        result = normalizer.normalize_and_save(str(raw_image_path), str(garment_dir))

        if result.success:
            logger.info(
                f"  ✓  {garment_id} — done "
                f"(scale={result.scale_to_canvas:.3f}, "
                f"area={result.mask_quality.area_ratio:.1%}, "
                f"valid={result.mask_quality.is_valid})"
            )
            if result.warnings:
                for w in result.warnings:
                    logger.warning(f"     ⚠  {w}")
            success_count += 1
        else:
            logger.error(f"  ✗  {garment_id} — FAILED: {result.error}")
            fail_count += 1

    # Summary
    logger.info("")
    logger.info("=" * 50)
    logger.info(f"RESULTS: {success_count} normalized, {skip_count} skipped, {fail_count} failed")
    logger.info("=" * 50)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
