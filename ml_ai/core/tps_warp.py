"""
Thin Plate Spline (TPS) Warping Module
AI-Based Virtual Try-On and Fit Recommendation System

TPS warping deforms a garment image so its anchor points align
with corresponding body keypoints detected by MediaPipe.

Usage:
    warped = tps_warp(garment_image, src_points, dst_points)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Core TPS math
# ---------------------------------------------------------------------------

def _radial_basis(r: np.ndarray) -> np.ndarray:
    """
    TPS radial basis function: U(r) = r^2 * log(r^2)
    Handles r=0 safely.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(r == 0.0, 0.0, r ** 2 * np.log(r ** 2 + 1e-12))
    return result


def _build_tps_system(src: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the TPS linear system matrix K and the full system matrix L.

    Args:
        src: (N, 2) source control points

    Returns:
        K: (N, N) pairwise RBF matrix
        L: (N+3, N+3) full TPS system matrix
    """
    n = src.shape[0]

    # Pairwise distances
    diff = src[:, None, :] - src[None, :, :]          # (N, N, 2)
    r = np.sqrt((diff ** 2).sum(axis=2))               # (N, N)
    K = _radial_basis(r)                               # (N, N)

    # Affine part P: [1, x, y]
    P = np.hstack([np.ones((n, 1)), src])              # (N, 3)

    # Assemble L
    top    = np.hstack([K, P])                         # (N, N+3)
    bottom = np.hstack([P.T, np.zeros((3, 3))])        # (3, N+3)
    L = np.vstack([top, bottom])                       # (N+3, N+3)

    return K, L


def _solve_tps_weights(
    src: np.ndarray,
    dst: np.ndarray,
    regularization: float = 0.0
) -> np.ndarray:
    """
    Solve for TPS weights W given source→destination point pairs.

    Args:
        src: (N, 2) source control points
        dst: (N, 2) destination control points
        regularization: smoothness regularization (0 = interpolating)

    Returns:
        W: (N+3, 2) weight matrix [w1..wN, a1, ax, ay] for x and y
    """
    n = src.shape[0]
    K, L = _build_tps_system(src)

    if regularization > 0:
        L[:n, :n] += regularization * np.eye(n)

    # RHS: target coordinates padded with zeros for affine constraints
    rhs = np.vstack([dst, np.zeros((3, 2))])           # (N+3, 2)

    W = np.linalg.solve(L, rhs)                        # (N+3, 2)
    return W


def _apply_tps(
    query_points: np.ndarray,
    src: np.ndarray,
    W: np.ndarray
) -> np.ndarray:
    """
    Map query points using solved TPS weights.

    Args:
        query_points: (M, 2) points to transform
        src:          (N, 2) original control points
        W:            (N+3, 2) TPS weights

    Returns:
        mapped: (M, 2) transformed points
    """
    n = src.shape[0]
    m = query_points.shape[0]

    # RBF part
    diff = query_points[:, None, :] - src[None, :, :]  # (M, N, 2)
    r    = np.sqrt((diff ** 2).sum(axis=2))             # (M, N)
    Kq   = _radial_basis(r)                             # (M, N)

    # Affine part
    P = np.hstack([np.ones((m, 1)), query_points])      # (M, 3)

    # Full basis
    basis = np.hstack([Kq, P])                          # (M, N+3)

    mapped = basis @ W                                  # (M, 2)
    return mapped


# ---------------------------------------------------------------------------
# Public warping API
# ---------------------------------------------------------------------------

def tps_warp(
    garment_image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    output_size: Tuple[int, int] | None = None,
    regularization: float = 0.001
) -> np.ndarray:
    """
    Warp garment_image so that src_points map to dst_points using TPS.

    Args:
        garment_image:  Source garment image (H, W, 3) or (H, W, 4)
        src_points:     (N, 2) float array of anchor points in garment image
                        coordinates [x, y]
        dst_points:     (N, 2) float array of target positions in output
                        image coordinates [x, y]
        output_size:    (out_h, out_w) — defaults to garment_image size
        regularization: TPS smoothness (0 = exact interpolation,
                        small positive = smoother)

    Returns:
        Warped image as numpy array, same dtype and channels as input.

    Raises:
        ValueError: if point arrays are mismatched or degenerate
    """
    src_points = np.asarray(src_points, dtype=np.float64)
    dst_points = np.asarray(dst_points, dtype=np.float64)

    if src_points.shape != dst_points.shape:
        raise ValueError(
            f"src_points shape {src_points.shape} != "
            f"dst_points shape {dst_points.shape}"
        )
    if src_points.ndim != 2 or src_points.shape[1] != 2:
        raise ValueError("Points must be (N, 2) arrays")
    if src_points.shape[0] < 3:
        raise ValueError("Need at least 3 control points for TPS")

    h, w = garment_image.shape[:2]
    out_h, out_w = output_size if output_size else (h, w)

    # Solve TPS from DST→SRC (we need inverse map for remap)
    W = _solve_tps_weights(dst_points, src_points, regularization)

    # Build dense inverse map over output grid
    grid_x, grid_y = np.meshgrid(
        np.arange(out_w, dtype=np.float64),
        np.arange(out_h, dtype=np.float64)
    )
    grid_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (out_h*out_w, 2)

    src_mapped = _apply_tps(grid_pts, dst_points, W)               # (out_h*out_w, 2)

    map_x = src_mapped[:, 0].reshape(out_h, out_w).astype(np.float32)
    map_y = src_mapped[:, 1].reshape(out_h, out_w).astype(np.float32)

    # Remap garment image
    warped = cv2.remap(
        garment_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0) if garment_image.shape[2] == 4 else (0, 0, 0)
    )

    return warped


def tps_warp_with_mask(
    garment_image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    output_size: Tuple[int, int] | None = None,
    regularization: float = 0.001
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp garment and return (warped_image, warped_mask).

    The mask is 1 where the warped garment has valid pixels, 0 elsewhere.
    Useful for compositing — lets the caller blend only valid garment pixels.

    Args:
        garment_image:  (H, W, 3) or (H, W, 4) garment image
        src_points:     (N, 2) anchor points in garment coordinates
        dst_points:     (N, 2) corresponding body keypoint positions
        output_size:    (out_h, out_w) target canvas size
        regularization: TPS smoothness

    Returns:
        warped:  Warped garment image
        mask:    (out_h, out_w) uint8 binary mask
    """
    warped = tps_warp(
        garment_image, src_points, dst_points,
        output_size=output_size,
        regularization=regularization
    )

    # Build mask from alpha channel or non-black pixels
    if warped.ndim == 3 and warped.shape[2] == 4:
        mask = (warped[:, :, 3] > 10).astype(np.uint8)
    else:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask = (gray > 10).astype(np.uint8)

    # Clean mask with morphological ops
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return warped, mask


def compute_warp_quality(
    src_points: np.ndarray,
    dst_points: np.ndarray
) -> dict:
    """
    Estimate warp quality before applying it.

    Checks for degenerate configurations (all points collinear,
    extreme scale changes, etc.)

    Args:
        src_points: (N, 2) source control points
        dst_points: (N, 2) destination control points

    Returns:
        dict with keys:
            is_valid (bool),
            scale_factor (float),
            warnings (list[str])
    """
    warnings = []

    src_span = np.ptp(src_points, axis=0)  # [x_range, y_range]
    dst_span = np.ptp(dst_points, axis=0)

    # Check for degenerate source (all points nearly collinear)
    if np.any(src_span < 5):
        warnings.append("Source points nearly collinear — warp may be unstable")

    # Scale factor estimate
    src_scale = float(np.mean(src_span))
    dst_scale = float(np.mean(dst_span))
    scale_factor = dst_scale / src_scale if src_scale > 0 else 1.0

    if scale_factor > 3.0:
        warnings.append(f"Large upscale ({scale_factor:.1f}x) may reduce quality")
    if scale_factor < 0.2:
        warnings.append(f"Large downscale ({scale_factor:.1f}x) — garment will be small")

    is_valid = len([w for w in warnings if "unstable" in w]) == 0

    return {
        "is_valid": is_valid,
        "scale_factor": round(scale_factor, 3),
        "warnings": warnings
    }