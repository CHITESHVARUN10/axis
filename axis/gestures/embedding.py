"""Path embeddings — fixed-length, scale-invariant signatures of a motion.

A recorded gesture path (variable-length sequence of palm positions) is
converted to a fixed-length *embedding*: the path is resampled to N
equally-spaced points, normalized to the palm size, and the displacement
vectors are kept. This makes two recordings of the "same" gesture — done
at different speeds, angles, or distances — produce similar embeddings,
which is exactly what template matching needs.

The raw recording is discarded after embedding (privacy + size): only the
embedding + a few scalars are persisted.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

EMBED_DIM = 16  # resampled points → 15 displacement vectors, each (dx, dy)

# Noise floor: below this, displacement is treated as stationary.
EPS = 1e-6


def resample_path(points: Sequence[np.ndarray], n: int) -> np.ndarray:
    """Resample a polyline of (x, y) points to exactly n equally-spaced points.

    Uses cumulative arc-length interpolation, so a fast stroke and a slow
    stroke of the same path produce the same resampled shape.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        # Degenerate: duplicate the single point.
        pts = np.vstack([pts, pts]) if len(pts) == 1 else np.zeros((2, 2))
    if len(pts) == n:
        return pts

    # Cumulative arc length.
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if arc[-1] < EPS:
        return np.repeat(pts[:1], n, axis=0)
    # Equally spaced arc positions.
    targets = np.linspace(0.0, arc[-1], n)
    return np.column_stack(
        [np.interp(targets, arc, pts[:, 0]), np.interp(targets, arc, pts[:, 1])]
    )


def path_embedding(
    points: Sequence[np.ndarray],
    palm_sizes: Sequence[float],
    dim: int = EMBED_DIM,
) -> np.ndarray:
    """Build a fixed-length embedding from a path + per-point palm sizes.

    The path is resampled to `dim` points; the average palm size (over
    samples where the hand was visible) normalizes the displacements, so
    the embedding is invariant to hand size and camera distance.

    Returns a flat (2*(dim-1),) float array, or an empty array if the
    path is degenerate.
    """
    if len(points) < 2:
        return np.array([], dtype=np.float64)
    resampled = resample_path(points, dim)
    # Mean palm size over the recording (ignoring zero sizes).
    sizes = [s for s in palm_sizes if s and s > EPS]
    scale = float(np.mean(sizes)) if sizes else 1.0
    if scale <= EPS:
        scale = 1.0
    vecs = np.diff(resampled, axis=0) / scale
    return vecs.reshape(-1)


def embedding_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine-ish similarity distance between two embeddings (0 = identical).

    Returns a small number for similar paths, large for different ones.
    Uses normalized L2 distance on the unit vectors — robust to the
    overall magnitude of the motion (a bigger sweep of the same path is
    the same gesture).
    """
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("inf")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return float("inf")
    # Normalize both to unit length, then L2 distance (0..2).
    return float(np.linalg.norm(a / na - b / nb))


__all__ = ["EMBED_DIM", "embedding_distance", "path_embedding", "resample_path"]
