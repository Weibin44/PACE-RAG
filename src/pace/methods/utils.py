"""Dataset-independent score normalization utilities."""
from __future__ import annotations

import math
import numpy as np


def normalize_nonnegative_by_max(scores: np.ndarray) -> np.ndarray:
    """Clip negative scores and divide by the largest positive score."""
    values = np.maximum(np.asarray(scores, dtype=np.float32), 0)
    maximum = float(values.max())
    return np.zeros_like(values) if maximum <= 1e-12 else values / maximum


def minmax_normalize(
    scores: np.ndarray,
    *,
    constant_value: float = 0.0,
) -> np.ndarray:
    """Map scores to [0, 1] with explicit constant-score behavior."""
    values = np.asarray(scores, dtype=np.float32)
    low = float(values.min())
    span = float(values.max() - low)

    if span <= 1e-12:
        return np.full_like(values, constant_value)

    return (values - low) / span


# def normalize_relevance(scores: np.ndarray, mode: str) -> np.ndarray:
#     """Map query-local scores to stable non-negative relevance weights."""
#     values = np.asarray(scores, dtype=np.float32)
#     if mode == "max":
#         positive = np.maximum(values, 0)
#         return positive / max(float(positive.max()), 1e-12)
#     if mode == "minmax":
#         low, high = float(values.min()), float(values.max())
#         return np.ones_like(values) if high <= low else (values - low) / (high - low)
#     raise ValueError(f"unknown relevance normalization: {mode}")


# def normalize_scores(values: np.ndarray, method: str) -> np.ndarray:
#     """Normalize one query's scores without fitted parameters."""
#     values = np.asarray(values, dtype=np.float32)
#     if method == "minmax":
#         span = float(values.max() - values.min())
#         return (
#             np.zeros_like(values) if span <= 1e-12 else (values - values.min()) / span
#         )
#     raise ValueError(f"unknown normalization method: {method}")