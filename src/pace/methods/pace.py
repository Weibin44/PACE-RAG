"""Core PACE relevance fusion and evidence-frontloading algorithms.""" 
from __future__ import annotations

import numpy as np
from .utils import minmax_normalize


def greedy_evidence_frontloading_order(
    query_weights: np.ndarray,
    document_features: np.ndarray,
    relevance: np.ndarray,
    limit: int,
) -> list[int]:
    """Return the greedy ranking prefix up to ``limit``."""
    query_weights = np.asarray(query_weights, dtype=np.float32)
    document_features = np.asarray(document_features, dtype=np.float32)
    relevance = np.asarray(relevance, dtype=np.float32)
    quality = document_features * np.sqrt(np.maximum(relevance, 0))[:, None]
    covered = np.zeros_like(query_weights)
    remaining = np.ones(len(relevance), dtype=bool)
    order = []
    while remaining.any() and len(order) < min(limit, len(relevance)):
        gains = (np.maximum(quality - covered, 0) * query_weights).sum(axis=1)
        gains[~remaining] = -np.inf
        selected = int(np.argmax(gains))
        order.append(selected)
        remaining[selected] = False
        covered = np.maximum(covered, quality[selected])
    return order



def soft_anchor_relevance_components(
    base_relevance: np.ndarray,
    similarities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return normalized query/anchor relevance and robust soft-anchor weights."""
    base = np.maximum(np.asarray(base_relevance, dtype=np.float32), 0)
    similarity = np.maximum(np.asarray(similarities, dtype=np.float32), 0)
    median = float(np.median(base))
    mad = float(np.median(np.abs(base - median)))
    # if mad <= 1e-12:
    #     weights = np.full(len(base), 1.0 / len(base), dtype=np.float32)
    # else:
    #     logits = (base - median) / mad
    #     logits -= logits.max()
    #     weights = np.exp(logits).astype(np.float32)
    #     weights /= weights.sum()
    epsilon = 1e-12
    logits = (base - median) / (mad + epsilon)
    logits -= logits.max()
    weights = np.exp(logits).astype(np.float32)
    weights /= weights.sum()

    # Each target document excludes itself from its soft-anchor estimate.
    numerators = weights @ similarity - weights * np.diag(similarity)
    denominators = 1.0 - weights
    anchor = np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators, dtype=np.float32),
        where=denominators > 1e-12,
    )
    base_normalized = minmax_normalize(base)
    anchor_normalized = minmax_normalize(anchor)
    effective_anchors = 1.0 / float(np.square(weights).sum())
    return base_normalized, anchor_normalized, weights, effective_anchors


def soft_anchor(
    base_relevance: np.ndarray,
    similarities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Parameter-free robust soft anchors followed by relevance fusion."""
    base, anchor, weights, effective_anchors = soft_anchor_relevance_components(
        base_relevance, similarities
    )
    combined = 1.0 - (1.0 - base) * (1.0 - anchor)
    return combined.astype(np.float32), weights, effective_anchors
