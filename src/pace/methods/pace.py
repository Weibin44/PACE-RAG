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



def frontload_evidence(
    query_features: np.ndarray,
    document_features: np.ndarray,
    query_relevance: np.ndarray,
    document_similarity: np.ndarray,
    budget: int,
) -> list[int]:
    """Return evidence-frontloaded document indices.

    Shapes:
        query_features: [features]
        document_features: [documents, features]
        query_relevance: [documents]
        document_similarity: [documents, documents]
    """
    query_features = np.asarray(query_features, dtype=np.float32)
    document_features = np.asarray(document_features, dtype=np.float32)
    query_relevance = np.asarray(query_relevance, dtype=np.float32)
    document_similarity = np.asarray(document_similarity, dtype=np.float32)

    if query_features.ndim != 1:
        raise ValueError("query_features must have shape [features]")

    if document_features.ndim != 2:
        raise ValueError(
            "document_features must have shape [documents, features]"
        )

    document_count = document_features.shape[0]

    if document_features.shape[1] != query_features.size:
        raise ValueError(
            "query and document feature dimensions must match"
        )

    if query_relevance.shape != (document_count,):
        raise ValueError(
            "query_relevance must have shape [documents]"
        )

    if document_similarity.shape != (
        document_count,
        document_count,
    ):
        raise ValueError(
            "document_similarity must have shape "
            "[documents, documents]"
        )

    if not 1 <= budget <= document_count:
        raise ValueError(
            "budget must be between 1 and the document count"
        )

    relevance, _, _ = soft_anchor(
        query_relevance,
        document_similarity,
    )

    return greedy_evidence_frontloading_order(
        query_features,
        document_features,
        relevance,
        limit=budget,
    )
