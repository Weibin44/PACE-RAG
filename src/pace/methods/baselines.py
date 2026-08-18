"""Pure ranking baselines reproduced from their published formulations."""

from __future__ import annotations

import numpy as np
import math


def _descending(values: np.ndarray) -> list[int]:
    return np.argsort(-np.asarray(values), kind="stable").tolist()


def rocchio_prf_order(
    query_similarity: np.ndarray,
    document_similarity: np.ndarray,
    feedback_order: list[int],
    *,
    alpha: float = 0.4,
    beta: float = 0.6,
    feedback_depth: int = 3,
) -> list[int]:
    """Vector-PRF Rocchio ranking using an exact normalized-vector Gram form."""
    query_similarity = np.asarray(query_similarity, dtype=np.float64)
    document_similarity = np.asarray(document_similarity, dtype=np.float64)
    feedback = np.asarray(feedback_order[:feedback_depth], dtype=np.int64)
    centroid_similarity = document_similarity[feedback].mean(axis=0)
    return _descending(alpha * query_similarity + beta * centroid_similarity)


def mmr_order(
    relevance: np.ndarray,
    document_similarity: np.ndarray,
    *,
    diversity: float,
    limit: int | None = None,
) -> list[int]:
    """Canonical greedy MMR: relevance minus maximum selected redundancy."""
    if not 0.0 <= diversity <= 1.0:
        raise ValueError("diversity must be in [0, 1]")
    relevance = np.asarray(relevance, dtype=np.float64)
    similarity = np.asarray(document_similarity, dtype=np.float64)
    count = len(relevance)
    limit = count if limit is None else min(limit, count)
    selected = [int(np.argmax(relevance))]
    remaining = np.ones(count, dtype=bool)
    remaining[selected[0]] = False
    redundancy = similarity[:, selected[0]].copy()
    while len(selected) < limit:
        scores = (1.0 - diversity) * relevance - diversity * redundancy
        scores[~remaining] = -np.inf
        index = int(np.argmax(scores))
        selected.append(index)
        remaining[index] = False
        redundancy = np.maximum(redundancy, similarity[:, index])
    return selected


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.exp(values - maximum).sum(axis=axis)
    )


def _dartboard_greedy(
    query_log_probabilities: np.ndarray,
    pair_log_probabilities: np.ndarray,
    limit: int,
) -> list[int]:
    """Log-space greedy search matching EmergenceAI/dartboard.py."""
    query_log_probabilities = np.asarray(query_log_probabilities, dtype=np.float64)
    pair_log_probabilities = np.asarray(pair_log_probabilities, dtype=np.float64)
    count = len(query_log_probabilities)
    limit = min(limit, count)
    selected = [int(np.argmax(query_log_probabilities))]
    maxima = pair_log_probabilities[selected[0]].copy()
    while len(selected) < limit:
        candidate_maxima = np.maximum(maxima[None, :], pair_log_probabilities)
        scores = _logsumexp(
            candidate_maxima + query_log_probabilities[None, :], axis=1
        )
        scores[selected] = -np.inf
        index = int(np.argmax(scores))
        selected.append(index)
        maxima = candidate_maxima[index]
    return selected


def _scaled_cosine_distance(similarity: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - (np.asarray(similarity) + 1.0) / 2.0, 0.0, 1.0)


def _log_normal(distance: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-5)
    return (
        -np.log(sigma)
        - 0.5 * np.log(2.0 * np.pi)
        - np.square(distance) / (2.0 * sigma * sigma)
    )


def dartboard_cosine_order(
    query_similarity: np.ndarray,
    document_similarity: np.ndarray,
    *,
    sigma: float,
    limit: int,
) -> list[int]:
    """Dartboard cosine variant from Pickett et al. (2024)."""
    query_logs = _log_normal(_scaled_cosine_distance(query_similarity), sigma)
    pair_logs = _log_normal(_scaled_cosine_distance(document_similarity), sigma)
    return _dartboard_greedy(query_logs, pair_logs, limit)


def dartboard_hybrid_order(
    reranker_scores: np.ndarray,
    document_similarity: np.ndarray,
    *,
    sigma: float,
    limit: int,
) -> list[int]:
    """Dartboard hybrid: cross-encoder query weights and cosine coverage."""
    sigma = max(float(sigma), 1e-5)
    query_logs = np.asarray(reranker_scores, dtype=np.float64) / sigma
    query_logs -= _logsumexp(query_logs, axis=0)
    coverage = np.clip((np.asarray(document_similarity) + 1.0) / 2.0, 1e-30, 1.0)
    return _dartboard_greedy(query_logs, np.log(coverage), limit)



def adaptive_k_cutoff(
    scores: np.ndarray,
    *,
    buffer: int = 5,
    search_fraction: float = 0.9,
    min_documents: int = 5,
) -> int:
    """Choose D from the largest adjacent dense-score gap (Adaptive-K baseline)."""
    values = np.asarray(scores, dtype=np.float32)
    if len(values) <= 1:
        return len(values)
    search_count = max(2, min(len(values), math.ceil(len(values) * search_fraction)))
    gap_index = int(np.argmax(values[: search_count - 1] - values[1:search_count]))
    return min(len(values), max(min_documents, gap_index + 1 + buffer))
