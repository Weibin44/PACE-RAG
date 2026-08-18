import numpy as np

from pace.methods.pace import (
    greedy_evidence_frontloading_order,
    soft_anchor,
    soft_anchor_relevance_components,
)


def test_soft_anchor_outputs_are_valid():
    relevance = np.array([0.1, 0.4, 0.8], dtype=np.float32)
    similarity = np.array(
        [
            [1.0, 0.2, 0.3],
            [0.2, 1.0, 0.7],
            [0.3, 0.7, 1.0],
        ],
        dtype=np.float32,
    )

    query, anchor, weights, effective = soft_anchor_relevance_components(
        relevance, similarity
    )

    assert np.isclose(weights.sum(), 1.0)
    assert np.all((query >= 0) & (query <= 1))
    assert np.all((anchor >= 0) & (anchor <= 1))
    assert 1.0 <= effective <= len(relevance)


def test_zero_mad_remains_finite():
    relevance = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    similarity = np.eye(4, dtype=np.float32)

    combined, weights, effective = soft_anchor(
        relevance, similarity
    )

    assert np.isfinite(combined).all()
    assert np.isfinite(weights).all()
    assert np.isfinite(effective)
    assert np.isclose(weights.sum(), 1.0)


def test_greedy_evidence_frontloading_order():
    query_weights = np.array([1.0, 1.0], dtype=np.float32)
    document_features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    relevance = np.array([1.0, 1.0, 0.25], dtype=np.float32)

    order = greedy_evidence_frontloading_order(
        query_weights, document_features, relevance, limit=2
    )

    assert order == [0, 1]