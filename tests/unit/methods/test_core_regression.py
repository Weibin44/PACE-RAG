import numpy as np

from pace.methods.baselines import (
    dartboard_cosine_order,
    mmr_order,
    rocchio_prf_order,
)
from pace.methods.pace import (
    greedy_evidence_frontloading_order,
    soft_anchor,
    soft_anchor_relevance_components,
)


RELEVANCE = np.array([0.15, 0.8, 0.45, 0.3], dtype=np.float32)

SIMILARITY = np.array(
    [
        [1.0, 0.2, 0.5, 0.1],
        [0.2, 1.0, 0.6, 0.3],
        [0.5, 0.6, 1.0, 0.4],
        [0.1, 0.3, 0.4, 1.0],
    ],
    dtype=np.float32,
)

QUERY_WEIGHTS = np.array([0.7, 0.3], dtype=np.float32)

DOCUMENT_FEATURES = np.array(
    [
        [1.0, 0.0],
        [0.2, 0.9],
        [0.8, 0.5],
        [0.0, 0.7],
    ],
    dtype=np.float32,
)


def test_soft_anchor_regression():
    query, anchor, weights, effective = soft_anchor_relevance_components(
        RELEVANCE, SIMILARITY
    )
    combined, _, _ = soft_anchor(RELEVANCE, SIMILARITY)

    np.testing.assert_allclose(
        query,
        [0.0, 1.0, 0.46153846, 0.23076925],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        anchor,
        [0.0, 0.72557253, 1.0, 0.22694527],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        weights,
        [0.01145407, 0.87277573, 0.08463477, 0.03113539],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        combined,
        [0.0, 1.0, 1.0, 0.40534258],
        rtol=1e-6,
    )
    assert np.isclose(effective, 1.2986994207907667)


def test_frontloading_order_regression():
    combined, _, _ = soft_anchor(RELEVANCE, SIMILARITY)

    order = greedy_evidence_frontloading_order(
        QUERY_WEIGHTS,
        DOCUMENT_FEATURES,
        combined,
        limit=4,
    )

    assert order == [2, 1, 0, 3]


def test_baseline_order_regression():
    query_similarity = np.array([0.9, 0.7, 0.5, 0.2])

    assert rocchio_prf_order(
        query_similarity,
        SIMILARITY,
        [0, 1, 2, 3],
        alpha=0.7,
        beta=0.3,
        feedback_depth=2,
    ) == [0, 1, 2, 3]

    assert mmr_order(
        RELEVANCE,
        SIMILARITY,
        diversity=0.35,
        limit=4,
    ) == [1, 3, 2, 0]

    assert dartboard_cosine_order(
        query_similarity,
        SIMILARITY,
        sigma=0.1,
        limit=4,
    ) == [0, 1, 2, 3]