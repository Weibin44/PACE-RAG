import numpy as np

from pace.methods.baselines import adaptive_k_cutoff, mmr_order


def test_mmr_avoids_redundant_document():
    relevance = np.array([1.0, 0.9, 0.8])
    similarity = np.array(
        [
            [1.0, 0.99, 0.0],
            [0.99, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    order = mmr_order(
        relevance,
        similarity,
        diversity=0.5,
        limit=2,
    )

    assert order == [0, 2]


def test_adaptive_k_uses_largest_score_gap():
    scores = np.array([1.0, 0.9, 0.8, 0.2, 0.1])

    cutoff = adaptive_k_cutoff(
        scores,
        buffer=0,
        search_fraction=1.0,
        min_documents=1,
    )

    assert cutoff == 3