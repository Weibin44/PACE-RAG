import numpy as np
import pytest

from pace.data.schema import Candidate, EvidenceExample
from pace.evaluation.rankings import (
    METHODS,
    RankingParameters,
    rank_example,
)


def make_example() -> EvidenceExample:
    candidates = tuple(
        Candidate(
            document_id=f"doc-{index}",
            text=f"Document {index}",
            retriever_score=score,
            covered_fact_ids=frozenset(),
        )
        for index, score in enumerate([0.9, 0.4, 0.7])
    )

    return EvidenceExample(
        query_id="q1",
        question="Example question",
        gold_fact_ids=frozenset(),
        candidates=candidates,
        reranker_scores=np.array(
            [0.2, 0.8, 0.5],
            dtype=np.float32,
        ),
        query_weights=np.array([1.0, 0.5], dtype=np.float32),
        document_features=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.6, 0.6],
            ],
            dtype=np.float32,
        ),
        document_similarity=np.array(
            [
                [1.0, 0.1, 0.5],
                [0.1, 1.0, 0.3],
                [0.5, 0.3, 1.0],
            ],
            dtype=np.float32,
        ),
        query_similarity=np.array(
            [0.9, 0.4, 0.7],
            dtype=np.float32,
        ),
    )


PARAMETERS = RankingParameters(
    rocchio_alpha=0.4,
    rocchio_depth=1,
    mmr_diversity=0.3,
    dartboard_sigma=0.5,
)


@pytest.mark.parametrize("stage", ["D", "K"])
@pytest.mark.parametrize("method", METHODS)
def test_all_methods_return_valid_prefix(stage, method):
    order = rank_example(
        make_example(),
        stage,
        method,
        PARAMETERS,
        limit=2,
    )

    assert len(order) == 2
    assert len(set(order)) == 2
    assert set(order) <= {0, 1, 2}


def test_standard_uses_stage_specific_scores():
    example = make_example()

    assert rank_example(
        example,
        "D",
        "standard",
        PARAMETERS,
    ) == [0, 2, 1]

    assert rank_example(
        example,
        "K",
        "standard",
        PARAMETERS,
    ) == [1, 2, 0]


def test_unknown_method_raises_error():
    with pytest.raises(ValueError, match="unknown ranking method"):
        rank_example(
            make_example(),
            "D",
            "unknown",
            PARAMETERS,
        )