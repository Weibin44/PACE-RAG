import numpy as np
import pytest

from pace.data.schema import Candidate, EvidenceExample


def make_example() -> EvidenceExample:
    candidates = (
        Candidate("d1", "document one", 1.0, frozenset({"f1"})),
        Candidate("d2", "document two", 0.5, frozenset({"f2"})),
    )

    return EvidenceExample(
        query_id="q1",
        question="example question",
        gold_fact_ids=frozenset({"f1", "f2"}),
        candidates=candidates,
        reranker_scores=np.zeros(2),
        query_weights=np.zeros(3),
        document_features=np.zeros((2, 3)),
        document_similarity=np.eye(2),
        query_similarity=np.zeros(2),
    )


def test_valid_example():
    example = make_example()

    example.validate()

    assert example.candidate_count == 2


def test_invalid_similarity_shape():
    example = make_example()
    object.__setattr__(
        example,
        "document_similarity",
        np.zeros((2, 3)),
    )

    with pytest.raises(ValueError):
        example.validate()