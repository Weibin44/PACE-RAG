import numpy as np

from pace.data.schema import Candidate, EvidenceExample
from pace.evaluation.metrics import evidence_scores


def make_example() -> EvidenceExample:
    candidates = (
        Candidate(
            document_id="doc-1",
            text="First document",
            retriever_score=0.9,
            covered_fact_ids=frozenset({"fact-1"}),
        ),
        Candidate(
            document_id="doc-2",
            text="Irrelevant document",
            retriever_score=0.8,
            covered_fact_ids=frozenset(),
        ),
        Candidate(
            document_id="doc-3",
            text="Third document",
            retriever_score=0.7,
            covered_fact_ids=frozenset({"fact-2"}),
        ),
    )

    return EvidenceExample(
        query_id="q1",
        question="Example question",
        gold_fact_ids=frozenset({"fact-1", "fact-2"}),
        candidates=candidates,
        reranker_scores=np.array(
            [0.9, 0.8, 0.7],
            dtype=np.float32,
        ),
        query_weights=np.array([1.0], dtype=np.float32),
        document_features=np.ones((3, 1), dtype=np.float32),
        document_similarity=np.eye(3, dtype=np.float32),
        query_similarity=np.array(
            [0.9, 0.8, 0.7],
            dtype=np.float32,
        ),
    )


def test_evidence_scores_complete_recall():
    scores = evidence_scores(make_example(), [0, 2])

    assert scores == {
        "complete_evidence_recall": 1.0,
        "supporting_fact_recall": 1.0,
        "precision": 1.0,
        "returned_k": 2,
    }


def test_evidence_scores_partial_recall():
    scores = evidence_scores(make_example(), [0, 1])

    assert scores == {
        "complete_evidence_recall": 0.0,
        "supporting_fact_recall": 0.5,
        "precision": 0.5,
        "returned_k": 2,
    }


def test_evidence_scores_empty_selection():
    scores = evidence_scores(make_example(), [])

    assert scores == {
        "complete_evidence_recall": 0.0,
        "supporting_fact_recall": 0.0,
        "precision": 0.0,
        "returned_k": 0,
    }


# from pace.evaluation.metrics import evidence_scores


# def test_evidence_scores():
#     sample = {
#         "facts": [
#             {"fact_id": "fact-1"},
#             {"fact_id": "fact-2"},
#         ],
#         "candidates": [
#             {"covered_facts": ["fact-1"]},
#             {"covered_facts": []},
#             {"covered_facts": ["fact-2"]},
#         ],
#     }

#     partial = evidence_scores(sample, [0])
#     complete = evidence_scores(sample, [0, 2])

#     assert partial["complete_evidence_recall"] == 0.0
#     assert partial["supporting_fact_recall"] == 0.5
#     assert complete["complete_evidence_recall"] == 1.0
#     assert complete["supporting_fact_recall"] == 1.0