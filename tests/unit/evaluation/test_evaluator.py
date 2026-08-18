import csv

import numpy as np

from pace.data.schema import Candidate, EvidenceExample
from pace.evaluation.evaluator import (
    evaluate_curves,
    write_curves_csv,
)
from pace.evaluation.rankings import RankingParameters
from pace.evaluation.evaluator import evaluate_adaptive_k

PARAMETERS = RankingParameters(
    rocchio_alpha=0.4,
    rocchio_depth=1,
    mmr_diversity=0.3,
    dartboard_sigma=0.5,
)


def make_query() -> EvidenceExample:
    candidates = (
        Candidate(
            document_id="doc-1",
            text="First evidence",
            retriever_score=0.9,
            covered_fact_ids=frozenset({"fact-1"}),
        ),
        Candidate(
            document_id="doc-2",
            text="Second evidence",
            retriever_score=0.8,
            covered_fact_ids=frozenset({"fact-2"}),
        ),
        Candidate(
            document_id="doc-3",
            text="Irrelevant",
            retriever_score=0.1,
            covered_fact_ids=frozenset(),
        ),
    )

    return EvidenceExample(
        query_id="q1",
        question="Example question",
        gold_fact_ids=frozenset({"fact-1", "fact-2"}),
        candidates=candidates,
        reranker_scores=np.array(
            [0.9, 0.8, 0.1],
            dtype=np.float32,
        ),
        query_weights=np.array([1.0], dtype=np.float32),
        document_features=np.ones((3, 1), dtype=np.float32),
        document_similarity=np.eye(3, dtype=np.float32),
        query_similarity=np.array(
            [0.9, 0.8, 0.1],
            dtype=np.float32,
        ),
    )


def test_evaluate_standard_curve():
    result = evaluate_curves(
        [make_query()],
        "D",
        PARAMETERS,
        cutoffs=[1, 2],
        methods=["standard"],
    )

    assert result.query_count == 1
    assert result.rows[0] == {
        "stage": "D",
        "method": "standard",
        "cutoff": 1,
        "complete_evidence_recall": 0.0,
        "supporting_fact_recall": 0.5,
        "precision": 1.0,
    }
    assert result.rows[1] == {
        "stage": "D",
        "method": "standard",
        "cutoff": 2,
        "complete_evidence_recall": 1.0,
        "supporting_fact_recall": 1.0,
        "precision": 1.0,
    }


def test_write_curves_csv(tmp_path):
    result = evaluate_curves(
        [make_query()],
        "K",
        PARAMETERS,
        cutoffs=[1],
        methods=["standard"],
    )

    output = tmp_path / "curves.csv"
    write_curves_csv(output, result)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["stage"] == "K"
    assert rows[0]["method"] == "standard"
    assert rows[0]["cutoff"] == "1"


def test_evaluate_adaptive_k():
    result = evaluate_adaptive_k(
        [make_query()],
        buffer=0,
        search_fraction=1.0,
        min_documents=1,
    )

    assert result["stage"] == "D"
    assert result["method"] == "adaptive_k"
    assert result["mean_cutoff"] == 2.0
    assert result["complete_evidence_recall"] == 1.0
    assert result["supporting_fact_recall"] == 1.0