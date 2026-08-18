"""Tests for cohort materialization."""

from pace.cohort.materialize import (
    materialize_sample,
    retain_complete_samples,
)
from pace.cohort.schema import (
    GoldFact,
    GoldQuery,
    RetrievedPassage,
)


QUERY = GoldQuery(
    query_id="q1",
    question="question",
    facts=(
        GoldFact(
            fact_id="A::0",
            title="A",
            sentence_index=0,
            text="supporting sentence",
        ),
    ),
)


def make_passage(
    document_id: str,
    text: str,
    rank: int,
) -> RetrievedPassage:
    return RetrievedPassage(
        document_id=document_id,
        text=text,
        retriever_score=1.0 / rank,
        retrieval_rank=rank,
        corpus_index=rank,
    )


def test_materialize_labels_covered_facts():
    sample = materialize_sample(
        QUERY,
        [
            make_passage(
                "doc-1",
                "A: supporting sentence.",
                1,
            ),
            make_passage(
                "doc-2",
                "B: unrelated sentence.",
                2,
            ),
        ],
        "hotpot",
    )

    assert sample.candidates[
        0
    ].covered_fact_ids == frozenset({"A::0"})
    assert not sample.candidates[1].covered_fact_ids
    assert sample.complete_evidence_retrieved


def test_materialize_detects_incomplete_sample():
    sample = materialize_sample(
        QUERY,
        [
            make_passage(
                "doc-1",
                "B: unrelated sentence.",
                1,
            )
        ],
        "hotpot",
    )

    assert not sample.complete_evidence_retrieved


def test_retain_complete_samples():
    complete = materialize_sample(
        QUERY,
        [
            make_passage(
                "doc-1",
                "A: supporting sentence.",
                1,
            )
        ],
        "hotpot",
    )
    incomplete = materialize_sample(
        QUERY,
        [
            make_passage(
                "doc-2",
                "B: unrelated sentence.",
                1,
            )
        ],
        "hotpot",
    )

    assert retain_complete_samples(
        [complete, incomplete]
    ) == (complete,)
