"""Tests for cohort data schemas."""

from pace.cohort.schema import (
    CohortCandidate,
    CohortSample,
    GoldFact,
    GoldQuery,
)


def make_sample(
    covered_fact_ids: frozenset[str],
) -> CohortSample:
    return CohortSample(
        query_id="q1",
        question="question",
        facts=(
            GoldFact(
                fact_id="A::0",
                title="A",
                sentence_index=0,
                text="first fact",
            ),
            GoldFact(
                fact_id="B::1",
                title="B",
                sentence_index=1,
                text="second fact",
            ),
        ),
        candidates=(
            CohortCandidate(
                document_id="10",
                text="candidate",
                retriever_score=1.5,
                retrieval_rank=1,
                covered_fact_ids=covered_fact_ids,
            ),
        ),
    )


def test_complete_evidence_retrieved():
    sample = make_sample(
        frozenset({"A::0", "B::1"})
    )

    assert sample.gold_fact_ids == frozenset(
        {"A::0", "B::1"}
    )
    assert sample.complete_evidence_retrieved


def test_incomplete_evidence_retrieved():
    sample = make_sample(frozenset({"A::0"}))

    assert sample.retrieved_fact_ids == frozenset({"A::0"})
    assert not sample.complete_evidence_retrieved


def test_gold_query_has_no_retrieval_state():
    query = GoldQuery(
        query_id="q1",
        question="question",
        facts=(
            GoldFact(
                fact_id="A::0",
                title="A",
                sentence_index=0,
                text="fact",
            ),
        ),
    )

    assert query.gold_fact_ids == frozenset({"A::0"})
    assert not hasattr(query, "candidates")

