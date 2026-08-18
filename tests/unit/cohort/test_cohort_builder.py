"""Tests for batch-level cohort construction."""

import numpy as np
import pytest

from pace.cohort.builder import build_cohort_batch
from pace.cohort.schema import (
    CorpusPassage,
    GoldFact,
    GoldQuery,
)


class FakeCorpus:
    def __init__(self):
        self.fetch_calls = 0

    def fetch(self, indices):
        self.fetch_calls += 1
        return {
            index: CorpusPassage(
                corpus_index=index,
                document_id=f"doc-{index}",
                text=(
                    "A: supporting sentence."
                    if index == 0
                    else "B: unrelated sentence."
                ),
            )
            for index in set(indices)
        }


def make_query(query_id):
    return GoldQuery(
        query_id=query_id,
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


def test_build_cohort_batch_fetches_corpus_once():
    corpus = FakeCorpus()

    samples = build_cohort_batch(
        [make_query("q1"), make_query("q2")],
        np.array([[0, 1], [1, 0]]),
        np.array([[1.0, 0.5], [0.8, 0.4]]),
        corpus,
        "hotpot",
    )

    assert corpus.fetch_calls == 1
    assert len(samples) == 2
    assert samples[0].candidates[0].document_id == "doc-0"
    assert samples[1].candidates[0].retrieval_rank == 1


def test_build_cohort_batch_can_retain_complete_only():
    corpus = FakeCorpus()

    samples = build_cohort_batch(
        [make_query("complete"), make_query("incomplete")],
        np.array([[0], [1]]),
        np.array([[1.0], [0.5]]),
        corpus,
        "hotpot",
        retain_complete=True,
    )

    assert [
        sample.query_id
        for sample in samples
    ] == ["complete"]


def test_build_cohort_batch_checks_query_count():
    with pytest.raises(
        ValueError,
        match="query count",
    ):
        build_cohort_batch(
            [make_query("q1")],
            np.array([[0], [1]]),
            np.array([[1.0], [0.5]]),
            FakeCorpus(),
            "hotpot",
        )