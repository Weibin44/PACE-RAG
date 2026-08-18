"""Regression tests for real cohort reconstruction."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pace.cohort.builder import build_cohort_batch
from pace.cohort.io import load_cohort_batch
from pace.cohort.retrieval import (
    load_retrieval_results,
)
from pace.cohort.schema import (
    CorpusPassage,
    GoldQuery,
)


DATA_ROOT_VALUE = os.environ.get("PACE_DATA_ROOT")

pytestmark = pytest.mark.skipif(
    DATA_ROOT_VALUE is None,
    reason="set PACE_DATA_ROOT to run real cohort tests",
)


class LookupCorpus:
    """Corpus reconstructed from an existing cohort."""

    def __init__(self, samples):
        self.passages = {}

        for sample in samples:
            for candidate in sample.candidates:
                assert candidate.corpus_index is not None
                self.passages[
                    candidate.corpus_index
                ] = CorpusPassage(
                    corpus_index=candidate.corpus_index,
                    document_id=candidate.document_id,
                    text=candidate.text,
                )

    def fetch(self, indices):
        return {
            index: self.passages[index]
            for index in set(indices)
        }


def gold_queries(samples):
    return [
        GoldQuery(
            query_id=sample.query_id,
            question=sample.question,
            facts=sample.facts,
        )
        for sample in samples
    ]


def test_rebuild_hotpot_batch():
    data_root = Path(DATA_ROOT_VALUE)
    batch = (
        data_root
        / "hotpotqa/top100_complete/cohort/batches"
        / "00000_00100"
    )

    expected = load_cohort_batch(
        batch / "samples_sentence_labels.json"
    )
    indices, scores = load_retrieval_results(
        batch / "dense_top100.npz"
    )

    actual = build_cohort_batch(
        gold_queries(expected),
        indices,
        scores,
        LookupCorpus(expected),
        "hotpot",
    )

    assert actual == expected


def test_rebuild_2wiki_part():
    data_root = Path(DATA_ROOT_VALUE)
    root = (
        data_root
        / "2wikimultihopqa/top100_complete"
    )
    part = "part00-of-08"

    expected = load_cohort_batch(
        root
        / "cohort/batches"
        / part
        / "samples_sentence_labels.json"
    )
    indices, scores = load_retrieval_results(
        root
        / "corpus_coverage"
        / f"dense_top100.{part}.npz"
    )
    coverage = json.loads(
        (
            root
            / "corpus_coverage"
            / f"top100_coverage.{part}.json"
        ).read_text(encoding="utf-8")
    )
    keep = [
        position
        for position, row in enumerate(
            coverage["per_query"]
        )
        if row["all_facts_covered"]
    ]

    actual = build_cohort_batch(
        gold_queries(expected),
        indices[keep],
        scores[keep],
        LookupCorpus(expected),
        "2wiki",
    )

    assert len(keep) == len(expected)
    assert actual == expected