"""Tests for 2Wiki cohort materialization."""

import json

import numpy as np
import pytest

from pace.cohort.io import load_cohort_batch
from pace.cohort.schema import CorpusPassage
from pace.cohort.two_wiki import materialize_2wiki_cohort


class FakeCorpus:
    def fetch(self, indices):
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


def write_inputs(tmp_path):
    labels = tmp_path / "dev.json"
    retrieval = tmp_path / "retrieval"
    report = tmp_path / "corpus_coverage.json"
    retrieval.mkdir()

    labels.write_text(
        json.dumps(
            [
                {
                    "_id": query_id,
                    "question": query_id,
                    "context": [
                        ["A", ["supporting sentence"]]
                    ],
                    "supporting_facts": [["A", 0]],
                }
                for query_id in ("q1", "q2")
            ]
        ),
        encoding="utf-8",
    )

    report.write_text(
        json.dumps(
            {"complete_query_ids": ["q1", "q2"]}
        ),
        encoding="utf-8",
    )

    np.savez(
        retrieval / "dense_top100.part00-of-01.npz",
        indices=np.array([[0], [1]]),
        scores=np.array(
            [[1.0], [0.5]],
            dtype=np.float32,
        ),
    )

    return labels, retrieval, report


def test_materialize_computes_coverage_and_filters(
    tmp_path,
):
    labels, retrieval, report = write_inputs(tmp_path)
    output = tmp_path / "cohort"

    count = materialize_2wiki_cohort(
        labels,
        retrieval,
        report,
        output,
        FakeCorpus(),
        num_parts=1,
    )

    samples = load_cohort_batch(
        output
        / "batches/part00-of-01"
        / "samples_sentence_labels.json"
    )
    coverage = json.loads(
        (
            output
            / "coverage"
            / "top100_coverage.part00-of-01.json"
        ).read_text(encoding="utf-8")
    )

    assert count == 1
    assert [sample.query_id for sample in samples] == [
        "q1"
    ]
    assert coverage["top_k_complete_queries"] == 1
    assert [
        row["all_facts_covered"]
        for row in coverage["per_query"]
    ] == [True, False]


def test_materialize_detects_row_count_mismatch(
    tmp_path,
):
    labels, retrieval, report = write_inputs(tmp_path)

    np.savez(
        retrieval / "dense_top100.part00-of-01.npz",
        indices=np.array([[0]]),
        scores=np.array([[1.0]], dtype=np.float32),
    )

    with pytest.raises(
        ValueError,
        match="retrieval contains 1 rows, expected 2",
    ):
        materialize_2wiki_cohort(
            labels,
            retrieval,
            report,
            tmp_path / "cohort",
            FakeCorpus(),
            num_parts=1,
        )
