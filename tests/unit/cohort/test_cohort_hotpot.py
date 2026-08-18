"""Tests for HotpotQA cohort materialization."""

import json

import numpy as np

from pace.cohort.hotpot import (
    materialize_hotpot_cohort,
)
from pace.cohort.io import load_cohort_batch
from pace.cohort.schema import CorpusPassage


class FakeCorpus:
    def fetch(self, indices):
        return {
            index: CorpusPassage(
                corpus_index=index,
                document_id=f"doc-{index}",
                text="A: supporting sentence.",
            )
            for index in set(indices)
        }


def test_materialize_hotpot_cohort(tmp_path):
    evaluation = tmp_path / "evaluation.json"
    labels = tmp_path / "labels.json"
    retrieval = tmp_path / "retrieval"
    output = tmp_path / "cohort"

    evaluation.write_text(
        json.dumps(
            [
                {"q_id": "q1", "question": "first"},
                {"q_id": "q2", "question": "second"},
            ]
        ),
        encoding="utf-8",
    )
    labels.write_text(
        json.dumps(
            [
                {
                    "_id": query_id,
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

    retrieval_file = (
        retrieval
        / "00000_00002"
        / "dense_top100.npz"
    )
    retrieval_file.parent.mkdir(parents=True)
    np.savez(
        retrieval_file,
        indices=np.array([[0], [0]]),
        scores=np.array(
            [[1.0], [0.9]],
            dtype=np.float32,
        ),
    )

    count = materialize_hotpot_cohort(
        evaluation,
        labels,
        retrieval,
        output,
        FakeCorpus(),
        batch_size=2,
    )

    samples = load_cohort_batch(
        output
        / "batches/00000_00002"
        / "samples_sentence_labels.json"
    )

    assert count == 2
    assert len(samples) == 2
    assert all(
        sample.complete_evidence_retrieved
        for sample in samples
    )
