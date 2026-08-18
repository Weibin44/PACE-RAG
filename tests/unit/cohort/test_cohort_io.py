"""Tests for legacy cohort input adapters."""

import json

from pace.cohort.io import (
    load_cohort_batch,
    write_cohort_batch,
)

def test_load_hotpot_candidate(tmp_path):
    path = tmp_path / "samples_sentence_labels.json"
    path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "q_id": "q1",
                        "question": "question",
                        "facts": [
                            {
                                "fact_id": "A::0",
                                "title": "A",
                                "sentence_id": 0,
                                "text": "fact",
                            }
                        ],
                        "candidates": [
                            {
                                "doc_id": "doc-10",
                                "corpus_index": 10,
                                "dense_rank": 1,
                                "dense_score": 2.5,
                                "text": "passage",
                                "covered_facts": ["A::0"],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    sample = load_cohort_batch(path)[0]
    candidate = sample.candidates[0]

    assert sample.complete_evidence_retrieved
    assert candidate.document_id == "doc-10"
    assert candidate.corpus_index == 10
    assert candidate.retrieval_rank == 1


def test_load_2wiki_candidate(tmp_path):
    path = tmp_path / "samples_sentence_labels.json"
    path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "q_id": "q2",
                        "question": "question",
                        "facts": [],
                        "candidates": [
                            {
                                "doc_index": 25,
                                "dense_score": 1.25,
                                "text": "passage",
                                "covered_facts": [],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidate = load_cohort_batch(path)[0].candidates[0]

    assert candidate.document_id == "25"
    assert candidate.corpus_index == 25
    assert candidate.retrieval_rank == 1

def test_cohort_batch_round_trip(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "q_id": "q1",
                        "question": "question",
                        "facts": [
                            {
                                "fact_id": "A::0",
                                "title": "A",
                                "sentence_id": 0,
                                "text": "fact",
                            }
                        ],
                        "candidates": [
                            {
                                "doc_id": "doc-1",
                                "corpus_index": 1,
                                "dense_rank": 1,
                                "dense_score": 2.0,
                                "text": "passage",
                                "covered_facts": ["A::0"],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    samples = load_cohort_batch(source)
    output = tmp_path / "output.json"

    write_cohort_batch(output, samples)

    assert load_cohort_batch(output) == samples
    assert output.read_text(encoding="utf-8").endswith(
        "\n"
    )
