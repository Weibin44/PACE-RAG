"""Optional integration tests against the complete local datasets."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pace.data.batched import load_batched_examples
from pace.data.musique import load_musique_examples


DATA_ROOT_VALUE = os.environ.get("PACE_DATA_ROOT")

pytestmark = pytest.mark.skipif(
    DATA_ROOT_VALUE is None,
    reason="set PACE_DATA_ROOT to run real-data tests",
)


@pytest.mark.parametrize(
    ("dataset", "expected_queries", "expected_documents"),
    [
        ("hotpot", 1187, 100),
        ("2wiki", 2961, 100),
        ("musique", 2417, None),
    ],
)
def test_real_dataset_integrity(
    dataset,
    expected_queries,
    expected_documents,
):
    data_root = Path(DATA_ROOT_VALUE)

    if dataset == "hotpot":
        base = data_root / "hotpotqa" / "top100_complete"
        examples = load_batched_examples(
            base / "cohort",
            base / "cache",
            base / "cache" / "splade_similarity",
            candidate_limit=100,
        )
    elif dataset == "2wiki":
        base = (
            data_root
            / "2wikimultihopqa"
            / "top100_complete"
        )
        examples = load_batched_examples(
            base / "cohort",
            base / "cache",
            base / "cache" / "splade_similarity",
            candidate_limit=100,
        )
    else:
        base = data_root / "musique"
        examples = load_musique_examples(
            base / "musique_ans_v1.0_dev.jsonl",
            base / "cache",
            base / "cache" / "splade_similarity",
            candidate_limit=20,
        )

    query_ids = set()
    query_count = 0

    for query in examples:
        query.validate()
        query_count += 1

        assert query.query_id not in query_ids
        query_ids.add(query.query_id)

        covered_fact_ids = set().union(
            *(
                candidate.covered_fact_ids
                for candidate in query.candidates
            )
        )
        assert query.gold_fact_ids <= covered_fact_ids

        if expected_documents is not None:
            assert query.candidate_count == expected_documents
        else:
            assert 1 <= query.candidate_count <= 20

    assert query_count == expected_queries
    assert len(query_ids) == expected_queries