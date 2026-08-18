"""Tests for raw gold-evidence loading."""

import json

import pytest

from pace.cohort.gold import (
    load_2wiki_gold_queries,
    load_2wiki_questions,
    load_hotpot_gold_queries,
    resolve_gold_facts,
)


def test_load_hotpot_uses_intersection_and_stable_order(
    tmp_path,
):
    evaluation = tmp_path / "evaluation.json"
    labels = tmp_path / "labels.json"

    evaluation.write_text(
        json.dumps(
            [
                {"q_id": "q2", "question": "second"},
                {"q_id": "q1", "question": "first"},
                {"q_id": "missing", "question": "unused"},
            ]
        ),
        encoding="utf-8",
    )
    labels.write_text(
        json.dumps(
            [
                {
                    "_id": query_id,
                    "context": [["A", ["fact"]]],
                    "supporting_facts": [["A", 0]],
                }
                for query_id in ("q1", "q2")
            ]
        ),
        encoding="utf-8",
    )

    first = load_hotpot_gold_queries(
        evaluation,
        labels,
        seed=2026,
    )
    second = load_hotpot_gold_queries(
        evaluation,
        labels,
        seed=2026,
    )

    assert first == second
    assert {query.query_id for query in first} == {
        "q1",
        "q2",
    }
    assert first[0].facts[0].text == "fact"


def test_unresolvable_supporting_fact_is_explicit():
    example = {
        "_id": "q1",
        "context": [["A", ["only sentence"]]],
        "supporting_facts": [["A", 2]],
    }

    with pytest.raises(
        ValueError,
        match="unresolvable supporting fact",
    ):
        resolve_gold_facts(example)


def test_load_2wiki_questions_preserves_dev_order(
    tmp_path,
):
    path = tmp_path / "dev.json"
    path.write_text(
        json.dumps(
            [
                {
                    "_id": "q2",
                    "question": "second",
                    "context": [],
                    "supporting_facts": [],
                },
                {
                    "_id": "q1",
                    "question": "first",
                    "context": [],
                    "supporting_facts": [],
                },
            ]
        ),
        encoding="utf-8",
    )

    queries = load_2wiki_questions(path)

    assert [query.query_id for query in queries] == [
        "q2",
        "q1",
    ]


def test_load_selected_2wiki_gold_queries(
    tmp_path,
):
    path = tmp_path / "dev.json"
    path.write_text(
        json.dumps(
            [
                {
                    "_id": query_id,
                    "question": query_id,
                    "context": [["A", ["fact"]]],
                    "supporting_facts": [["A", 0]],
                }
                for query_id in ("q1", "q2")
            ]
        ),
        encoding="utf-8",
    )

    queries = load_2wiki_gold_queries(
        path,
        ["q2", "q1"],
    )

    assert [query.query_id for query in queries] == [
        "q2",
        "q1",
    ]
    assert queries[0].facts[0].text == "fact"
