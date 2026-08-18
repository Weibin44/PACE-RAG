"""Tests for corpus-wide evidence coverage."""

from pace.cohort.corpus_coverage import (
    audit_2wiki_corpus_coverage,
)


def make_example(
    query_id,
    title,
    sentence_index=0,
):
    return {
        "_id": query_id,
        "context": [
            [title, ["supporting sentence"]]
        ],
        "supporting_facts": [
            [title, sentence_index]
        ],
    }


def test_audit_2wiki_corpus_coverage():
    examples = [
        make_example("complete", "A"),
        make_example("missing", "B"),
        make_example(
            "invalid",
            "C",
            sentence_index=5,
        ),
    ]
    corpus = [
        "A: supporting sentence.",
        "Other: unrelated text.",
    ]

    report = audit_2wiki_corpus_coverage(
        examples,
        corpus,
    )

    assert report["dev_queries"] == 3
    assert report["corpus_chunks"] == 2
    assert report["invalid_sentence_labels"] == 1
    assert report["queries_with_all_facts"] == 1
    assert report["complete_query_ids"] == [
        "complete"
    ]
