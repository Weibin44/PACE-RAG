"""Tests for the BERGEN corpus adapter."""

import pytest

from pace.cohort.corpus import BergenCorpus


class FakeDataset:
    def __init__(self, include_ids=True):
        self.column_names = (
            ["id", "content"]
            if include_ids
            else ["content"]
        )
        self.rows = [
            {"id": "d0", "content": "zero"},
            {"id": "d1", "content": "one"},
            {"id": "d2", "content": "two"},
        ]

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            indices = range(
                *indices.indices(len(self.rows))
            )
        output = {
            "content": [
                self.rows[index]["content"]
                for index in indices
            ]
        }
        if "id" in self.column_names:
            output["id"] = [
                self.rows[index]["id"]
                for index in indices
            ]
        return output
    
    def __len__(self):
        return len(self.rows)


def test_fetches_unique_sorted_passages():
    corpus = BergenCorpus(FakeDataset())

    passages = corpus.fetch([2, 0, 2])

    assert list(passages) == [0, 2]
    assert passages[0].document_id == "d0"
    assert passages[2].text == "two"


def test_uses_corpus_index_without_id_column():
    corpus = BergenCorpus(
        FakeDataset(include_ids=False)
    )

    passage = corpus.fetch([1])[1]

    assert passage.document_id == "1"


def test_rejects_negative_corpus_index():
    corpus = BergenCorpus(FakeDataset())

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        corpus.fetch([-1])

def test_iter_contents_preserves_corpus_order():
    corpus = BergenCorpus(FakeDataset())

    contents = list(
        corpus.iter_contents(batch_size=2)
    )

    assert contents == [
        "zero",
        "one",
        "two",
    ]


def test_iter_contents_requires_positive_batch_size():
    corpus = BergenCorpus(FakeDataset())

    with pytest.raises(
        ValueError,
        match="batch size must be positive",
    ):
        list(
            corpus.iter_contents(batch_size=0)
        )

