"""Tests for retrieval-result loading and conversion."""

import numpy as np
import pytest

from pace.cohort.retrieval import (
    load_retrieval_results,
    retrieved_passages_from_row,
    save_retrieval_results,
)
from pace.cohort.schema import CorpusPassage


class FakeCorpus:
    def fetch(self, indices):
        return {
            index: CorpusPassage(
                corpus_index=index,
                document_id=f"doc-{index}",
                text=f"text-{index}",
            )
            for index in sorted(set(indices))
        }


def test_load_retrieval_results(tmp_path):
    path = tmp_path / "retrieval.npz"
    np.savez(
        path,
        indices=np.array([[2, 0]], dtype=np.int64),
        scores=np.array([[0.9, 0.5]], dtype=np.float32),
    )

    indices, scores = load_retrieval_results(path)

    assert indices.dtype == np.int64
    assert scores.dtype == np.float32
    assert indices.tolist() == [[2, 0]]


def test_retrieval_results_require_equal_shapes(tmp_path):
    path = tmp_path / "retrieval.npz"
    np.savez(
        path,
        indices=np.array([[0, 1]]),
        scores=np.array([[1.0]]),
    )

    with pytest.raises(
        ValueError,
        match="equal shapes",
    ):
        load_retrieval_results(path)


def test_retrieved_passages_preserve_rank_order():
    passages = retrieved_passages_from_row(
        FakeCorpus(),
        [2, 0],
        [0.9, 0.5],
    )

    assert [
        passage.corpus_index
        for passage in passages
    ] == [2, 0]
    assert [
        passage.retrieval_rank
        for passage in passages
    ] == [1, 2]
    assert passages[0].document_id == "doc-2"


def test_retrieval_row_rejects_duplicates():
    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        retrieved_passages_from_row(
            FakeCorpus(),
            [1, 1],
            [0.9, 0.8],
        )

def test_save_retrieval_results_round_trip(
    tmp_path,
):
    path = tmp_path / "retrieval.npz"

    save_retrieval_results(
        path,
        np.array([[2, 0]], dtype=np.int64),
        np.array([[0.9, 0.5]], dtype=np.float32),
    )
    indices, scores = load_retrieval_results(path)

    assert indices.tolist() == [[2, 0]]
    np.testing.assert_allclose(
        scores,
        np.array(
            [[0.9, 0.5]],
            dtype=np.float32,
        ),
    )
