"""Tests for sharded SPLADE Top-K retrieval."""

from types import SimpleNamespace

import pytest
import torch
import numpy as np
from pace.cohort.splade_retrieval import (
    embedding_shard_paths,
    encode_retrieval_queries,
    merge_topk,
    retrieve_topk,
    retrieve_questions,
)

def initial_topk(
    query_count: int,
    candidate_count: int,
):
    return (
        torch.full(
            (query_count, candidate_count),
            -torch.inf,
        ),
        torch.full(
            (query_count, candidate_count),
            -1,
            dtype=torch.long,
        ),
    )


def test_merge_topk_across_shards():
    scores, indices = initial_topk(1, 2)

    scores, indices = merge_topk(
        scores,
        indices,
        torch.tensor([[0.2, 0.8, 0.3]]),
        corpus_offset=0,
        candidate_count=2,
    )
    scores, indices = merge_topk(
        scores,
        indices,
        torch.tensor([[0.9, 0.1]]),
        corpus_offset=3,
        candidate_count=2,
    )

    torch.testing.assert_close(
        scores,
        torch.tensor([[0.9, 0.8]]),
    )
    assert indices.tolist() == [[3, 1]]


def test_embedding_shards_use_numeric_order(
    tmp_path,
):
    (tmp_path / "embedding_chunk_10.pt").touch()
    (tmp_path / "embedding_chunk_2.pt").touch()

    paths = embedding_shard_paths(tmp_path)

    assert [
        path.name
        for path in paths
    ] == [
        "embedding_chunk_2.pt",
        "embedding_chunk_10.pt",
    ]


def test_embedding_shards_must_exist(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match="no embedding shards",
    ):
        embedding_shard_paths(tmp_path)


def test_merge_topk_rejects_negative_offset():
    scores, indices = initial_topk(1, 2)

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        merge_topk(
            scores,
            indices,
            torch.tensor([[1.0]]),
            corpus_offset=-1,
            candidate_count=2,
        )

def test_retrieve_topk_scans_all_shards(
    tmp_path,
):
    torch.save(
        torch.tensor(
            [
                [0.5, 0.0],
                [0.0, 1.0],
            ]
        ),
        tmp_path / "embedding_chunk_0.pt",
    )
    torch.save(
        torch.tensor(
            [
                [0.9, 0.0],
                [0.1, 0.0],
            ]
        ),
        tmp_path / "embedding_chunk_1.pt",
    )

    indices, scores = retrieve_topk(
        torch.tensor([[1.0, 0.0]]),
        tmp_path,
        device="cpu",
        candidate_count=2,
    )

    assert indices.tolist() == [[2, 0]]
    torch.testing.assert_close(
        torch.from_numpy(scores),
        torch.tensor([[0.9, 0.5]]),
    )


def test_retrieve_topk_requires_enough_documents(
    tmp_path,
):
    torch.save(
        torch.tensor([[1.0, 0.0]]),
        tmp_path / "embedding_chunk_0.pt",
    )

    with pytest.raises(
        ValueError,
        match="fewer passages",
    ):
        retrieve_topk(
            torch.tensor([[1.0, 0.0]]),
            tmp_path,
            device="cpu",
            candidate_count=2,
        )


def test_encode_retrieval_queries(
    monkeypatch,
):
    expected = torch.tensor([[1.0, 2.0]])
    calls = {}

    def fake_encode(
        model,
        tokenizer,
        texts,
        device,
        *,
        max_length,
    ):
        calls["values"] = (
            model,
            tokenizer,
            texts,
            device,
            max_length,
        )
        return expected

    monkeypatch.setattr(
        "pace.cohort.splade_retrieval.encode_splade",
        fake_encode,
    )

    bundle = SimpleNamespace(
        model="model",
        tokenizer="tokenizer",
        device=torch.device("cpu"),
    )
    actual = encode_retrieval_queries(
        bundle,
        ["question"],
    )

    assert actual is expected
    assert calls["values"] == (
        "model",
        "tokenizer",
        ["question"],
        "cpu",
        128,
    )

def test_retrieve_questions_runs_complete_pipeline(
    tmp_path,
    monkeypatch,
):
    calls = {}

    class FakeModel:
        def to(self, device):
            calls["model_destination"] = device
            return self

    bundle = SimpleNamespace(
        model=FakeModel(),
        tokenizer="tokenizer",
        device=torch.device("cpu"),
    )
    expected_indices = np.array(
        [[2, 0]],
        dtype=np.int64,
    )
    expected_scores = np.array(
        [[0.9, 0.5]],
        dtype=np.float32,
    )

    monkeypatch.setattr(
        "pace.cohort.splade_retrieval.load_splade_model",
        lambda **kwargs: bundle,
    )
    monkeypatch.setattr(
        "pace.cohort.splade_retrieval."
        "encode_retrieval_queries",
        lambda *args, **kwargs: torch.tensor(
            [[1.0, 0.0]]
        ),
    )
    monkeypatch.setattr(
        "pace.cohort.splade_retrieval.retrieve_topk",
        lambda *args, **kwargs: (
            expected_indices,
            expected_scores,
        ),
    )

    saved = {}

    def fake_save(path, indices, scores):
        saved["values"] = (
            path,
            indices,
            scores,
        )

    monkeypatch.setattr(
        "pace.cohort.splade_retrieval."
        "save_retrieval_results",
        fake_save,
    )

    output = tmp_path / "dense_top100.npz"
    indices, scores = retrieve_questions(
        ["question"],
        tmp_path / "index",
        output,
        device="cpu",
    )

    assert calls["model_destination"] == "cpu"
    assert indices is expected_indices
    assert scores is expected_scores
    assert saved["values"] == (
        output,
        expected_indices,
        expected_scores,
    )

