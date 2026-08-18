import json

import numpy as np
import torch

from pace.preprocessing.batched_cache import (
    write_batched_coverage_features,
    write_batched_reranker_scores,
)


def make_source(tmp_path):
    source = tmp_path / "cohort"
    batch_dir = source / "batches" / "part00-of-01"
    batch_dir.mkdir(parents=True)

    sample = {
        "q_id": "q1",
        "question": "question",
        "facts": [{"fact_id": "fact-1"}],
        "candidates": [
            {
                "doc_index": 1,
                "text": "document one",
                "dense_score": 0.9,
                "covered_facts": ["fact-1"],
            },
            {
                "doc_index": 2,
                "text": "document two",
                "dense_score": 0.8,
                "covered_facts": [],
            },
        ],
    }

    (
        batch_dir / "samples_sentence_labels.json"
    ).write_text(
        json.dumps({"samples": [sample]}),
        encoding="utf-8",
    )
    return source


def test_write_batched_coverage_features(tmp_path):
    source = make_source(tmp_path)
    cache = tmp_path / "cache"

    vectors = {
        "question": [1.0, 0.0, 2.0],
        "document one": [0.5, 4.0, 1.0],
        "document two": [0.2, 3.0, 0.7],
    }

    def encode(texts):
        return torch.tensor(
            [vectors[text] for text in texts],
            dtype=torch.float32,
        )

    count = write_batched_coverage_features(
        source,
        cache,
        encode,
    )

    assert count == 1

    path = (
        cache
        / "coverage_features"
        / "part00-of-01"
        / "000.npz"
    )
    with np.load(path) as values:
        np.testing.assert_array_equal(
            values["query_weights"],
            [1.0, 2.0],
        )
        assert values["query_weights"].dtype == np.float32
        assert values["document_features"].dtype == np.float16
        np.testing.assert_array_equal(
            values["document_features"],
            np.array(
                [
                    [0.5, 1.0],
                    [0.2, 0.7],
                ],
                dtype=np.float16,
            ),
        )

    def must_not_encode(_):
        raise AssertionError("valid cache should be reused")

    write_batched_coverage_features(
        source,
        cache,
        must_not_encode,
    )


def test_write_batched_reranker_scores(tmp_path):
    source = make_source(tmp_path)
    cache = tmp_path / "cache"

    def score(pairs):
        assert len(pairs) == 2
        return np.array([0.8, 0.6], dtype=np.float32)

    count = write_batched_reranker_scores(
        source,
        cache,
        score,
    )

    assert count == 1

    output = (
        cache
        / "reranker_scores"
        / "part00-of-01.npy"
    )
    scores = np.load(output)
    assert scores.dtype == np.float32
    np.testing.assert_array_equal(
        scores,
        np.array(
            [[0.8, 0.6]],
            dtype=np.float32,
        ),
    )

    def must_not_score(_):
        raise AssertionError("valid cache should be reused")

    write_batched_reranker_scores(
        source,
        cache,
        must_not_score,
    )