import json

import numpy as np
import torch

from pace.preprocessing.musique_cache import (
    write_musique_coverage_features,
    write_musique_reranker_scores,
)


def make_source(tmp_path):
    source = tmp_path / "musique.jsonl"

    example = {
        "id": "q1",
        "question": "question",
        "paragraphs": [
            {
                "idx": 0,
                "title": "First",
                "paragraph_text": "first text",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "Second",
                "paragraph_text": "second text",
                "is_supporting": False,
            },
        ],
    }

    source.write_text(
        json.dumps(example) + "\n",
        encoding="utf-8",
    )
    return source


def test_write_musique_coverage_features(tmp_path):
    source = make_source(tmp_path)
    cache = tmp_path / "cache"

    vectors = {
        "question": [1.0, 0.0, 2.0],
        "First: first text": [0.5, 4.0, 1.0],
        "Second: second text": [0.2, 3.0, 0.7],
    }

    def encode(texts):
        return torch.tensor(
            [vectors[text] for text in texts],
            dtype=torch.float32,
        )

    assert write_musique_coverage_features(
        source,
        cache,
        encode,
    ) == 1

    with np.load(
        cache / "coverage_features" / "0000.npz"
    ) as values:
        assert values["query_weights"].dtype == np.float32
        assert values["document_features"].dtype == np.float16
        assert values["dense_scores"].dtype == np.float32

        np.testing.assert_array_equal(
            values["query_weights"],
            np.array([1.0, 2.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            values["dense_scores"],
            np.array([2.5, 1.6], dtype=np.float32),
        )


def test_write_musique_reranker_scores(tmp_path):
    source = make_source(tmp_path)
    cache = tmp_path / "cache"

    def score(pairs):
        assert len(pairs) == 2
        return np.array([0.8, 0.6], dtype=np.float32)

    assert write_musique_reranker_scores(
        source,
        cache,
        score,
    ) == 1

    values = np.load(cache / "reranker_scores.npy")

    assert values.shape == (1, 20)
    assert values.dtype == np.float32
    np.testing.assert_array_equal(
        values[0, :2],
        np.array([0.8, 0.6], dtype=np.float32),
    )
    assert np.isneginf(values[0, 2:]).all()