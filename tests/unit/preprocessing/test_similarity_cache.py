import json

import numpy as np
import torch

from pace.preprocessing.similarity_cache import (
    write_batched_splade_similarity,
    write_musique_splade_similarity,
)


VECTORS = {
    "question": [1.0, 0.0],
    "document one": [1.0, 0.0],
    "document two": [0.0, 1.0],
    "First: first text": [1.0, 0.0],
    "Second: second text": [0.0, 1.0],
}


def encode(texts):
    return torch.tensor(
        [VECTORS[text] for text in texts],
        dtype=torch.float32,
    )


def test_write_batched_similarity_filters_incomplete(tmp_path):
    source = tmp_path / "cohort"
    batch_dir = source / "batches" / "part00-of-01"
    batch_dir.mkdir(parents=True)

    complete = {
        "q_id": "q1",
        "question": "question",
        "facts": [{"fact_id": "fact-1"}],
        "candidates": [
            {
                "text": "document one",
                "covered_facts": ["fact-1"],
            },
            {
                "text": "document two",
                "covered_facts": [],
            },
        ],
    }
    incomplete = {
        "q_id": "q2",
        "question": "question",
        "facts": [{"fact_id": "missing"}],
        "candidates": [
            {
                "text": "document one",
                "covered_facts": [],
            }
        ],
    }

    (
        batch_dir / "samples_sentence_labels.json"
    ).write_text(
        json.dumps(
            {"samples": [complete, incomplete]}
        ),
        encoding="utf-8",
    )

    cache = tmp_path / "cache"
    assert write_batched_splade_similarity(
        source,
        cache,
        encode,
    ) == 1

    path = (
        cache
        / "splade_similarity"
        / "part00-of-01"
        / "000.npz"
    )
    assert path.is_file()
    assert not path.with_name("001.npz").exists()

    with np.load(path) as values:
        np.testing.assert_array_equal(
            values["query_similarity"],
            np.array([1.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            values["document_similarity"],
            np.eye(2, dtype=np.float16),
        )


def test_write_musique_similarity(tmp_path):
    source = tmp_path / "musique.jsonl"
    source.write_text(
        json.dumps(
            {
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
        )
        + "\n",
        encoding="utf-8",
    )

    cache = tmp_path / "cache"
    assert write_musique_splade_similarity(
        source,
        cache,
        encode,
    ) == 1

    with np.load(
        cache / "splade_similarity" / "0000.npz"
    ) as values:
        assert values["query_similarity"].dtype == np.float32
        assert values["document_similarity"].dtype == np.float16
        np.testing.assert_array_equal(
            values["query_similarity"],
            np.array([1.0, 0.0], dtype=np.float32),
        )