import json

import numpy as np

from pace.data.musique import load_musique_examples


def test_load_musique_examples(tmp_path):
    source = tmp_path / "musique_dev.jsonl"
    cache = tmp_path / "cache"
    similarity_cache = tmp_path / "similarity"

    (cache / "coverage_features").mkdir(parents=True)
    similarity_cache.mkdir(parents=True)

    raw_example = {
        "id": "musique-q1",
        "question": "Who wrote the referenced novel?",
        "paragraphs": [
            {
                "idx": 3,
                "title": "First",
                "paragraph_text": "First paragraph.",
                "is_supporting": True,
            },
            {
                "idx": 8,
                "title": "Second",
                "paragraph_text": "Second paragraph.",
                "is_supporting": True,
            },
        ],
    }
    source.write_text(
        json.dumps(raw_example) + "\n",
        encoding="utf-8",
    )

    np.save(
        cache / "reranker_scores.npy",
        np.array([[0.8, 0.6]], dtype=np.float32),
    )

    np.savez(
        cache / "coverage_features" / "0000.npz",
        query_weights=np.array([1.0, 0.5], dtype=np.float32),
        document_features=np.array(
            [[1.0, 0.2], [0.3, 0.8]],
            dtype=np.float32,
        ),
        dense_scores=np.array([0.9, 0.7], dtype=np.float32),
    )

    np.savez(
        similarity_cache / "0000.npz",
        query_similarity=np.array([0.9, 0.7], dtype=np.float32),
        document_similarity=np.array(
            [[1.0, 0.3], [0.3, 1.0]],
            dtype=np.float32,
        ),
    )

    examples = list(
        load_musique_examples(
            source,
            cache,
            similarity_cache,
            candidate_limit=1,
        )
    )

    assert len(examples) == 1

    example = examples[0]
    assert example.query_id == "0"
    assert example.question == "Who wrote the referenced novel?"
    assert example.gold_fact_ids == frozenset({"3", "8"})

    assert example.candidate_count == 1
    assert example.candidates[0].document_id == "3"
    assert example.candidates[0].text == "First: First paragraph."
    assert example.candidates[0].retriever_score == np.float32(0.9)
    assert example.candidates[0].covered_fact_ids == frozenset({"3"})

    assert example.reranker_scores.shape == (1,)
    assert example.document_features.shape == (1, 2)
    assert example.document_similarity.shape == (1, 1)
    assert example.query_similarity.shape == (1,)