import json

import numpy as np

from pace.data.batched import load_batched_examples


def test_load_batched_examples(tmp_path):
    source = tmp_path / "cohort"
    cache = tmp_path / "cache"
    similarity_cache = tmp_path / "similarity"
    batch = "part00-of-08"

    sample_dir = source / "batches" / batch
    feature_dir = cache / "coverage_features" / batch
    reranker_dir = cache / "reranker_scores"
    similarity_dir = similarity_cache / batch

    for directory in (
        sample_dir,
        feature_dir,
        reranker_dir,
        similarity_dir,
    ):
        directory.mkdir(parents=True)

    sample = {
        "q_id": "q1",
        "question": "example question",
        "facts": [
            {"fact_id": "fact-1"},
            {"fact_id": "fact-2"},
        ],
        "candidates": [
            {
                "doc_index": 10,
                "text": "document one",
                "dense_score": 0.9,
                "covered_facts": ["fact-1"],
            },
            {
                "doc_index": 20,
                "text": "document two",
                "dense_score": 0.7,
                "covered_facts": ["fact-2"],
            },
        ],
    }

    incomplete_sample = {
        "q_id": "q2",
        "question": "incomplete question",
        "facts": [
            {"fact_id": "fact-1"},
            {"fact_id": "fact-2"},
        ],
        "candidates": [
            {
                "doc_index": 30,
                "text": "only partial evidence",
                "dense_score": 0.6,
                "covered_facts": ["fact-1"],
            }
        ],
    }

    (sample_dir / "samples_sentence_labels.json").write_text(
        json.dumps({"samples": [sample, incomplete_sample]})
    )

    np.save(
        reranker_dir / f"{batch}.npy",
        np.array(
            [
                [0.8, 0.6],
                [0.5, 0.0],
            ],
            dtype=np.float32,
        )
    )

    np.savez(
        feature_dir / "000.npz",
        query_weights=np.array([1.0, 0.5, 0.2]),
        document_features=np.array(
            [
                [1.0, 0.0, 0.2],
                [0.0, 1.0, 0.4],
            ]
        ),
    )

    np.savez(
        similarity_dir / "000.npz",
        query_similarity=np.array([0.9, 0.7]),
        document_similarity=np.array(
            [
                [1.0, 0.3],
                [0.3, 1.0],
            ]
        ),
    )

    examples = list(
        load_batched_examples(
            source,
            cache,
            similarity_cache,
            candidate_limit=1,
        )
    )

    assert len(examples) == 1

    example = examples[0]
    assert example.query_id == "q1"
    assert example.candidate_count == 1
    assert example.candidates[0].document_id == "10"
    assert example.gold_fact_ids == frozenset({"fact-1", "fact-2"})
    assert example.reranker_scores.shape == (1,)
    assert example.document_features.shape == (1, 3)
    assert example.document_similarity.shape == (1, 1)
    assert example.query_similarity.shape == (1,)