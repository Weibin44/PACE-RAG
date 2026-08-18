"""Materialize HotpotQA cohort batches."""

from __future__ import annotations

from pathlib import Path

from pace.cohort.builder import build_cohort_batch
from pace.cohort.gold import load_hotpot_gold_queries
from pace.cohort.io import write_cohort_batch
from pace.cohort.retrieval import (
    CorpusReader,
    load_retrieval_results,
)


def materialize_hotpot_cohort(
    evaluation_path: Path,
    labels_path: Path,
    retrieval_dir: Path,
    output_dir: Path,
    corpus: CorpusReader,
    *,
    batch_size: int = 100,
    seed: int = 2026,
) -> int:
    """Materialize all HotpotQA retrieval batches."""

    if batch_size <= 0:
        raise ValueError(
            "batch size must be positive"
        )

    queries = load_hotpot_gold_queries(
        evaluation_path,
        labels_path,
        seed=seed,
    )

    for start in range(
        0,
        len(queries),
        batch_size,
    ):
        stop = min(
            start + batch_size,
            len(queries),
        )
        batch_name = f"{start:05d}_{stop:05d}"

        indices, scores = load_retrieval_results(
            retrieval_dir
            / batch_name
            / "dense_top100.npz"
        )
        samples = build_cohort_batch(
            queries[start:stop],
            indices,
            scores,
            corpus,
            "hotpot",
        )

        write_cohort_batch(
            output_dir
            / "batches"
            / batch_name
            / "samples_sentence_labels.json",
            samples,
        )

    return len(queries)
