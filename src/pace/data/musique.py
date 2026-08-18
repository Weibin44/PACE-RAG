"""Loader for the MuSiQue closed-context development set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .schema import Candidate, EvidenceExample


def load_musique_examples(
    source: Path,
    cache_dir: Path,
    similarity_cache: Path,
    *,
    candidate_limit: int | None = None,
) -> Iterator[EvidenceExample]:
    """Load MuSiQue examples and their cached ranking features."""

    if not source.is_file():
        raise FileNotFoundError(f"MuSiQue source not found: {source}")

    examples = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    reranker_scores = np.load(cache_dir / "reranker_scores.npy")
    if reranker_scores.shape[0] != len(examples):
        raise ValueError(
            "reranker score rows do not match MuSiQue examples"
        )

    for index, raw_example in enumerate(examples):
        paragraphs = raw_example["paragraphs"]
        count = (
            len(paragraphs)
            if candidate_limit is None
            else min(candidate_limit, len(paragraphs))
        )

        gold_fact_ids = frozenset(
            str(paragraph["idx"])
            for paragraph in paragraphs
            if paragraph["is_supporting"]
        )

        feature_path = (
            cache_dir / "coverage_features" / f"{index:04d}.npz"
        )
        with np.load(feature_path) as values:
            query_weights = values["query_weights"].astype(np.float32)
            document_features = values[
                "document_features"
            ][:count].astype(np.float32)
            dense_scores = values[
                "dense_scores"
            ][:count].astype(np.float32)

        similarity_path = similarity_cache / f"{index:04d}.npz"
        with np.load(similarity_path) as values:
            query_similarity = values[
                "query_similarity"
            ][:count].astype(np.float32)
            document_similarity = values[
                "document_similarity"
            ][:count, :count].astype(np.float32)

        candidates = tuple(
            Candidate(
                document_id=str(paragraph["idx"]),
                text=(
                    f'{paragraph["title"]}: '
                    f'{paragraph["paragraph_text"]}'
                ),
                retriever_score=float(dense_scores[position]),
                covered_fact_ids=(
                    frozenset({str(paragraph["idx"])})
                    if paragraph["is_supporting"]
                    else frozenset()
                ),
            )
            for position, paragraph in enumerate(paragraphs[:count])
        )

        example = EvidenceExample(
            # Preserve the position-based query IDs used by the original
            # effectiveness experiment and its calibration split.
            query_id=str(index),
            question=raw_example["question"],
            gold_fact_ids=gold_fact_ids,
            candidates=candidates,
            reranker_scores=reranker_scores[
                index, :count
            ].astype(np.float32),
            query_weights=query_weights,
            document_features=document_features,
            document_similarity=document_similarity,
            query_similarity=query_similarity,
        )
        example.validate()
        yield example