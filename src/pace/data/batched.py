"""Loader for batched Top-100-complete HotpotQA and 2Wiki cohorts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .schema import Candidate, EvidenceExample


def _candidate_id(candidate: dict) -> str:
    for key in ("doc_id", "corpus_index", "doc_index"):
        if key in candidate:
            return str(candidate[key])
    raise KeyError("candidate has no document identifier")


def load_batched_examples(
    source: Path,
    cache_dir: Path,
    similarity_cache: Path,
    *,
    candidate_limit: int | None = None,
) -> Iterator[EvidenceExample]:
    """Load HotpotQA or 2Wiki examples and their cached features."""

    sample_paths = sorted(
        (source / "batches").glob("*/samples_sentence_labels.json")
    )
    if not sample_paths:
        raise FileNotFoundError(f"no cohort batches found under {source}")

    for sample_path in sample_paths:
        batch_name = sample_path.parent.name
        samples = json.loads(sample_path.read_text())["samples"]

        reranker_path = (
            cache_dir / "reranker_scores" / f"{batch_name}.npy"
        )
        reranker_scores = np.load(reranker_path)

        if len(reranker_scores) != len(samples):
            raise ValueError(
                f"{batch_name}: reranker rows do not match samples"
            )

        for index, sample in enumerate(samples):
            raw_candidates = sample["candidates"]

            gold_fact_ids = frozenset(
                fact["fact_id"] for fact in sample["facts"]
            )
            available_fact_ids = {
                fact_id
                for candidate in raw_candidates
                for fact_id in candidate.get("covered_facts", [])
            }

            # Keep only queries whose complete gold evidence occurs
            # in the retrieved candidate pool.
            if not gold_fact_ids <= available_fact_ids:
                continue
            count = (
                len(raw_candidates)
                if candidate_limit is None
                else min(candidate_limit, len(raw_candidates))
            )
            raw_candidates = raw_candidates[:count]

            candidates = tuple(
                Candidate(
                    document_id=_candidate_id(candidate),
                    text=candidate["text"],
                    retriever_score=float(
                        candidate.get(
                            "dense_score",
                            candidate.get("score", 0.0),
                        )
                    ),
                    covered_fact_ids=frozenset(
                        candidate.get("covered_facts", [])
                    ),
                )
                for candidate in raw_candidates
            )

            feature_path = (
                cache_dir
                / "coverage_features"
                / batch_name
                / f"{index:03d}.npz"
            )
            with np.load(feature_path) as values:
                query_weights = values["query_weights"].astype(np.float32)
                document_features = values[
                    "document_features"
                ][:count].astype(np.float32)

            similarity_path = (
                similarity_cache
                / batch_name
                / f"{index:03d}.npz"
            )
            with np.load(similarity_path) as values:
                query_similarity = values[
                    "query_similarity"
                ][:count].astype(np.float32)
                document_similarity = values[
                    "document_similarity"
                ][:count, :count].astype(np.float32)

            example = EvidenceExample(
                query_id=str(sample["q_id"]),
                question=sample["question"],
                # gold_fact_ids=frozenset(
                #     fact["fact_id"] for fact in sample["facts"]
                # ),
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