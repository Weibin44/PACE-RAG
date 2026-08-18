"""Dataset-independent ranking-method dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pace.data.schema import EvidenceExample
from pace.methods.baselines import (
    dartboard_cosine_order,
    dartboard_hybrid_order,
    mmr_order,
    rocchio_prf_order,
)
from pace.methods.pace import (
    greedy_evidence_frontloading_order,
    soft_anchor,
    soft_anchor_relevance_components,
)
from pace.methods.utils import (
    minmax_normalize,
    normalize_nonnegative_by_max,
)


Stage = Literal["D", "K"]

METHODS = (
    "standard",
    "coverage_only",
    "query_only",
    "anchor_only",
    "ours",
    "rocchio_prf",
    "mmr",
    "dartboard",
)


@dataclass(frozen=True)
class RankingParameters:
    """Calibrated parameters for methods that require them."""

    rocchio_alpha: float
    rocchio_depth: int
    mmr_diversity: float
    dartboard_sigma: float

    @property
    def rocchio_beta(self) -> float:
        return 1.0 - self.rocchio_alpha


def _stage_scores(
    example: EvidenceExample,
    stage: Stage,
) -> np.ndarray:
    if stage == "D":
        return np.asarray(
            [
                candidate.retriever_score
                for candidate in example.candidates
            ],
            dtype=np.float32,
        )

    if stage == "K":
        return example.reranker_scores

    raise ValueError(f"unknown ranking stage: {stage}")


def rank_example(
    example: EvidenceExample,
    stage: Stage,
    method: str,
    parameters: RankingParameters,
    *,
    limit: int | None = None,
) -> list[int]:
    """Rank one example using one method at either D or K stage."""

    raw_scores = _stage_scores(example, stage)
    count = example.candidate_count
    limit = count if limit is None else min(limit, count)

    if limit < 0:
        raise ValueError("limit must be non-negative")

    if limit == 0:
        return []

    standard_order = np.argsort(
        -raw_scores,
        kind="stable",
    ).tolist()

    if method == "standard":
        return standard_order[:limit]

    base_relevance = (
        normalize_nonnegative_by_max(raw_scores)
        if stage == "D"
        else minmax_normalize(raw_scores)
    )

    if method == "coverage_only":
        return greedy_evidence_frontloading_order(
            example.query_weights,
            example.document_features,
            np.ones(count, dtype=np.float32),
            limit,
        )

    query_relevance, anchor_relevance, _, _ = (
        soft_anchor_relevance_components(
            base_relevance,
            example.document_similarity,
        )
    )

    if method == "query_only":
        relevance = base_relevance
    elif method == "anchor_only":
        relevance = anchor_relevance
    elif method == "ours":
        relevance, _, _ = soft_anchor(
            base_relevance,
            example.document_similarity,
        )
    elif method == "rocchio_prf":
        return rocchio_prf_order(
            example.query_similarity,
            example.document_similarity,
            standard_order,
            alpha=parameters.rocchio_alpha,
            beta=parameters.rocchio_beta,
            feedback_depth=parameters.rocchio_depth,
        )[:limit]
    elif method == "mmr":
        return mmr_order(
            minmax_normalize(raw_scores),
            example.document_similarity,
            diversity=parameters.mmr_diversity,
            limit=limit,
        )
    elif method == "dartboard":
        if stage == "D":
            return dartboard_cosine_order(
                example.query_similarity,
                example.document_similarity,
                sigma=parameters.dartboard_sigma,
                limit=limit,
            )

        return dartboard_hybrid_order(
            raw_scores,
            example.document_similarity,
            sigma=parameters.dartboard_sigma,
            limit=limit,
        )
    else:
        raise ValueError(f"unknown ranking method: {method}")

    return greedy_evidence_frontloading_order(
        example.query_weights,
        example.document_features,
        relevance,
        limit,
    )