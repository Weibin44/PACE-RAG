"""Effectiveness evaluation over ranking cutoffs."""

from __future__ import annotations
import numpy as np
from pace.methods.baselines import adaptive_k_cutoff
import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pace.data.schema import EvidenceExample
from pace.evaluation.metrics import evidence_scores
from pace.evaluation.rankings import (
    METHODS,
    RankingParameters,
    Stage,
    rank_example,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated effectiveness curves and evaluated query count."""

    rows: tuple[dict[str, str | int | float], ...]
    query_count: int


def evaluate_curves(
    examples: Iterable[EvidenceExample],
    stage: Stage,
    parameters: RankingParameters,
    *,
    cutoffs: Sequence[int],
    methods: Sequence[str] = METHODS,
    excluded_query_ids: frozenset[str] = frozenset(),
) -> EvaluationResult:
    """Evaluate multiple ranking methods at the requested cutoffs."""

    normalized_cutoffs = tuple(sorted(set(cutoffs)))
    if not normalized_cutoffs:
        raise ValueError("at least one cutoff is required")
    if normalized_cutoffs[0] <= 0:
        raise ValueError("cutoffs must be positive")

    metric_names = (
        "complete_evidence_recall",
        "supporting_fact_recall",
        "precision",
    )

    totals = {
        (method, cutoff, metric): 0.0
        for method in methods
        for cutoff in normalized_cutoffs
        for metric in metric_names
    }

    query_count = 0
    maximum_cutoff = normalized_cutoffs[-1]

    for query in examples:
        if query.query_id in excluded_query_ids:
            continue

        query_count += 1

        for method in methods:
            order = rank_example(
                query,
                stage,
                method,
                parameters,
                limit=maximum_cutoff,
            )

            for cutoff in normalized_cutoffs:
                scores = evidence_scores(query, order[:cutoff])
                for metric in metric_names:
                    totals[(method, cutoff, metric)] += scores[metric]

    if query_count == 0:
        raise ValueError("no evaluation queries remain")

    rows = tuple(
        {
            "stage": stage,
            "method": method,
            "cutoff": cutoff,
            **{
                metric: totals[(method, cutoff, metric)] / query_count
                for metric in metric_names
            },
        }
        for method in methods
        for cutoff in normalized_cutoffs
    )

    return EvaluationResult(rows=rows, query_count=query_count)


def write_curves_csv(
    path: Path,
    result: EvaluationResult,
) -> None:
    """Write aggregated effectiveness curves to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        "stage",
        "method",
        "cutoff",
        "complete_evidence_recall",
        "supporting_fact_recall",
        "precision",
    )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.rows)


def evaluate_adaptive_k(
    examples: Iterable[EvidenceExample],
    *,
    excluded_query_ids: frozenset[str] = frozenset(),
    buffer: int = 5,
    search_fraction: float = 0.9,
    min_documents: int = 5,
) -> dict[str, str | float | int]:
    """Evaluate the Adaptive-K D-stage baseline."""

    metric_names = (
        "complete_evidence_recall",
        "supporting_fact_recall",
        "precision",
    )
    metric_totals = {metric: 0.0 for metric in metric_names}
    cutoff_total = 0
    query_count = 0

    for query in examples:
        if query.query_id in excluded_query_ids:
            continue

        retriever_scores = np.asarray(
            [
                candidate.retriever_score
                for candidate in query.candidates
            ],
            dtype=np.float32,
        )

        cutoff = adaptive_k_cutoff(
            retriever_scores,
            buffer=buffer,
            search_fraction=search_fraction,
            min_documents=min_documents,
        )

        # Candidates are stored in descending retriever order.
        order = list(range(cutoff))
        scores = evidence_scores(query, order)

        cutoff_total += cutoff
        query_count += 1
        for metric in metric_names:
            metric_totals[metric] += scores[metric]

    if query_count == 0:
        raise ValueError("no evaluation queries remain")

    return {
        "stage": "D",
        "method": "adaptive_k",
        "mean_cutoff": cutoff_total / query_count,
        **{
            metric: metric_totals[metric] / query_count
            for metric in metric_names
        },
        "query_count": query_count,
    }


def write_adaptive_point_csv(
    path: Path,
    row: dict[str, str | float | int],
) -> None:
    """Write the Adaptive-K result point."""

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        "stage",
        "method",
        "mean_cutoff",
        "complete_evidence_recall",
        "supporting_fact_recall",
        "precision",
        "query_count",
    )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)