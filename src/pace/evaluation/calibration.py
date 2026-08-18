"""Deterministic calibration split and parameter manifest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from pace.config import BATCH_SIZES
from pace.evaluation.rankings import (
    RankingParameters,
    Stage,
    rank_example,
)
import numpy as np

from pace.data.schema import EvidenceExample
from pace.evaluation.metrics import evidence_scores

# EXPECTED_BATCH_SIZES = {
#     "reranker": 8,
#     "generator": 10,
#     "compressor": 4,
# }


@dataclass(frozen=True)
class CalibrationManifest:
    """Frozen calibration cohort and selected baseline parameters."""

    dataset: str
    calibration_query_ids: frozenset[str]
    parameters_by_stage: Mapping[Stage, RankingParameters]

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "num_calibration_queries": len(
                self.calibration_query_ids
            ),
            "calibration_query_ids": sorted(
                self.calibration_query_ids
            ),
            "selected_parameters": {
                stage: {
                    "rocchio_alpha": parameters.rocchio_alpha,
                    "rocchio_depth": parameters.rocchio_depth,
                    "mmr_diversity": parameters.mmr_diversity,
                    "dartboard_sigma": parameters.dartboard_sigma,
                }
                for stage, parameters
                in self.parameters_by_stage.items()
            },
            "batch_sizes": BATCH_SIZES.manifest_dict(),
        }

    @classmethod
    def from_dict(cls, values: dict) -> "CalibrationManifest":
        batch_sizes = values.get("batch_sizes")
        if batch_sizes != BATCH_SIZES.manifest_dict():
            raise ValueError(
                "manifest batch sizes do not match the fixed "
                "experiment configuration"
            )

        raw_parameters = values["selected_parameters"]
        if set(raw_parameters) != {"D", "K"}:
            raise ValueError(
                "manifest must contain D and K parameters"
            )

        parameters_by_stage = {
            stage: RankingParameters(**raw_parameters[stage])
            for stage in ("D", "K")
        }

        query_ids = frozenset(values["calibration_query_ids"])
        if len(query_ids) != values["num_calibration_queries"]:
            raise ValueError(
                "calibration query count does not match query IDs"
            )

        return cls(
            dataset=values["dataset"],
            calibration_query_ids=query_ids,
            parameters_by_stage=parameters_by_stage,
        )


def select_calibration_ids(
    query_ids: Iterable[str],
    count: int,
) -> frozenset[str]:
    """Select a deterministic query subset using SHA-256 ordering."""

    if count < 0:
        raise ValueError("calibration count must be non-negative")

    unique_ids = set(query_ids)
    if len(unique_ids) < count:
        raise ValueError(
            f"requested {count} calibration queries, "
            f"but only {len(unique_ids)} unique queries are available"
        )

    ordered_ids = sorted(
        unique_ids,
        key=lambda query_id: (
            hashlib.sha256(query_id.encode("utf-8")).digest(),
            query_id,
        ),
    )
    return frozenset(ordered_ids[:count])


def save_calibration_manifest(
    path: Path,
    manifest: CalibrationManifest,
) -> None:
    """Write a calibration manifest as readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )


def load_calibration_manifest(
    path: Path,
) -> CalibrationManifest:
    """Load and validate a calibration manifest."""

    values = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationManifest.from_dict(values)

def _mean_supporting_recall_auc(
    queries: Sequence[EvidenceExample],
    stage: Stage,
    method: str,
    parameters: RankingParameters,
    maximum_cutoff: int,
) -> float:
    """Return mean supporting-fact recall AUC over queries."""

    query_scores = []

    for query in queries:
        order = rank_example(
            query,
            stage,
            method,
            parameters,
            limit=maximum_cutoff,
        )

        recalls = [
            evidence_scores(
                query,
                order[:cutoff],
            )["supporting_fact_recall"]
            for cutoff in range(1, maximum_cutoff + 1)
        ]
        query_scores.append(float(np.mean(recalls)))

    return float(np.mean(query_scores))


def calibrate_stage(
    examples: Iterable[EvidenceExample],
    calibration_query_ids: frozenset[str],
    stage: Stage,
    maximum_cutoff: int,
    *,
    parameter_step: float = 0.05,
    rocchio_alphas: Sequence[float] = (
        0.1,
        0.3,
        0.5,
        0.7,
        0.9,
    ),
    rocchio_depths: Sequence[int] = (1, 3, 5, 10),
) -> RankingParameters:
    """Select PRF, MMR, and Dartboard parameters for one stage."""

    if maximum_cutoff <= 0:
        raise ValueError("maximum cutoff must be positive")

    if not 0.0 < parameter_step <= 1.0:
        raise ValueError("parameter step must be in (0, 1]")

    queries = [
        query
        for query in examples
        if query.query_id in calibration_query_ids
    ]

    found_ids = {query.query_id for query in queries}
    missing_ids = calibration_query_ids - found_ids
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} calibration queries were not loaded"
        )

    if not queries:
        raise ValueError("no calibration queries were selected")

    placeholder = RankingParameters(
        rocchio_alpha=0.0,
        rocchio_depth=1,
        mmr_diversity=0.0,
        dartboard_sigma=parameter_step,
    )

    rocchio_candidates = [
        (float(alpha), int(depth))
        for alpha in rocchio_alphas
        for depth in rocchio_depths
    ]

    rocchio_scores = {}
    for alpha, depth in rocchio_candidates:
        parameters = RankingParameters(
            rocchio_alpha=alpha,
            rocchio_depth=depth,
            mmr_diversity=placeholder.mmr_diversity,
            dartboard_sigma=placeholder.dartboard_sigma,
        )
        rocchio_scores[(alpha, depth)] = (
            _mean_supporting_recall_auc(
                queries,
                stage,
                "rocchio_prf",
                parameters,
                maximum_cutoff,
            )
        )

    best_alpha, best_depth = max(
        rocchio_candidates,
        key=lambda candidate: (
            rocchio_scores[candidate],
            -candidate[0],
            -candidate[1],
        ),
    )

    mmr_grid = [
        float(value)
        for value in np.arange(
            0.0,
            1.0 + parameter_step / 2.0,
            parameter_step,
        )
    ]
    mmr_scores = {}

    for diversity in mmr_grid:
        parameters = RankingParameters(
            rocchio_alpha=best_alpha,
            rocchio_depth=best_depth,
            mmr_diversity=diversity,
            dartboard_sigma=placeholder.dartboard_sigma,
        )
        mmr_scores[diversity] = _mean_supporting_recall_auc(
            queries,
            stage,
            "mmr",
            parameters,
            maximum_cutoff,
        )

    best_diversity = max(
        mmr_grid,
        key=lambda value: (
            mmr_scores[value],
            -value,
        ),
    )

    dartboard_grid = [
        float(value)
        for value in np.arange(
            parameter_step,
            1.0 + parameter_step / 2.0,
            parameter_step,
        )
    ]
    dartboard_scores = {}

    for sigma in dartboard_grid:
        parameters = RankingParameters(
            rocchio_alpha=best_alpha,
            rocchio_depth=best_depth,
            mmr_diversity=best_diversity,
            dartboard_sigma=sigma,
        )
        dartboard_scores[sigma] = (
            _mean_supporting_recall_auc(
                queries,
                stage,
                "dartboard",
                parameters,
                maximum_cutoff,
            )
        )

    best_sigma = max(
        dartboard_grid,
        key=lambda value: (
            dartboard_scores[value],
            -value,
        ),
    )

    return RankingParameters(
        rocchio_alpha=best_alpha,
        rocchio_depth=best_depth,
        mmr_diversity=best_diversity,
        dartboard_sigma=best_sigma,
    )



