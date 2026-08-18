"""Slow regression tests against official effectiveness curves."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from pace.data.batched import load_batched_examples
from pace.data.musique import load_musique_examples
from pace.evaluation.calibration import select_calibration_ids
from pace.evaluation.evaluator import (
    evaluate_adaptive_k,
    evaluate_curves,
)
from pace.evaluation.rankings import (
    METHODS,
    RankingParameters,
)


DATA_ROOT_VALUE = os.environ.get("PACE_DATA_ROOT")
REFERENCE_ROOT_VALUE = os.environ.get("PACE_REFERENCE_ROOT")

pytestmark = pytest.mark.skipif(
    DATA_ROOT_VALUE is None or REFERENCE_ROOT_VALUE is None,
    reason=(
        "set PACE_DATA_ROOT and PACE_REFERENCE_ROOT "
        "to run real effectiveness regression"
    ),
)


REFERENCE_METHOD_NAMES = {
    "standard": "standard",
    "coverage_only": "coverage_only",
    "query_only": "sqrt_quality",
    "anchor_only": "anchor_only",
    "ours": "soft_anchor_noisy_or",
    "rocchio_prf": "rocchio_prf",
    "mmr": "mmr",
    "dartboard": "dartboard",
}


def build_dataset(dataset: str, data_root: Path):
    if dataset == "hotpot":
        base = data_root / "hotpotqa" / "top100_complete"

        def factory():
            return load_batched_examples(
                base / "cohort",
                base / "cache",
                base / "cache" / "splade_similarity",
                candidate_limit=100,
            )

        return factory, base / "calibration", 100

    if dataset == "2wiki":
        base = (
            data_root
            / "2wikimultihopqa"
            / "top100_complete"
        )

        def factory():
            return load_batched_examples(
                base / "cohort",
                base / "cache",
                base / "cache" / "splade_similarity",
                candidate_limit=100,
            )

        return (
            factory,
            data_root / "2wikimultihopqa" / "calibration",
            100,
        )

    base = data_root / "musique"

    def factory():
        return load_musique_examples(
            base / "musique_ans_v1.0_dev.jsonl",
            base / "cache",
            base / "cache" / "splade_similarity",
            candidate_limit=20,
        )

    return factory, base / "calibration", 20


def load_parameters(
    calibration_dir: Path,
    stage: str,
) -> RankingParameters:
    manifest = json.loads(
        (
            calibration_dir
            / "literature_baselines_manifest.json"
        ).read_text(encoding="utf-8")
    )
    selected = manifest["selected_parameters"]

    return RankingParameters(
        rocchio_alpha=selected[f"{stage}_rocchio_alpha"],
        rocchio_depth=selected[f"{stage}_rocchio_depth"],
        mmr_diversity=selected[f"{stage}_mmr"],
        dartboard_sigma=selected[f"{stage}_dartboard"],
    )


def load_reference_rows(
    reference_root: Path,
    dataset: str,
) -> dict[tuple[str, str, int], dict[str, str]]:
    path = (
        reference_root
        / "Effectiveness"
        / dataset
        / "run_001"
        / "curves.csv"
    )

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    return {
        (
            row["stage"],
            row["method"],
            int(row["cutoff"]),
        ): row
        for row in rows
        if row["method"] != "fixed_anchor"
    }


@pytest.mark.parametrize(
    "dataset",
    ["hotpot", "2wiki", "musique"],
)
def test_effectiveness_curves_match_official_results(dataset):
    data_root = Path(DATA_ROOT_VALUE)
    reference_root = Path(REFERENCE_ROOT_VALUE)

    factory, calibration_dir, d_maximum = build_dataset(
        dataset,
        data_root,
    )

    calibration_ids = select_calibration_ids(
        (query.query_id for query in factory()),
        100,
    )

    d_result = evaluate_curves(
        factory(),
        "D",
        load_parameters(calibration_dir, "D"),
        cutoffs=range(1, d_maximum + 1),
        excluded_query_ids=calibration_ids,
    )
    k_result = evaluate_curves(
        factory(),
        "K",
        load_parameters(calibration_dir, "K"),
        cutoffs=range(1, 16),
        excluded_query_ids=calibration_ids,
    )

    actual_rows = d_result.rows + k_result.rows
    reference_rows = load_reference_rows(
        reference_root,
        dataset,
    )

    assert len(actual_rows) == len(METHODS) * (
        d_maximum + 15
    )
    assert len(reference_rows) == len(actual_rows)

    for actual in actual_rows:
        reference_method = REFERENCE_METHOD_NAMES[
            actual["method"]
        ]
        key = (
            actual["stage"],
            reference_method,
            actual["cutoff"],
        )
        expected = reference_rows[key]
        # The intended MAD + epsilon formulation differs from the
        # legacy zero-MAD branch for one MuSiQue query. 
        tolerance = (
            5e-4
            if dataset == "musique"
            and actual["method"] == "anchor_only"
            else 1e-12
        )

        assert actual["complete_evidence_recall"] == pytest.approx(
            float(expected["complete_evidence_recall"]),
            abs=tolerance,
        )
        assert actual["supporting_fact_recall"] == pytest.approx(
            float(expected["supporting_recall"]),
            abs=tolerance,
        )


@pytest.mark.parametrize(
    "dataset",
    ["hotpot", "2wiki", "musique"],
)
def test_adaptive_k_matches_official_result(dataset):
    data_root = Path(DATA_ROOT_VALUE)
    reference_root = Path(REFERENCE_ROOT_VALUE)

    factory, _, _ = build_dataset(dataset, data_root)

    calibration_ids = select_calibration_ids(
        (query.query_id for query in factory()),
        100,
    )

    actual = evaluate_adaptive_k(
        factory(),
        excluded_query_ids=calibration_ids,
        buffer=5,
        search_fraction=0.9,
        min_documents=5,
    )

    reference_path = (
        reference_root
        / "Effectiveness"
        / dataset
        / "run_001"
        / "adaptive_points.csv"
    )
    with reference_path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        expected = next(csv.DictReader(handle))

    assert actual["mean_cutoff"] == pytest.approx(
        float(expected["mean_cutoff"]),
        abs=1e-12,
    )
    assert actual["complete_evidence_recall"] == pytest.approx(
        float(expected["complete_evidence_recall"]),
        abs=1e-12,
    )
    assert actual["supporting_fact_recall"] == pytest.approx(
        float(expected["supporting_recall"]),
        abs=1e-12,
    )