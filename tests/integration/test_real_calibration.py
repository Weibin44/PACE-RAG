"""Slow regression tests for baseline calibration parameters."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pace.data.batched import load_batched_examples
from pace.data.musique import load_musique_examples
from pace.evaluation.calibration import (
    calibrate_stage,
    select_calibration_ids,
)
from pace.evaluation.rankings import RankingParameters


DATA_ROOT_VALUE = os.environ.get("PACE_DATA_ROOT")

pytestmark = pytest.mark.skipif(
    DATA_ROOT_VALUE is None,
    reason="set PACE_DATA_ROOT to run real calibration tests",
)


def build_factory(dataset: str, data_root: Path):
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


def parameters_from_legacy(
    selected: dict,
    stage: str,
) -> RankingParameters:
    return RankingParameters(
        rocchio_alpha=selected[f"{stage}_rocchio_alpha"],
        rocchio_depth=selected[f"{stage}_rocchio_depth"],
        mmr_diversity=selected[f"{stage}_mmr"],
        dartboard_sigma=selected[f"{stage}_dartboard"],
    )


@pytest.mark.parametrize(
    "dataset",
    ["hotpot", "2wiki", "musique"],
)
def test_calibration_matches_legacy_manifest(dataset):
    data_root = Path(DATA_ROOT_VALUE)
    factory, calibration_dir, d_maximum = build_factory(
        dataset,
        data_root,
    )

    legacy_path = (
        calibration_dir / "literature_baselines_manifest.json"
    )
    legacy_manifest = json.loads(
        legacy_path.read_text(encoding="utf-8")
    )
    selected = legacy_manifest["selected_parameters"]

    calibration_ids = select_calibration_ids(
        (query.query_id for query in factory()),
        100,
    )

    # Keep only the 100 calibration queries in memory so the two
    # parameter searches do not repeatedly load the full dataset.
    calibration_queries = [
        query
        for query in factory()
        if query.query_id in calibration_ids
    ]
    assert len(calibration_queries) == 100

    actual_d = calibrate_stage(
        calibration_queries,
        calibration_ids,
        "D",
        d_maximum,
        parameter_step=0.05,
    )
    actual_k = calibrate_stage(
        calibration_queries,
        calibration_ids,
        "K",
        15,
        parameter_step=0.05,
    )

    assert actual_d == parameters_from_legacy(selected, "D")
    assert actual_k == parameters_from_legacy(selected, "K")