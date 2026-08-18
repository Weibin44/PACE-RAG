from pace.evaluation.calibration import (
    CalibrationManifest,
    load_calibration_manifest,
    save_calibration_manifest,
    select_calibration_ids,
)
from pace.evaluation.rankings import RankingParameters

import numpy as np
import pytest

from pace.data.schema import Candidate, EvidenceExample
from pace.evaluation.calibration import calibrate_stage

D_PARAMETERS = RankingParameters(
    rocchio_alpha=0.3,
    rocchio_depth=3,
    mmr_diversity=0.2,
    dartboard_sigma=0.4,
)

K_PARAMETERS = RankingParameters(
    rocchio_alpha=0.5,
    rocchio_depth=5,
    mmr_diversity=0.1,
    dartboard_sigma=0.3,
)

def make_calibration_query(query_id: str) -> EvidenceExample:
    candidates = (
        Candidate(
            document_id="doc-1",
            text="First",
            retriever_score=0.9,
            covered_fact_ids=frozenset(),
        ),
        Candidate(
            document_id="doc-2",
            text="Second",
            retriever_score=0.5,
            covered_fact_ids=frozenset(),
        ),
    )

    return EvidenceExample(
        query_id=query_id,
        question="Question",
        gold_fact_ids=frozenset(),
        candidates=candidates,
        reranker_scores=np.array([0.8, 0.4], dtype=np.float32),
        query_weights=np.array([1.0], dtype=np.float32),
        document_features=np.ones((2, 1), dtype=np.float32),
        document_similarity=np.eye(2, dtype=np.float32),
        query_similarity=np.array([0.8, 0.4], dtype=np.float32),
    )


def test_calibrate_stage_uses_stable_tie_breaking():
    parameters = calibrate_stage(
        [make_calibration_query("q1")],
        frozenset({"q1"}),
        "D",
        maximum_cutoff=2,
        parameter_step=0.5,
        rocchio_alphas=(0.1, 0.3),
        rocchio_depths=(1, 3),
    )

    assert parameters == RankingParameters(
        rocchio_alpha=0.1,
        rocchio_depth=1,
        mmr_diversity=0.0,
        dartboard_sigma=0.5,
    )


def test_calibrate_stage_rejects_missing_query():
    with pytest.raises(
        ValueError,
        match="calibration queries were not loaded",
    ):
        calibrate_stage(
            [make_calibration_query("q1")],
            frozenset({"missing"}),
            "D",
            maximum_cutoff=2,
            parameter_step=0.5,
        )

def test_calibration_selection_is_deterministic():
    forward = select_calibration_ids(
        ["q1", "q2", "q3", "q4"],
        2,
    )
    backward = select_calibration_ids(
        ["q4", "q3", "q2", "q1"],
        2,
    )

    assert forward == backward
    assert len(forward) == 2


def test_calibration_manifest_round_trip(tmp_path):
    manifest = CalibrationManifest(
        dataset="hotpot",
        calibration_query_ids=frozenset({"q1", "q2"}),
        parameters_by_stage={
            "D": D_PARAMETERS,
            "K": K_PARAMETERS,
        },
    )

    path = tmp_path / "calibration_manifest.json"
    save_calibration_manifest(path, manifest)
    loaded = load_calibration_manifest(path)

    assert loaded == manifest
    assert loaded.parameters_by_stage["D"] == D_PARAMETERS
    assert loaded.parameters_by_stage["K"] == K_PARAMETERS