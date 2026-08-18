"""Optional regression tests using real locally cached models."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from pace.preprocessing.batched_cache import (
    write_batched_coverage_features,
    write_batched_reranker_scores,
)
from pace.preprocessing.modeling import (
    encode_splade,
    load_reranker_model,
    load_splade_model,
    score_reranker_pairs,
)
from pace.preprocessing.similarity_cache import (
    write_batched_splade_similarity,
)


DATA_ROOT_VALUE = os.environ.get("PACE_DATA_ROOT")
DATA_ROOT = Path(
    DATA_ROOT_VALUE or "__missing_data_root__"
)
HOTPOT_ROOT = DATA_ROOT / "hotpotqa/top100_complete"

BATCH_NAME = "00000_00100"
QUERY_INDEX = 8


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("PACE_RUN_REAL_MODEL_TESTS") != "1",
        reason="real-model regression test is disabled",
    ),
    pytest.mark.skipif(
        DATA_ROOT_VALUE is None,
        reason=(
            "set PACE_DATA_ROOT to run real-model "
            "regression tests"
        ),
    ),
]


def make_single_query_source(tmp_path: Path) -> Path:
    source_file = (
        HOTPOT_ROOT
        / "cohort/batches"
        / BATCH_NAME
        / "samples_sentence_labels.json"
    )
    payload = json.loads(source_file.read_text(encoding="utf-8"))
    sample = payload["samples"][QUERY_INDEX]

    output = (
        tmp_path
        / "cohort/batches/single_query"
        / "samples_sentence_labels.json"
    )
    output.parent.mkdir(parents=True)
    output.write_text(
        json.dumps({"samples": [sample]}),
        encoding="utf-8",
    )
    return tmp_path / "cohort"


@pytest.fixture(scope="module")
def device() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FP16 cache comparison")
    return "cuda:0"


def test_real_splade_caches_match_existing(
    tmp_path: Path,
    device: str,
):
    source = make_single_query_source(tmp_path)
    output = tmp_path / "cache"

    bundle = load_splade_model(
        device=device,
        local_files_only=True,
    )

    def encode_128(texts):
        return encode_splade(
            bundle.model,
            bundle.tokenizer,
            texts,
            str(bundle.device),
            max_length=128,
        )

    def encode_256(texts):
        return encode_splade(
            bundle.model,
            bundle.tokenizer,
            texts,
            str(bundle.device),
            max_length=256,
        )

    write_batched_coverage_features(
        source,
        output,
        encode_128,
        batch_name="single_query",
    )
    write_batched_splade_similarity(
        source,
        output,
        encode_256,
        batch_name="single_query",
    )

    generated_coverage = np.load(
        output
        / "coverage_features/single_query/000.npz"
    )
    existing_coverage = np.load(
        HOTPOT_ROOT
        / "cache/coverage_features"
        / BATCH_NAME
        / f"{QUERY_INDEX:03d}.npz"
    )

    np.testing.assert_allclose(
        generated_coverage["query_weights"],
        existing_coverage["query_weights"],
        rtol=1e-3,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        generated_coverage["document_features"],
        existing_coverage["document_features"],
        rtol=1e-3,
        atol=1e-3,
    )

    generated_similarity = np.load(
        output
        / "splade_similarity/single_query/000.npz"
    )
    existing_similarity = np.load(
        HOTPOT_ROOT
        / "cache/splade_similarity"
        / BATCH_NAME
        / f"{QUERY_INDEX:03d}.npz"
    )

    np.testing.assert_allclose(
        generated_similarity["query_similarity"],
        existing_similarity["query_similarity"],
        rtol=1e-3,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        generated_similarity["document_similarity"],
        existing_similarity["document_similarity"],
        rtol=1e-3,
        atol=1e-3,
    )


def test_real_reranker_cache_matches_existing(
    tmp_path: Path,
    device: str,
):
    source = make_single_query_source(tmp_path)
    output = tmp_path / "cache"

    bundle = load_reranker_model(
        device=device,
        local_files_only=True,
    )

    def score(pairs):
        return score_reranker_pairs(
            bundle.model,
            bundle.tokenizer,
            pairs,
            str(bundle.device),
            max_length=256,
        )

    write_batched_reranker_scores(
        source,
        output,
        score,
        batch_name="single_query",
    )

    generated = np.load(
        output / "reranker_scores/single_query.npy"
    )[0]
    existing = np.load(
        HOTPOT_ROOT
        / "cache/reranker_scores"
        / f"{BATCH_NAME}.npy"
    )[QUERY_INDEX]

    np.testing.assert_allclose(
        generated,
        existing,
        rtol=1e-3,
        atol=1e-3,
    )