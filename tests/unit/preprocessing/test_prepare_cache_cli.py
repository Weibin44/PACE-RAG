from pathlib import Path

from pace.preprocessing.prepare_cache import (
    build_parser,
    cache_manifest_path,
    stage_max_length,
)

import json
from types import SimpleNamespace

import pytest
import torch

import pace.preprocessing.prepare_cache as cache_cli
from pace.preprocessing.prepare_cache import (
    build_parser,
    cache_manifest_path,
    prepare_cache,
    stage_max_length,
)


def test_prepare_cache_parser_defaults():
    args = build_parser().parse_args(
        [
            "--dataset",
            "hotpot",
            "--stage",
            "coverage_features",
            "--source",
            "cohort",
            "--cache-dir",
            "cache",
        ]
    )

    assert args.device == "auto"
    assert args.local_files_only is False
    assert args.batch_name is None


def test_stage_max_lengths():
    assert stage_max_length(
        "hotpot",
        "coverage_features",
    ) == 128
    assert stage_max_length(
        "2wiki",
        "coverage_features",
    ) == 128
    assert stage_max_length(
        "musique",
        "coverage_features",
    ) == 256
    assert stage_max_length(
        "hotpot",
        "splade_similarity",
    ) == 256
    assert stage_max_length(
        "musique",
        "reranker_scores",
    ) == 256


def test_cache_manifest_path():
    path = cache_manifest_path(
        Path("cache"),
        "2wiki",
        "splade_similarity",
        "part00-of-08",
    )

    assert path == Path(
        "cache/manifests/"
        "2wiki_splade_similarity_part00-of-08.json"
    )

WRITER_NAMES = (
    "write_batched_coverage_features",
    "write_batched_reranker_scores",
    "write_batched_splade_similarity",
    "write_musique_coverage_features",
    "write_musique_reranker_scores",
    "write_musique_splade_similarity",
)


@pytest.mark.parametrize(
    (
        "dataset",
        "stage",
        "expected_writer",
        "expected_loader",
        "expected_max_length",
    ),
    [
        (
            "hotpot",
            "coverage_features",
            "write_batched_coverage_features",
            "splade",
            128,
        ),
        (
            "hotpot",
            "reranker_scores",
            "write_batched_reranker_scores",
            "reranker",
            256,
        ),
        (
            "hotpot",
            "splade_similarity",
            "write_batched_splade_similarity",
            "splade",
            256,
        ),
        (
            "musique",
            "coverage_features",
            "write_musique_coverage_features",
            "splade",
            256,
        ),
        (
            "musique",
            "reranker_scores",
            "write_musique_reranker_scores",
            "reranker",
            256,
        ),
        (
            "musique",
            "splade_similarity",
            "write_musique_splade_similarity",
            "splade",
            256,
        ),
    ],
)
def test_prepare_cache_routes_without_loading_models(
    tmp_path,
    monkeypatch,
    dataset,
    stage,
    expected_writer,
    expected_loader,
    expected_max_length,
):
    calls = []

    bundle = SimpleNamespace(
        tokenizer=object(),
        model=object(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    def load_splade(**kwargs):
        calls.append("splade")
        assert kwargs["local_files_only"] is False
        return bundle

    def load_reranker(**kwargs):
        calls.append("reranker")
        assert kwargs["local_files_only"] is False
        return bundle

    monkeypatch.setattr(
        cache_cli,
        "load_splade_model",
        load_splade,
    )
    monkeypatch.setattr(
        cache_cli,
        "load_reranker_model",
        load_reranker,
    )

    for writer_name in WRITER_NAMES:
        def writer(
            *args,
            _name=writer_name,
            **kwargs,
        ):
            calls.append(_name)
            return 7

        monkeypatch.setattr(
            cache_cli,
            writer_name,
            writer,
        )

    cache_dir = tmp_path / "cache"
    args = build_parser().parse_args(
        [
            "--dataset",
            dataset,
            "--stage",
            stage,
            "--source",
            str(tmp_path / "source"),
            "--cache-dir",
            str(cache_dir),
            "--device",
            "cpu",
        ]
    )

    manifest = prepare_cache(args)

    assert calls == [expected_loader, expected_writer]
    assert manifest["complete"] is True
    assert manifest["num_queries"] == 7
    assert manifest["max_length"] == expected_max_length
    assert manifest["resolved_device"] == "cpu"
    assert manifest["dtype"] == "float32"
    assert manifest["batch_sizes"] == {
        "reranker": 8,
        "llm": 10,
        "provence": 4,
        "splade_encoder": 8,
    }

    path = cache_manifest_path(
        cache_dir,
        dataset,
        stage,
        None,
    )
    saved = json.loads(
        path.read_text(encoding="utf-8")
    )
    assert saved == manifest


def test_failed_cache_generation_keeps_incomplete_manifest(
    tmp_path,
    monkeypatch,
):
    bundle = SimpleNamespace(
        tokenizer=object(),
        model=object(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    monkeypatch.setattr(
        cache_cli,
        "load_splade_model",
        lambda **kwargs: bundle,
    )

    def fail_writer(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(
        cache_cli,
        "write_batched_coverage_features",
        fail_writer,
    )

    cache_dir = tmp_path / "cache"
    args = build_parser().parse_args(
        [
            "--dataset",
            "hotpot",
            "--stage",
            "coverage_features",
            "--source",
            str(tmp_path / "source"),
            "--cache-dir",
            str(cache_dir),
            "--device",
            "cpu",
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="simulated failure",
    ):
        prepare_cache(args)

    path = cache_manifest_path(
        cache_dir,
        "hotpot",
        "coverage_features",
        None,
    )
    manifest = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert manifest["complete"] is False
    assert "num_queries" not in manifest