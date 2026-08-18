"""Tests for the cohort materialization CLI."""

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import pace.cohort.materialize_cohort as cli


def make_args(
    tmp_path: Path,
    dataset: str,
) -> Namespace:
    return Namespace(
        dataset=dataset,
        labels=tmp_path / "labels.json",
        evaluation=(
            tmp_path / "evaluation.json"
            if dataset == "hotpot"
            else None
        ),
        corpus_coverage_report=(
            tmp_path / "corpus_coverage.json"
            if dataset == "2wiki"
            else None
        ),
        retrieval_dir=tmp_path / "retrieval",
        corpus_dir=tmp_path / "corpus",
        output_dir=tmp_path / "output",
        batch_size=100,
        num_parts=8,
        seed=2026,
    )


def test_materialize_hotpot_routes_and_writes_manifest(
    tmp_path,
    monkeypatch,
):
    args = make_args(tmp_path, "hotpot")
    corpus = object()
    calls = {}

    monkeypatch.setattr(
        cli,
        "BergenCorpus",
        SimpleNamespace(
            from_disk=lambda path: corpus
        ),
    )

    def fake_materialize(
        evaluation,
        labels,
        retrieval,
        output,
        loaded_corpus,
        *,
        batch_size,
        seed,
    ):
        calls["values"] = (
            evaluation,
            labels,
            retrieval,
            output,
            loaded_corpus,
            batch_size,
            seed,
        )
        return 5600

    monkeypatch.setattr(
        cli,
        "materialize_hotpot_cohort",
        fake_materialize,
    )

    manifest = cli.materialize_cohort(args)

    assert manifest["complete"] is True
    assert manifest["num_queries"] == 5600
    assert manifest["batch_sizes"] == {
        "reranker": 8,
        "llm": 10,
        "provence": 4,
    }
    assert calls["values"][4] is corpus
    assert (
        args.output_dir / "manifest.json"
    ).is_file()


def test_materialize_2wiki_routes(
    tmp_path,
    monkeypatch,
):
    args = make_args(tmp_path, "2wiki")

    monkeypatch.setattr(
        cli,
        "BergenCorpus",
        SimpleNamespace(
            from_disk=lambda path: object()
        ),
    )
    calls = {}

    def fake_materialize(
        labels,
        retrieval,
        corpus_coverage_report,
        output,
        corpus,
        *,
        num_parts,
    ):
        calls["values"] = (
            labels,
            retrieval,
            corpus_coverage_report,
            output,
            corpus,
            num_parts,
        )
        return 2961

    monkeypatch.setattr(
        cli,
        "materialize_2wiki_cohort",
        fake_materialize,
    )

    manifest = cli.materialize_cohort(args)

    assert manifest["complete"] is True
    assert manifest["num_queries"] == 2961
    assert manifest["num_parts"] == 8
    assert calls["values"][2] == (
        args.corpus_coverage_report
    )
    assert manifest["corpus_coverage_report"] == str(
        args.corpus_coverage_report
    )


def test_hotpot_requires_evaluation(tmp_path):
    args = make_args(tmp_path, "hotpot")
    args.evaluation = None

    with pytest.raises(
        ValueError,
        match="requires --evaluation",
    ):
        cli.materialize_cohort(args)

def test_2wiki_requires_corpus_coverage_report(
    tmp_path,
):
    args = make_args(tmp_path, "2wiki")
    args.corpus_coverage_report = None

    with pytest.raises(
        ValueError,
        match="requires --corpus-coverage-report",
    ):
        cli.materialize_cohort(args)
