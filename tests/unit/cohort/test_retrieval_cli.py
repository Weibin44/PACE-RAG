"""Tests for the dataset-aware retrieval CLI."""

import json

import pace.cohort.run_retrieval as cli


def test_run_hotpot_batch(
    tmp_path,
    monkeypatch,
):
    evaluation = tmp_path / "evaluation.json"
    labels = tmp_path / "labels.json"

    evaluation.write_text(
        json.dumps(
            [
                {
                    "q_id": f"q{index}",
                    "question": f"question-{index}",
                }
                for index in range(3)
            ]
        ),
        encoding="utf-8",
    )
    labels.write_text(
        json.dumps(
            [
                {
                    "_id": f"q{index}",
                    "context": [["A", ["fact"]]],
                    "supporting_facts": [["A", 0]],
                }
                for index in range(3)
            ]
        ),
        encoding="utf-8",
    )

    calls = {}

    def fake_retrieve(
        questions,
        index_dir,
        output_path,
        **kwargs,
    ):
        calls["values"] = (
            questions,
            index_dir,
            output_path,
            kwargs,
        )

    monkeypatch.setattr(
        cli,
        "retrieve_questions",
        fake_retrieve,
    )

    args = cli.build_parser().parse_args(
        [
            "hotpot",
            "--evaluation",
            str(evaluation),
            "--labels",
            str(labels),
            "--index-dir",
            str(tmp_path / "index"),
            "--output-dir",
            str(tmp_path / "output"),
            "--batch-index",
            "1",
            "--batch-size",
            "2",
        ]
    )

    output, count = cli.run_retrieval(args)

    assert count == 1
    assert output.name == "dense_top100.npz"
    assert output.parent.name == "00002_00003"
    assert len(calls["values"][0]) == 1


def test_run_2wiki_round_robin_part(
    tmp_path,
    monkeypatch,
):
    labels = tmp_path / "dev.json"
    report = tmp_path / "report.json"

    labels.write_text(
        json.dumps(
            [
                {
                    "_id": f"q{index}",
                    "question": f"question-{index}",
                }
                for index in range(5)
            ]
        ),
        encoding="utf-8",
    )
    report.write_text(
        json.dumps(
            {
                "complete_query_ids": [
                    f"q{index}"
                    for index in range(5)
                ]
            }
        ),
        encoding="utf-8",
    )

    calls = {}

    monkeypatch.setattr(
        cli,
        "retrieve_questions",
        lambda questions, *args, **kwargs: (
            calls.setdefault(
                "questions",
                list(questions),
            )
        ),
    )

    args = cli.build_parser().parse_args(
        [
            "2wiki",
            "--labels",
            str(labels),
            "--coverage-report",
            str(report),
            "--index-dir",
            str(tmp_path / "index"),
            "--output-dir",
            str(tmp_path / "output"),
            "--part-index",
            "1",
            "--num-parts",
            "2",
        ]
    )

    output, count = cli.run_retrieval(args)

    assert count == 2
    assert calls["questions"] == [
        "question-1",
        "question-3",
    ]
    assert output.name == (
        "dense_top100.part01-of-02.npz"
    )
