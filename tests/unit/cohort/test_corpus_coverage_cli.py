"""Tests for the 2Wiki corpus-coverage CLI."""

import json
from types import SimpleNamespace

import pace.cohort.run_corpus_coverage as cli


class FakeCorpus:
    def iter_contents(self, *, batch_size):
        assert batch_size == 65536
        yield "A: supporting sentence."


def test_run_corpus_coverage_audit(
    tmp_path,
    monkeypatch,
):
    labels = tmp_path / "dev.json"
    output = tmp_path / "report.json"

    labels.write_text(
        json.dumps(
            [
                {
                    "_id": "q1",
                    "context": [
                        ["A", ["supporting sentence"]]
                    ],
                    "supporting_facts": [["A", 0]],
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        cli,
        "BergenCorpus",
        SimpleNamespace(
            from_disk=lambda path: FakeCorpus()
        ),
    )

    args = cli.build_parser().parse_args(
        [
            "--labels",
            str(labels),
            "--corpus-dir",
            str(tmp_path / "corpus"),
            "--output",
            str(output),
        ]
    )
    report = cli.run_audit(args)

    assert report["complete"] is True
    assert report["queries_with_all_facts"] == 1
    assert report["complete_query_ids"] == ["q1"]
    assert report["batch_sizes"] == {
        "reranker": 8,
        "llm": 10,
        "provence": 4,
    }
    assert json.loads(
        output.read_text(encoding="utf-8")
    ) == report
