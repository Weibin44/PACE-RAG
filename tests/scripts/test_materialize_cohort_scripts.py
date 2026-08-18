"""Smoke tests for cohort materialization scripts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("script_name", "dataset"),
    [
        ("materialize_hotpot_cohort.sh", "hotpot"),
        ("materialize_2wiki_cohort.sh", "2wiki"),
    ],
)
def test_materialize_cohort_script(
    tmp_path: Path,
    script_name: str,
    dataset: str,
):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()

    log_path = tmp_path / "arguments.txt"
    fake_command = (
        binary_dir / "pace-materialize-cohort"
    )
    fake_command.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$@" > "${PACE_SMOKE_LOG}"\n',
        encoding="utf-8",
    )
    fake_command.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": (
                f"{binary_dir}:"
                f"{environment.get('PATH', '')}"
            ),
            "PACE_DATA_ROOT": str(tmp_path / "data"),
            "PACE_COHORT_OUTPUT_ROOT": str(
                tmp_path / "outputs"
            ),
            "BERGEN_CORPUS_DIR": str(
                tmp_path / "corpus"
            ),
            "PACE_SMOKE_LOG": str(log_path),
        }
    )

    subprocess.run(
        [
            str(
                PROJECT_ROOT
                / "scripts"
                / script_name
            )
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = log_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert arguments[:2] == [
        "--dataset",
        dataset,
    ]
    assert "--corpus-dir" in arguments
    assert str(tmp_path / "corpus") in arguments
    assert "--output-dir" in arguments
    assert str(
        tmp_path / "outputs" / dataset
    ) in arguments

    if dataset == "2wiki":
        assert "--corpus-coverage-report" in arguments
        assert str(
            tmp_path
            / "data/2wikimultihopqa/top100_complete"
            / "corpus_coverage/report.json"
        ) in arguments