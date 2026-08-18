"""Smoke tests for dataset launch scripts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    (
        "script_name",
        "dataset",
        "source_suffix",
        "cache_suffix",
        "output_suffix",
    ),
    [
        (
            "run_hotpot.sh",
            "hotpot",
            "hotpotqa/top100_complete/cohort",
            "hotpotqa/top100_complete/cache",
            "hotpot",
        ),
        (
            "run_2wiki.sh",
            "2wiki",
            "2wikimultihopqa/top100_complete/cohort",
            "2wikimultihopqa/top100_complete/cache",
            "2wiki",
        ),
        (
            "run_musique.sh",
            "musique",
            "musique/musique_ans_v1.0_dev.jsonl",
            "musique/cache",
            "musique",
        ),
    ],
)
def test_launch_script_arguments(
    tmp_path: Path,
    script_name: str,
    dataset: str,
    source_suffix: str,
    cache_suffix: str,
    output_suffix: str,
):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()

    log_path = tmp_path / "arguments.txt"
    fake_evaluator = binary_dir / "pace-evaluate"
    fake_evaluator.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$@" > "${PACE_SMOKE_LOG}"\n',
        encoding="utf-8",
    )
    fake_evaluator.chmod(0o755)

    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": (
                f"{binary_dir}:"
                f"{environment.get('PATH', '')}"
            ),
            "PACE_DATA_ROOT": str(data_root),
            "PACE_OUTPUT_ROOT": str(output_root),
            "PACE_SMOKE_LOG": str(log_path),
            "PREPARE_CACHE": "0",
        }
    )

    subprocess.run(
        [str(PROJECT_ROOT / "scripts" / script_name)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = log_path.read_text(
        encoding="utf-8"
    ).splitlines()

    source = data_root / source_suffix
    cache = data_root / cache_suffix
    output = output_root / output_suffix

    assert arguments == [
        "--dataset",
        dataset,
        "--source",
        str(source),
        "--cache-dir",
        str(cache),
        "--similarity-cache",
        str(cache / "splade_similarity"),
        "--output-dir",
        str(output),
        "--calibration-manifest",
        str(output / "calibration.json"),
    ]