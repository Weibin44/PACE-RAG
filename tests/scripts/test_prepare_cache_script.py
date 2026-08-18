"""Smoke tests for the cache preparation shell script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_cache.sh"


@pytest.mark.parametrize(
    "dataset",
    ["hotpot", "2wiki", "musique"],
)
def test_prepare_cache_stages(
    tmp_path: Path,
    dataset: str,
):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()

    log_path = tmp_path / "calls.txt"
    fake_command = binary_dir / "pace-prepare-cache"
    fake_command.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\t" "$@" >> "${PACE_SMOKE_LOG}"\n'
        'printf "\\n" >> "${PACE_SMOKE_LOG}"\n',
        encoding="utf-8",
    )
    fake_command.chmod(0o755)

    source = tmp_path / "source"
    cache_dir = tmp_path / "cache"

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": (
                f"{binary_dir}:"
                f"{environment.get('PATH', '')}"
            ),
            "PACE_SMOKE_LOG": str(log_path),
        }
    )

    result = subprocess.run(
        [
            str(SCRIPT),
            dataset,
            str(source),
            str(cache_dir),
            "cuda:3",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = [
        line.rstrip("\t").split("\t")
        for line in log_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    expected_common = [
        "--dataset",
        dataset,
    ]

    assert calls == [
        [
            *expected_common,
            "--stage",
            "coverage_features",
            "--source",
            str(source),
            "--cache-dir",
            str(cache_dir),
            "--device",
            "cuda:3",
        ],
        [
            *expected_common,
            "--stage",
            "reranker_scores",
            "--source",
            str(source),
            "--cache-dir",
            str(cache_dir),
            "--device",
            "cuda:3",
        ],
        [
            *expected_common,
            "--stage",
            "splade_similarity",
            "--source",
            str(source),
            "--cache-dir",
            str(cache_dir),
            "--device",
            "cuda:3",
        ],
    ]

    assert (
        "Batch sizes: reranker=8, llm=10, provence=4"
        in result.stdout
    )


def test_prepare_cache_rejects_unknown_dataset(
    tmp_path: Path,
):
    result = subprocess.run(
        [
            str(SCRIPT),
            "unknown",
            str(tmp_path / "source"),
            str(tmp_path / "cache"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    

    assert result.returncode == 2
    assert "Unsupported dataset: unknown" in result.stderr