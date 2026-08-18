#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PACE_DATA_ROOT:-${REPO_ROOT}/../data}"
OUTPUT_ROOT="${PACE_OUTPUT_ROOT:-${REPO_ROOT}/outputs/effectiveness}"
OUT="${OUTPUT_ROOT}/joint_d_to_k"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUT}"

"${PYTHON_BIN}" -m pace.evaluation.run_joint_d_to_k \
    --dataset hotpot \
    --source "${DATA_ROOT}/hotpotqa/top100_complete/cohort" \
    --cache-dir "${DATA_ROOT}/hotpotqa/top100_complete/cache" \
    --similarity-cache "${DATA_ROOT}/hotpotqa/top100_complete/cache/splade_similarity" \
    --calibration-manifest "${OUTPUT_ROOT}/hotpot/calibration.json" \
    --output-dir "${OUT}/hotpot"

"${PYTHON_BIN}" -m pace.evaluation.run_joint_d_to_k \
    --dataset 2wiki \
    --source "${DATA_ROOT}/2wikimultihopqa/top100_complete/cohort" \
    --cache-dir "${DATA_ROOT}/2wikimultihopqa/top100_complete/cache" \
    --similarity-cache "${DATA_ROOT}/2wikimultihopqa/top100_complete/cache/splade_similarity" \
    --calibration-manifest "${OUTPUT_ROOT}/2wiki/calibration.json" \
    --output-dir "${OUT}/2wiki"

"${PYTHON_BIN}" -m pace.evaluation.run_joint_d_to_k \
    --dataset musique \
    --source "${DATA_ROOT}/musique/musique_ans_v1.0_dev.jsonl" \
    --cache-dir "${DATA_ROOT}/musique/cache" \
    --similarity-cache "${DATA_ROOT}/musique/cache/splade_similarity" \
    --calibration-manifest "${OUTPUT_ROOT}/musique/calibration.json" \
    --d-values "5,10,20" \
    --output-dir "${OUT}/musique"

"${PYTHON_BIN}" - "${OUT}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []

for dataset in ("hotpot", "2wiki", "musique"):
    path = root / dataset / "joint_recall.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))

output = root / "joint_recall.csv"
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(rows[0]),
    )
    writer.writeheader()
    writer.writerows(rows)
PY
