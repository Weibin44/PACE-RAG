#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONLINE_OUTPUT_ROOT="${PACE_ONLINE_OUTPUT_ROOT:-${REPO_ROOT}/outputs/online}"
INPUT_DIR="${ONLINE_OUTPUT_ROOT}/hotpot"
OUTPUT_DIR="${INPUT_DIR}/paper_plots"

if [[ ! -f "${INPUT_DIR}/summary.jsonl" ]]; then
    echo "Missing online summary: ${INPUT_DIR}/summary.jsonl" >&2
    exit 1
fi

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pace-matplotlib}" \
pace-plot-serving \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}"