#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PACE_DATA_ROOT:-${REPO_ROOT}/../data}"
OUTPUT_ROOT="${PACE_COHORT_OUTPUT_ROOT:-${REPO_ROOT}/outputs/cohorts}"

: "${BERGEN_CORPUS_DIR:?Set BERGEN_CORPUS_DIR first}"

pace-materialize-cohort \
    --dataset 2wiki \
    --labels "${DATA_ROOT}/2wikimultihopqa/dev.json" \
    --retrieval-dir "${DATA_ROOT}/2wikimultihopqa/top100_complete/corpus_coverage" \
    --corpus-coverage-report "${DATA_ROOT}/2wikimultihopqa/top100_complete/corpus_coverage/report.json" \
    --corpus-dir "${BERGEN_CORPUS_DIR}" \
    --output-dir "${OUTPUT_ROOT}/2wiki" \
    --num-parts 8