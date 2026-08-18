#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PACE_DATA_ROOT:-${REPO_ROOT}/../data}"
OUTPUT_ROOT="${PACE_COHORT_OUTPUT_ROOT:-${REPO_ROOT}/outputs/cohorts}"

: "${BERGEN_CORPUS_DIR:?Set BERGEN_CORPUS_DIR first}"

pace-materialize-cohort \
    --dataset hotpot \
    --labels "${DATA_ROOT}/hotpotqa/hotpot_dev_distractor_v1.json" \
    --evaluation "${DATA_ROOT}/hotpotqa/eval_dev_out.json" \
    --retrieval-dir "${DATA_ROOT}/hotpotqa/top100_complete/cohort/batches" \
    --corpus-dir "${BERGEN_CORPUS_DIR}" \
    --output-dir "${OUTPUT_ROOT}/hotpot" \
    --batch-size 100 \
    --seed 2026
