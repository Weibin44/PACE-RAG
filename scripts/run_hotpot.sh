#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PACE_DATA_ROOT:-"${REPO_ROOT}/../data"}"
OUTPUT_ROOT="${PACE_OUTPUT_ROOT:-"${REPO_ROOT}/outputs/effectiveness"}"

SOURCE="${DATA_ROOT}/hotpotqa/top100_complete/cohort"
CACHE_DIR="${DATA_ROOT}/hotpotqa/top100_complete/cache"
OUTPUT_DIR="${OUTPUT_ROOT}/hotpot"
DEVICE="${PACE_DEVICE:-auto}"

if [[ "${PREPARE_CACHE:-1}" == "1" ]]; then
    "${REPO_ROOT}/scripts/prepare_cache.sh" \
        hotpot "${SOURCE}" "${CACHE_DIR}" "${DEVICE}"
fi

pace-evaluate \
    --dataset hotpot \
    --source "${SOURCE}" \
    --cache-dir "${CACHE_DIR}" \
    --similarity-cache "${CACHE_DIR}/splade_similarity" \
    --output-dir "${OUTPUT_DIR}" \
    --calibration-manifest "${OUTPUT_DIR}/calibration.json"
