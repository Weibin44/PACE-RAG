#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?Usage: prepare_cache.sh DATASET SOURCE CACHE_DIR [DEVICE]}"
SOURCE="${2:?Missing source path}"
CACHE_DIR="${3:?Missing cache directory}"
DEVICE="${4:-auto}"

case "${DATASET}" in
    hotpot|2wiki)
        COVERAGE_MAX_LENGTH=128
        ;;
    musique)
        COVERAGE_MAX_LENGTH=256
        ;;
    *)
        echo "Unsupported dataset: ${DATASET}" >&2
        exit 2
        ;;
esac

echo "Dataset: ${DATASET}"
echo "Source: ${SOURCE}"
echo "Cache directory: ${CACHE_DIR}"
echo "Device: ${DEVICE}"
echo "Coverage max length: ${COVERAGE_MAX_LENGTH}"
echo "Batch sizes: reranker=8, llm=10, provence=4"

pace-prepare-cache \
    --dataset "${DATASET}" \
    --stage coverage_features \
    --source "${SOURCE}" \
    --cache-dir "${CACHE_DIR}" \
    --device "${DEVICE}"

pace-prepare-cache \
    --dataset "${DATASET}" \
    --stage reranker_scores \
    --source "${SOURCE}" \
    --cache-dir "${CACHE_DIR}" \
    --device "${DEVICE}"

pace-prepare-cache \
    --dataset "${DATASET}" \
    --stage splade_similarity \
    --source "${SOURCE}" \
    --cache-dir "${CACHE_DIR}" \
    --device "${DEVICE}"