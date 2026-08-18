#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PACE_DATA_ROOT:-${REPO_ROOT}/../data}"
EFFECTIVENESS_ROOT="${PACE_OUTPUT_ROOT:-${REPO_ROOT}/outputs/effectiveness}"
ONLINE_OUTPUT_ROOT="${PACE_ONLINE_OUTPUT_ROOT:-${REPO_ROOT}/outputs/online}"

for path in \
    "${DATA_ROOT}/hotpotqa/top100_complete/cohort" \
    "${DATA_ROOT}/hotpotqa/top100_complete/cache/coverage_features" \
    "${DATA_ROOT}/hotpotqa/top100_complete/cache/splade_similarity" \
    "${DATA_ROOT}/hotpotqa/top100_complete/workloads/heldout_1087" \
    "${EFFECTIVENESS_ROOT}/hotpot/calibration.json"
do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required input: ${path}" >&2
        exit 1
    fi
done

# python -m pace.serving.benchmark \
pace-serve \
    --source "${DATA_ROOT}/hotpotqa/top100_complete/cohort" \
    --cache-dir "${DATA_ROOT}/hotpotqa/top100_complete/cache" \
    --calibration-manifest "${EFFECTIVENESS_ROOT}/hotpot/calibration.json" \
    --workload-dir "${DATA_ROOT}/hotpotqa/top100_complete/workloads/heldout_1087" \
    --output-dir "${ONLINE_OUTPUT_ROOT}/hotpot" \
    --qps-values "0.5,0.75,1.0,1.15,1.3,1.45,1.6,1.8" \
    --warmup-seconds 60 \
    --measurement-seconds 300 \
    --unique-measurement-pass \
    --doc-count 100 \
    --candidate-pool-size 100 \
    --dynamic-d-min 20 \
    --top-k 5 \
    --output-tokens 128 \
    --natural-eos \
    --frontend-device cuda:0 \
    --llm-device cuda:1 \
    --provence-device cuda:2


# only test one method with one QPS
# pace-serve \
#   --source "${PACE_DATA_ROOT}/hotpotqa/top100_complete/cohort" \
#   --cache-dir "${PACE_DATA_ROOT}/hotpotqa/top100_complete/cache" \
#   --calibration-manifest "${PACE_OUTPUT_ROOT}/hotpot/calibration.json" \
#   --workload-dir "${PACE_DATA_ROOT}/hotpotqa/top100_complete/workloads/heldout_1087" \
#   --output-dir /tmp/pace_online_qps_0p5_check \
#   --methods queue_adaptive_soft_anchor \
#   --qps-values 0.5 \
#   --warmup-seconds 60 \
#   --measurement-seconds 300 \
#   --unique-measurement-pass \
#   --doc-count 100 \
#   --candidate-pool-size 100 \
#   --dynamic-d-min 20 \
#   --top-k 5 \
#   --output-tokens 128 \
#   --natural-eos \
#   --frontend-device cuda:0 \
#   --llm-device cuda:1 \
#   --provence-device cuda:2
