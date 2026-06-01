#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-$REPO_DIR/model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mineru-diffusion}"

ARGS=(
  --model-path "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --max-length "$MAX_LENGTH"
  --block-size "$BLOCK_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --data-parallel-size "$DATA_PARALLEL_SIZE"
)

if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi

python "$REPO_DIR/engines/nano_dvlm/server.py" "${ARGS[@]}"
