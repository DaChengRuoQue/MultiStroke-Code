#!/usr/bin/env bash
set -euo pipefail

: "${IMAGENET_DATASET_ROOT:?Set IMAGENET_DATASET_ROOT to the ImageNet dataset directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for corrector checkpoints and logs}"
: "${NUM_PROCESSES:?Set NUM_PROCESSES to the local torchrun process count}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
if ! [[ "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ && "$GLOBAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_PROCESSES and GLOBAL_BATCH_SIZE must be positive integers" >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE % NUM_PROCESSES != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by NUM_PROCESSES" >&2
  exit 2
fi
BATCH_SIZE_PER_RANK=$((GLOBAL_BATCH_SIZE / NUM_PROCESSES))

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export BATCH_SIZE_PER_RANK

arguments=(
  --dataset-root "$IMAGENET_DATASET_ROOT"
  --output-dir "$OUTPUT_DIR"
  --total-steps "${TOTAL_STEPS:-10000}"
  --batch-size "$BATCH_SIZE_PER_RANK"
  --learning-rate "${LEARNING_RATE:-1e-4}"
  --weight-decay "${WEIGHT_DECAY:-1e-2}"
  --num-workers "${NUM_WORKERS:-10}"
  --coarsen-factor 8
  --save-every "${SAVE_EVERY:-10000}"
  --log-every "${LOG_EVERY:-10}"
)
if [[ "${RESUME:-0}" == "1" ]]; then
  arguments+=(--resume)
fi

torchrun --standalone --nnodes=1 --nproc-per-node "$NUM_PROCESSES" -m src.train_corrector "${arguments[@]}"

