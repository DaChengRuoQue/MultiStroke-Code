#!/usr/bin/env bash
set -euo pipefail

: "${IMAGENET_DATASET_ROOT:?Set IMAGENET_DATASET_ROOT to the ImageNet dataset directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints and logs}"
: "${NUM_PROCESSES:?Set NUM_PROCESSES to the local torchrun process count}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
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
export TRAINING_MODE=baseline
export TRAIN_SEED="${TRAIN_SEED:-0}"
export SAVE_DIR="$OUTPUT_DIR"
export GLOBAL_BATCH_SIZE BATCH_SIZE_PER_RANK

torchrun --standalone --nnodes=1 --nproc-per-node "$NUM_PROCESSES" -m src.train

