#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a trained baseline checkpoint directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for generated images}"
: "${NUM_PROCESSES:?Set NUM_PROCESSES to the local torchrun process count}"

LOAD_FILE="${LOAD_FILE:-model_latest.pkl}"
LOAD_DEF_FILE="${LOAD_DEF_FILE:-model_params_latest.json}"
[[ -f "$CHECKPOINT_DIR/$LOAD_FILE" ]] || { echo "Missing checkpoint: $CHECKPOINT_DIR/$LOAD_FILE" >&2; exit 2; }
[[ -f "$CHECKPOINT_DIR/$LOAD_DEF_FILE" ]] || { echo "Missing checkpoint definition: $CHECKPOINT_DIR/$LOAD_DEF_FILE" >&2; exit 2; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

torchrun --standalone --nnodes=1 --nproc-per-node "$NUM_PROCESSES" -m src.sample \
  --load-dir "$CHECKPOINT_DIR" \
  --load-file "$LOAD_FILE" \
  --load-def-file "$LOAD_DEF_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --num-samples "${NUM_SAMPLES:-5000}" \
  --batch-size "${SAMPLE_BATCH_SIZE:-64}" \
  --num-steps 100 \
  --guidance "${GUIDANCE:-3}" \
  --sampler euler

