#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a checkpoint directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the sample output directory}"
: "${NUM_PROCESSES:=8}"
: "${SAMPLE_BATCH_SIZE:=128}"
: "${METHOD:=auto}"

torchrun --standalone --nproc_per_node="${NUM_PROCESSES}" scripts/sample_cifar_ddpm.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --method "${METHOD}" \
  --batch-size "${SAMPLE_BATCH_SIZE}" \
  --num-inference-steps 100 \
  --model-var-type fixedsmall \
  "$@"
