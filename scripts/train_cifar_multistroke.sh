#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:=data/cifar10}"
: "${OUTPUT_DIR:=runs/cifar10_multistroke}"
: "${NUM_PROCESSES:=2}"

torchrun --standalone --nproc_per_node="${NUM_PROCESSES}" scripts/train_cifar_ddpm.py \
  --method multistroke \
  --config configs/cifar10_ddpm.yaml \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
