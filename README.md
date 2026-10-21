# MultiStroke

This repository contains the ImageNet-256 flow-matching and CIFAR-10 DDPM implementations for matched baselines and MultiStroke. It provides focused training and sampling entry points without model checkpoints, generated outputs, or dataset files.

## Method overview

The baseline trains a class-conditioned latent flow model on ImageNet images encoded by `stabilityai/sd-vae-ft-ema`.

The default MultiStroke configuration uses the same model, optimizer, training duration, and global batch size. Before VAE encoding, it blends each image with an 8-by-8 block-averaged representation according to the continuous flow time. The roughness schedule is aligned with 100 Euler sampling steps.

The lightweight corrector is trained separately with the VAE frozen. Its input is the latent representation of an image at one roughness level. Its target is the latent residual to the immediately finer roughness level of the same image. During corrected sampling, it refines the predicted image endpoint between adjacent Euler steps.

For CIFAR-10, the matched experiment trains a class-conditional pixel-space DDPM for 100 diffusion steps. MultiStroke applies the two-pixel block operator to both the noisy model input and the noise target. Sampling follows the same original diffusion coordinates, clips the predicted clean image in ordinary image space, and uses the same fixed-small posterior variance for the baseline and MultiStroke models.

## Installation

Python 3.10 and a CUDA-enabled PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The VAE is loaded from Hugging Face on first use and cached under `VAE/`.

## CIFAR-10 DDPM experiments

The checked-in configuration at `configs/cifar10_ddpm.yaml` reproduces the matched setup used for the baseline and MultiStroke comparison. Both runs use the same U-Net, optimizer, training duration, per-process batch size, and diffusion schedule. The only training difference is the MultiStroke forward process and target.

The default launchers use two processes and a per-process batch size of 40. Set `NUM_PROCESSES` to match the available GPUs. Changing it changes the global batch size unless `train_batch_size` or gradient accumulation is adjusted accordingly.

Download CIFAR-10 and train the baseline:

```bash
DATA_ROOT=/path/to/cifar10 \
OUTPUT_DIR=/path/to/runs/cifar10_ddpm_baseline \
NUM_PROCESSES=2 \
bash scripts/train_cifar_baseline.sh --download
```

Train MultiStroke with the same configuration:

```bash
DATA_ROOT=/path/to/cifar10 \
OUTPUT_DIR=/path/to/runs/cifar10_multistroke \
NUM_PROCESSES=2 \
bash scripts/train_cifar_multistroke.sh
```

To resume either run, append `--resume /path/to/run/checkpoints/latest`. Checkpoints are written under `OUTPUT_DIR/checkpoints/latest`, with immutable milestone copies in adjacent `progress_*` directories.

Generate matched baseline samples:

```bash
CHECKPOINT_DIR=/path/to/runs/cifar10_ddpm_baseline/checkpoints/latest \
OUTPUT_DIR=/path/to/samples/cifar10_ddpm_baseline \
NUM_PROCESSES=8 \
bash scripts/sample_cifar.sh
```

Generate MultiStroke samples by pointing the same sampler at the MultiStroke checkpoint:

```bash
CHECKPOINT_DIR=/path/to/runs/cifar10_multistroke/checkpoints/latest \
OUTPUT_DIR=/path/to/samples/cifar10_multistroke \
NUM_PROCESSES=8 \
bash scripts/sample_cifar.sh
```

The sampler reads the method and diffusion settings from the checkpoint. It defaults to 100 ancestral steps, fixed-small variance, clean-space clipping, a per-process batch size of 128, and indexed PNG output under `OUTPUT_DIR/images`. Append `--resume` to skip existing image indices after an interrupted run. Use the same standard CIFAR-10 reference statistics and evaluator for both output directories when computing FID.

Older checkpoints that do not store the method remain supported. Set `METHOD=ddpm` or `METHOD=multistroke` when invoking `scripts/sample_cifar.sh` for those checkpoints.

## ImageNet data

Set `IMAGENET_DATASET_ROOT` to a local ImageNet-1K training dataset. The loaders support either:

- Hugging Face Arrow shards named `imagenet-1k-train-*.arrow`; or
- a directory readable by `datasets.load_dataset("parquet", data_dir=...)`.

Each record must provide an `image` field and an integer `label` field. Images are converted to RGB, resized and center-cropped to 256 pixels, and normalized to the range expected by the VAE.

## Reference training configuration

The baseline and MultiStroke launchers share these defaults:

| Setting | Value |
|---|---:|
| Training updates | 100,000 |
| Global batch size | 512 |
| Learning rate | 0.0001 |
| Warmup updates | 1,000 |
| Weight decay | 0.01 |
| EMA update interval | 100 |
| EMA decay | 0.99 |
| Transformer blocks | 17 |
| Token dimension | 1,088 |
| Class-conditioning dimension | 2,048 |
| Patch size | 2 |

The launchers derive the per-process batch size from `GLOBAL_BATCH_SIZE` and `NUM_PROCESSES`. The global batch must be divisible by the process count. GPU count is an execution choice and is not part of the method configuration.

## Train the baseline

```bash
IMAGENET_DATASET_ROOT=/path/to/imagenet \
OUTPUT_DIR=/path/to/checkpoints/baseline \
NUM_PROCESSES=<local_process_count> \
bash scripts/train_baseline.sh
```

Optional environment variables include `GLOBAL_BATCH_SIZE`, `TOTAL_STEPS`, `SAVE_STEPS`, `LOG_STEPS`, `NUM_WORKERS`, `LEARNING_RATE`, and `AUTO_RESUME`. The rolling checkpoint is resumed when all associated model, optimizer, scheduler, and scaler files are present and `AUTO_RESUME` is enabled.

## Train MultiStroke

```bash
IMAGENET_DATASET_ROOT=/path/to/imagenet \
OUTPUT_DIR=/path/to/checkpoints/multistroke \
NUM_PROCESSES=<local_process_count> \
bash scripts/train_multistroke.sh
```

This command changes only the training objective. The default coarsening factor is 8 and the schedule contains 100 roughness levels.

## Train the lightweight corrector

```bash
IMAGENET_DATASET_ROOT=/path/to/imagenet \
OUTPUT_DIR=/path/to/checkpoints/corrector \
NUM_PROCESSES=<local_process_count> \
bash scripts/train_corrector.sh
```

The corrector launcher defaults to 10,000 updates, a global batch size of 256, a constant learning rate of 0.0001, and adjacent roughness transitions. Set `RESUME=1` to continue from `corrector_latest.pt`.

## Sample the baseline

```bash
CHECKPOINT_DIR=/path/to/checkpoints/baseline \
OUTPUT_DIR=/path/to/samples/baseline \
NUM_PROCESSES=<local_process_count> \
bash scripts/sample_baseline.sh
```

The sampler writes indexed PNG files under `OUTPUT_DIR/images`. Existing indices are skipped by default, allowing interrupted generation to resume without changing the remaining per-image conditioning.

## Sample MultiStroke with the corrector

```bash
CHECKPOINT_DIR=/path/to/checkpoints/multistroke \
CORRECTOR_CHECKPOINT=/path/to/checkpoints/corrector/corrector_latest.pt \
OUTPUT_DIR=/path/to/samples/multistroke_corrected \
NUM_PROCESSES=<local_process_count> \
bash scripts/sample_multistroke_corrector.sh
```

Corrector-assisted sampling requires Euler integration with 100 steps. The sampler verifies the corrector checkpoint type and architecture before initializing CUDA.

Sampling throughput can be adjusted with `SAMPLE_BATCH_SIZE`. This value does not change the trained model configuration.

## Checkpoints and outputs

Flow training writes:

- `model_latest.pkl` and `model_params_latest.json`;
- `optim_latest.pkl`, `scheduler_latest.pkl`, and `scaler_latest.pkl`;
- milestone checkpoints at the configured save updates; and
- `training_loss.csv`.

Corrector training writes `corrector_latest.pt`, milestone corrector checkpoints, and `training_loss.csv`.

ImageNet data, VAE files, trained checkpoints, and generated images are not included in this repository.

## Direct entry points

The shell launchers are thin wrappers around:

```bash
python -m src.train --help
python -m src.train_corrector --help
python -m src.sample --help
python scripts/train_cifar_ddpm.py --help
python scripts/sample_cifar_ddpm.py --help
```

Use the launchers for distributed runs. The ImageNet launchers validate required paths and batch divisibility; the CIFAR-10 launchers provide the matched experiment defaults.

## Troubleshooting

- `GLOBAL_BATCH_SIZE must be divisible by WORLD_SIZE`: select a global batch compatible with the process count.
- `ImageNet dataset root not found`: verify `IMAGENET_DATASET_ROOT` and dataset permissions.
- `Flow checkpoint not found`: verify `CHECKPOINT_DIR`, `LOAD_FILE`, and `LOAD_DEF_FILE`.
- `Latent corrector checkpoint not found`: verify `CORRECTOR_CHECKPOINT`.
- `The matched CIFAR experiment uses 100 diffusion steps`: use the checked-in CIFAR configuration for the reproduced experiment.
- `Missing trainer_state.pt`: point `CHECKPOINT_DIR` to a checkpoint directory such as `checkpoints/latest`, not the run root.
- `Sampling requires CUDA`: run sampling in a CUDA-enabled environment.
- Out-of-memory errors: reduce `SAMPLE_BATCH_SIZE` for sampling. For training, preserve the global batch by increasing process count when reducing the per-process batch.
