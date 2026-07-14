# Variational Reasoning

Minimal code for training and evaluating a PTRM/TRM model from scratch with IVON.

## Setup

Use Python with CUDA-enabled PyTorch and install the packages imported by this project, including `numpy`, `tqdm`, and `ivon`.

The launch scripts expect this directory layout by default:

```text
repository-root/
├── variational_reasoning/
├── third_party/TinyRecursiveModels/
└── data/sudoku-extreme-1k-aug-1000/
```

The paths can instead be supplied with the `TRM_REPO` and `DATASET` environment variables.

## Train

Run from the repository root:

```bash
./variational_reasoning/code/ptrm/scripts/run_ivon_scratch_train.sh
```

Common settings can be changed with environment variables:

```bash
RUN_NAME=my_run CUDA_VISIBLE_DEVICES=1 TRAIN_STEPS=50000 \
  ./variational_reasoning/code/ptrm/scripts/run_ivon_scratch_train.sh
```

Additional training options can be appended directly:

```bash
./variational_reasoning/code/ptrm/scripts/run_ivon_scratch_train.sh \
  --ivon-noise-scale 0.3 --history-interval 50
```

The default learning-rate schedule is in `code/ptrm/configs/ivon_scratch_schedule.json`.

### Weights & Biases

W&B logging is disabled by default. Enable it explicitly when needed:

```bash
WANDB_ENABLED=1 WANDB_PROJECT=my-project \
  ./variational_reasoning/code/ptrm/scripts/run_ivon_scratch_train.sh
```

Set `WANDB_ENTITY` only when an entity is required.

## Evaluate

The evaluation script uses the final checkpoint from the selected run:

```bash
RUN_NAME=my_run \
  ./variational_reasoning/code/ptrm/scripts/run_ivon_scratch_eval.sh
```

Example with posterior comparison and 20 samples:

```bash
RUN_NAME=my_run METHOD=ivon_compare_selection K=20 \
  ./variational_reasoning/code/ptrm/scripts/run_ivon_scratch_eval.sh
```

Use `IVON_CHECKPOINT=/path/to/checkpoint.pt` to evaluate a different checkpoint. Training outputs are written to `outputs/ivon_scratch/<run-name>/` by default.

For all available arguments:

```bash
python3 -m variational_reasoning.code.ptrm.train_scratch --help
python3 -m variational_reasoning.code.ptrm.eval --help
```
