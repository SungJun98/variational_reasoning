# Variational Reasoning

Minimal code for training a PTRM/TRM model from scratch or fine-tuning a
compatible checkpoint with IVON, followed by posterior-sampling evaluation.

## Setup

Use Python with CUDA-enabled PyTorch and install the packages imported by this project, including `numpy`, `tqdm`, and `ivon`.

The launch scripts expect this directory layout by default:

```text
repository-root/
├── optim/
├── ptrm/
├── third_party/TinyRecursiveModels/
└── data/sudoku-extreme-1k-aug-1000/
```

The paths can instead be supplied with the `TRM_REPO` and `DATASET` environment variables.

## Hyperparameter Search Results

The compact HP-search snapshot is available at [`experiments/from_scratch_ivon_hp_results.csv`](experiments/from_scratch_ivon_hp_results.csv).
It was generated on `2026-07-14 11:34:41 UTC`.

- The snapshot contains `701` completed `K=10` evaluation rows.
- The snapshot contains `6` runs that were still training at the snapshot time.
- Each row records the training HPs and the training/evaluation data scope.
- The only result metrics are `selected_exact_pct` and `pass_at_10_exact_pct`.

Exact duplicate evaluations are removed.
When one run was evaluated at several checkpoints, the CSV retains the checkpoint with the best selected accuracy and the checkpoint with the best pass@10 accuracy.
These can be different checkpoints.
`BEST_SELECTED` and `BEST_PASS_AT_10` are assigned only within the same comparison group: training-data scope, evaluation set/limit, depth, and evaluation method.

`train_data_scope` and `evaluation_set` should be read separately.
The `evaluation_set` column contains `subset` or `full` for evaluated rows.
For example, `full_train` + `subset` means that the model trained on the full training set but was evaluated on a limited puzzle subset.
Full-set completion is recorded separately through `status`, `eval_count`, `eval_total`, and `eval_fraction`.
Rows with `status=running` have an empty `evaluation_set` and empty result fields.

### Reference Baselines

These are the full-set reference results used for comparison in the project.
They are not rows from the from-scratch HP-search CSV.
Subset results in the CSV are diagnostic measurements and should not be treated as directly matched full-set comparisons.

| method | inference diversity | evaluation set | selected | pass@10 |
|---|---|---|---:|---:|
| TRM | deterministic | full | 87.18% | 87.18% |
| PTRM | Gaussian latent noise | full | 95.79% | 95.80% |
| IVON fine-tuning | IVON posterior samples | full | **98.41%** | **98.43%** |

### Current Best Results in the Snapshot

| training data | evaluation set | best for | run | step | key HP | selected | pass@10 |
|---|---|---|---|---:|---|---:|---:|
| full | subset (1,000), D16 | selected | `ess3e7_lr09_post12_r005` | 30K | batch=768, lr=9e-4, clip=0.2, ESS=3e7, h0=1, beta2=0.99, noise=0.275, q=0.35, post-12K LR ratio=0.05 | **90.40%** | 90.70% |
| full | subset (1,000), D16 | pass@10 | `v8b03_highpass_b128_q030_n025` | 45K | batch=128, lr=7e-4, clip=0.2, ESS=1e7, h0=1, beta2=0.99, noise=0.25, q=0.30 | 88.80% | **91.60%** |
| 10% proxy | subset (1,000), D16 | selected and pass@10 | `p768_lr09_ess3e7_n025` | 8K | batch=768, lr=9e-4, clip=0.2, ESS=3e7, h0=1, beta2=0.99, noise=0.25, q=0.35 | **79.50%** | **79.70%** |
| full | full (partial: 96,448 / 422,786) | selected and pass@10 | `ess3e7_lr09_post12_r005` | 30K | same HP as the best-selected subset row | **93.54%** | **93.64%** |

The full-evaluation row was stopped after `22.8%` of the test set, so it must not be compared as if it were a completed full evaluation.
Regenerate the CSV before sharing newer results because the listed running experiments may have advanced beyond this snapshot.

## Train From Scratch

Run from the repository root:

```bash
./ptrm/scripts/run_ivon_scratch_train.sh
```

Common settings can be changed with environment variables:

```bash
RUN_NAME=my_run CUDA_VISIBLE_DEVICES=1 TRAIN_STEPS=50000 \
  ./ptrm/scripts/run_ivon_scratch_train.sh
```

Additional training options can be appended directly:

```bash
./ptrm/scripts/run_ivon_scratch_train.sh \
  --ivon-noise-scale 0.3 --history-interval 50
```

The default learning-rate schedule is in `ptrm/configs/ivon_scratch_schedule.json`.

## Fine-Tune a Checkpoint

Fine-tuning uses the same model, IVON optimizer, exact-resume checkpoint format,
and evaluator as from-scratch training. A compatible pretrained PTRM/TRM
checkpoint is required and is not bundled with this repository.

The base checkpoint may be either a raw model state dictionary or a checkpoint
dictionary containing `model_state_dict`. Use `BASE_STATE_KEY` when the weights
are stored under another key, such as `ema_model_state_dict`. The model
architecture options passed to `train_ft.py` must match the base checkpoint.

Run the reference fine-tuning configuration from the repository root:

```bash
BASE_CHECKPOINT=/path/to/base-checkpoint.pt \
  ./ptrm/scripts/run_ivon_ft_train.sh
```

Common settings can be changed with environment variables:

```bash
BASE_CHECKPOINT=/path/to/base-checkpoint.pt \
RUN_NAME=my_ft_run CUDA_VISIBLE_DEVICES=1 TRAIN_STEPS=500 \
  ./ptrm/scripts/run_ivon_ft_train.sh
```

Additional non-scheduled options can be appended directly. Options appended to
the command take precedence over launcher defaults:

```bash
BASE_CHECKPOINT=/path/to/base-checkpoint.pt \
  ./ptrm/scripts/run_ivon_ft_train.sh \
  --ivon-noise-scale 0.8 --ivon-hess-init 4.0
```

The tracked reference schedule is in `ptrm/configs/ivon_ft_schedule.json`. Its
main defaults are 200 optimizer steps, dense LR `1e-4`, puzzle-embedding LR
`1e-2`, ESS `3e5`, Hessian initialization `3.0`, IVON beta2 `0.99999`, IVON
weight decay `0.225`, and no finite clipping radius. Outputs are written to
`outputs/ivon_ft/<run-name>/` by default.

The schedule controls dense LR, puzzle-embedding LR, ESS, and the Hessian
approximation at every optimizer step. To override one of those values, copy
the schedule JSON, edit it, and pass its path through `SCHEDULE_JSON`.

### Weights & Biases

W&B logging is disabled by default. Enable it explicitly when needed:

```bash
WANDB_ENABLED=1 WANDB_PROJECT=my-project \
  ./ptrm/scripts/run_ivon_scratch_train.sh
```

Set `WANDB_ENTITY` only when an entity is required.

## Evaluate

The generic evaluator accepts final checkpoints produced by either training
entrypoint:

```bash
IVON_CHECKPOINT=outputs/ivon_ft/my_ft_run/checkpoints/checkpoint.pt \
  ./ptrm/scripts/run_ivon_eval.sh
```

Example with posterior comparison and 20 samples:

```bash
IVON_CHECKPOINT=outputs/ivon_scratch/my_run/checkpoints/checkpoint.pt \
METHOD=ivon_compare_selection K=20 \
  ./ptrm/scripts/run_ivon_eval.sh
```

`run_ivon_scratch_eval.sh` remains available as a compatibility wrapper that
selects `outputs/ivon_scratch/<run-name>/checkpoints/checkpoint.pt` from
`RUN_NAME`.

For all available arguments:

```bash
python3 -m ptrm.train_scratch --help
python3 -m ptrm.train_ft --help
python3 -m ptrm.eval --help
```
