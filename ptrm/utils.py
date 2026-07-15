from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from optim.factory import (
    create_dense_optimizer,
    move_optimizer_state_to_device,
    optimizer_name,
    optimizer_sampling_context,
)


EXACT_FORMAT_VERSION = 4
RESUME_ARG_IGNORED_KEYS = {
    "device",
    "output_root",
    "run_name",
    "resume_checkpoint",
    "schedule_json",
    "stop_step",
    "status_interval",
    "history_interval",
    "checkpoint_interval",
    "wandb",
    "wandb_entity",
    "wandb_group",
    "wandb_run_id",
    "wandb_mode",
    "wandb_project",
}


class ExactTrainBatchStream:
    def __init__(self, config: Any, split: str = "train") -> None:
        from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

        self.dataset = PuzzleDataset(
            PuzzleDatasetConfig(
                seed=config.seed,
                dataset_paths=config.data_paths,
                global_batch_size=config.global_batch_size,
                test_set_mode=False,
                epochs_per_iter=config.epochs,
                rank=0,
                num_replicas=1,
            ),
            split=split,
        )
        self.dataset._lazy_load_dataset()
        self.set_names = list(self.dataset._data.keys())
        if not self.set_names:
            raise ValueError("Training dataset has no sets.")
        self.set_index = -1
        self.iteration = 0
        self.start_index = 0
        self.group_order: np.ndarray | None = None
        self.rng: np.random.Generator | None = None
        self._advance_set()

    @property
    def metadata(self) -> Any:
        return self.dataset.metadata

    def _build_group_order(self, set_name: str, rng: np.random.Generator) -> np.ndarray:
        dataset = self.dataset._data[set_name]
        group_count = dataset["group_indices"].size - 1
        return np.concatenate([rng.permutation(group_count) for _ in range(self.dataset.config.epochs_per_iter)])

    def _advance_set(self) -> None:
        self.set_index = (self.set_index + 1) % len(self.set_names)
        self.iteration += 1
        set_name = self.set_names[self.set_index]
        self.rng = np.random.Generator(np.random.Philox(seed=self.dataset.config.seed + self.iteration))
        self.group_order = self._build_group_order(set_name, self.rng)
        self.start_index = 0

    def state_dict(self) -> dict[str, Any]:
        if self.rng is None or self.group_order is None:
            raise RuntimeError("ExactTrainBatchStream is not initialized.")
        return {
            "set_index": self.set_index,
            "iteration": self.iteration,
            "start_index": self.start_index,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.set_index = int(state["set_index"])
        self.iteration = int(state["iteration"])
        self.start_index = int(state["start_index"])
        if not 0 <= self.set_index < len(self.set_names):
            raise ValueError(f"Invalid set_index in exact stream state: {self.set_index}")
        set_name = self.set_names[self.set_index]
        self.rng = np.random.Generator(np.random.Philox(seed=self.dataset.config.seed + self.iteration))
        self.group_order = self._build_group_order(set_name, self.rng)
        self.rng.bit_generator.state = state["rng_state"]

    def next_batch(self) -> tuple[str, dict[str, torch.Tensor], int]:
        from puzzle_dataset import _sample_batch

        while True:
            if self.rng is None or self.group_order is None:
                raise RuntimeError("ExactTrainBatchStream is not initialized.")
            if self.start_index >= self.group_order.size:
                self._advance_set()
                continue

            set_name = self.set_names[self.set_index]
            dataset = self.dataset._data[set_name]
            self.start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                self.rng,
                group_order=self.group_order,
                puzzle_indices=dataset["puzzle_indices"],
                group_indices=dataset["group_indices"],
                start_index=self.start_index,
                global_batch_size=self.dataset.config.global_batch_size,
            )
            global_effective_batch_size = batch_puzzle_indices.size
            if global_effective_batch_size < self.dataset.config.global_batch_size:
                self._advance_set()
                continue

            local_batch_size = self.dataset.local_batch_size
            batch_indices = batch_indices[:local_batch_size]
            batch_puzzle_indices = batch_puzzle_indices[:local_batch_size]
            batch = self.dataset._collate_batch(
                {
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices],
                }
            )
            return set_name, batch, global_effective_batch_size


def import_reference_code(trm_repo: Path) -> None:
    repo = trm_repo.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"TRM repo not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def update_status(status_path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    with status_path.open("a") as handle:
        handle.write(f"- `{timestamp}` {message}\n")


def numeric_metrics(metrics: dict[str, Any] | None) -> dict[str, float]:
    if not metrics:
        return {}
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float, bool))}


def load_schedule(path: Path) -> dict[str, Any]:
    schedule = json.loads(path.read_text())
    if not isinstance(schedule, dict):
        raise TypeError(f"schedule-json must contain a JSON object: {path}")
    return schedule


def piecewise_value(spec: dict[str, Any], step: int, default: Any) -> Any:
    values = spec.get("values")
    if not isinstance(values, list) or not values:
        return default
    last_value = default
    for item in values:
        if not isinstance(item, dict):
            raise TypeError("piecewise values must be objects")
        last_value = item["value"]
        end_step = item.get("end_step")
        if end_step is None or step <= int(end_step):
            return last_value
    return last_value


def linear_warmup_piecewise(spec: dict[str, Any], step: int, default: float) -> float:
    base = float(spec.get("base", default))
    warmup_steps = int(spec.get("warmup_steps", 0))
    if warmup_steps > 0 and step < warmup_steps:
        return base * float(step) / float(max(1, warmup_steps))
    return float(piecewise_value(spec, step, base))


def linear_warmup_cosine(spec: dict[str, Any], step: int, default: float) -> float:
    base = float(spec.get("base", default))
    warmup_steps = int(spec.get("warmup_steps", 0))
    total_steps = int(spec.get("total_steps", max(step, warmup_steps + 1)))
    min_ratio = float(spec.get("min_ratio", 1.0))
    if warmup_steps > 0 and step < warmup_steps:
        return base * float(step) / float(max(1, warmup_steps))
    if total_steps <= warmup_steps:
        return base
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return base * (min_ratio + max(0.0, (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


def schedule_value(schedule: dict[str, Any], key: str, step: int, default: Any) -> Any:
    spec = schedule.get(key)
    if spec is None:
        return default
    if isinstance(spec, (int, float, str)):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(f"Unsupported schedule spec for {key}: {spec!r}")
    schedule_type = spec.get("type", "constant")
    if schedule_type == "constant":
        return spec.get("value", default)
    if schedule_type == "piecewise":
        return piecewise_value(spec, step, default)
    if schedule_type == "linear_warmup_piecewise":
        return linear_warmup_piecewise(spec, step, float(default))
    if schedule_type == "linear_warmup_cosine":
        return linear_warmup_cosine(spec, step, float(default))
    raise ValueError(f"Unsupported schedule type for {key}: {schedule_type}")


def apply_optimizer_schedule(
    dense_optimizer: Any,
    schedule: dict[str, Any],
    step: int,
    defaults: argparse.Namespace,
) -> dict[str, Any]:
    name = optimizer_name(dense_optimizer)
    dense_lr = float(schedule_value(schedule, "dense_lr", step, defaults.lr))
    for group in dense_optimizer.param_groups:
        group["lr"] = dense_lr
    values: dict[str, Any] = {"dense_lr": dense_lr, "optimizer": name}

    if name == "ivon":
        ivon_ess = float(
            schedule_value(schedule, "ivon_ess", step, defaults.ivon_ess)
        )
        hess_approx = str(
            schedule_value(
                schedule, "ivon_hess_approx", step, defaults.ivon_hess_approx
            )
        )
        if hess_approx not in ("price", "gradsq"):
            raise ValueError(f"Invalid scheduled ivon_hess_approx: {hess_approx}")
        dense_optimizer.hess_approx = hess_approx
        for group in dense_optimizer.param_groups:
            group["ess"] = ivon_ess
        values.update(ivon_ess=ivon_ess, ivon_hess_approx=hess_approx)
    elif name == "evon":
        default_ess = (
            defaults.evon_ess
            if getattr(defaults, "evon_ess", None) is not None
            else defaults.ivon_ess
        )
        schedule_key = "evon_ess" if "evon_ess" in schedule else "ivon_ess"
        evon_ess = float(schedule_value(schedule, schedule_key, step, default_ess))
        for group in dense_optimizer.param_groups:
            group["ess"] = evon_ess
        values["evon_ess"] = evon_ess
    return values


def apply_ivon_schedule(
    dense_optimizer: Any,
    schedule: dict[str, Any],
    step: int,
    defaults: argparse.Namespace,
) -> dict[str, Any]:
    """Backward-compatible alias for the optimizer-agnostic scheduler."""
    return apply_optimizer_schedule(dense_optimizer, schedule, step, defaults)


def make_config(args: argparse.Namespace) -> Any:
    from pretrain import PretrainConfig

    if args.hidden_size % args.num_heads != 0:
        raise ValueError(f"hidden_size must be divisible by num_heads: {args.hidden_size}/{args.num_heads}")
    if args.puzzle_emb_ndim == 0 and args.puzzle_emb_len != 0:
        raise ValueError("--puzzle-emb-len must be 0 when --puzzle-emb-ndim is 0.")
    train_group_count = max(1, int(getattr(args, "train_group_count_for_epochs", 1000)))
    epochs = max(1, math.ceil(args.train_steps * args.global_batch_size / train_group_count))
    return PretrainConfig(
        arch={
            "name": "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1",
            "loss": {"name": "losses@ACTLossHead", "loss_type": "stablemax_cross_entropy"},
            "halt_exploration_prob": args.halt_exploration_prob,
            "halt_max_steps": args.halt_max_steps,
            "H_cycles": args.H_cycles,
            "L_cycles": args.L_cycles,
            "H_layers": args.H_layers,
            "L_layers": args.L_layers,
            "hidden_size": args.hidden_size,
            "num_heads": args.num_heads,
            "expansion": args.expansion,
            "puzzle_emb_ndim": args.puzzle_emb_ndim,
            "pos_encodings": args.pos_encodings,
            "forward_dtype": args.forward_dtype,
            "mlp_t": args.mlp_t,
            "puzzle_emb_len": args.puzzle_emb_len,
            "no_ACT_continue": args.no_ACT_continue,
        },
        data_paths=[str(args.dataset)],
        data_paths_test=[],
        evaluators=[],
        global_batch_size=args.global_batch_size,
        epochs=epochs,
        eval_interval=epochs,
        checkpoint_every_eval=False,
        lr=args.lr,
        lr_min_ratio=args.lr_min_ratio,
        lr_warmup_steps=args.lr_warmup_steps,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        puzzle_emb_lr=args.puzzle_emb_lr,
        puzzle_emb_weight_decay=args.puzzle_emb_weight_decay,
        seed=args.seed,
        ema=args.ema,
        ema_rate=args.ema_rate,
        freeze_weights=False,
    )


def replace_dense_optimizer(train_state: Any, args: argparse.Namespace) -> None:
    previous_dense_optimizer = train_state.optimizers[-1]
    parameter_names = {id(param): name for name, param in train_state.model.named_parameters()}
    named_dense_params = [
        (parameter_names[id(param)], param)
        for group in previous_dense_optimizer.param_groups
        for param in group["params"]
        if param is not None
    ]
    dense_optimizer = create_dense_optimizer(named_dense_params, args)
    if len(train_state.optimizers) == 1:
        train_state.optimizers = [dense_optimizer]
        train_state.optimizer_lrs = [args.lr]
    else:
        train_state.optimizers = [train_state.optimizers[0], dense_optimizer]
        train_state.optimizer_lrs = [args.puzzle_emb_lr, args.lr]


def patch_ivon_noise_scale(dense_optimizer: Any, scale: float) -> None:
    if scale < 0:
        raise ValueError("--ivon-noise-scale must be non-negative.")
    if optimizer_name(dense_optimizer) != "ivon" or scale == 1.0:
        return

    def scaled_sample_params() -> tuple[torch.Tensor, torch.Tensor]:
        noise_samples = []
        param_avgs = []
        offset = 0
        for group in dense_optimizer.param_groups:
            gnumel = group["numel"]
            noise_sample = (
                torch.randn(gnumel, device=dense_optimizer._device, dtype=dense_optimizer._dtype)
                / (group["ess"] * (group["hess"] + group["weight_decay"])).sqrt()
            ) * scale
            noise_samples.append(noise_sample)
            goffset = 0
            for param in group["params"]:
                if param is None:
                    continue
                param_avg = param.data.flatten()
                numel = param.numel()
                param_noise = noise_sample[goffset : goffset + numel]
                param_avgs.append(param_avg)
                param.data = (param_avg + param_noise).view(param.shape)
                goffset += numel
                offset += numel
            assert goffset == group["numel"]
        assert offset == dense_optimizer._numel
        return torch.cat(param_avgs, 0), torch.cat(noise_samples, 0)

    dense_optimizer._sample_params = scaled_sample_params


def mode_id(value: str) -> float:
    table = {
        "normal": 1.0,
        "fixed_depth": 2.0,
        "delayed_fixed_depth": 3.0,
        "full": 1.0,
        "token_only": 2.0,
        "delayed_token_only": 3.0,
    }
    return table.get(value, 0.0)


def effective_modes(args: argparse.Namespace, step: int) -> tuple[str, str]:
    act_mode = args.act_mode
    loss_mode = args.loss_mode
    if args.act_mode == "delayed_fixed_depth":
        act_mode = "fixed_depth" if step <= args.delay_act_steps else "normal"
    if args.loss_mode == "delayed_token_only":
        loss_mode = "token_only" if step <= args.delay_act_steps else "full"
    return act_mode, loss_mode


def generalized_loss(
    loss_head: Any,
    carry: Any,
    batch: dict[str, torch.Tensor],
    *,
    loss_mode: str,
    q_loss_weight: float,
    return_keys: list[str],
) -> tuple[Any, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    new_carry, outputs = loss_head.model(carry=carry, batch=batch)
    labels = new_carry.current_data["labels"]

    with torch.no_grad():
        outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)
        mask = labels != -100
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
        is_correct = mask & (outputs["preds"] == labels)
        seq_is_correct = is_correct.sum(-1) == loss_counts
        valid_metrics = new_carry.halted & (loss_counts > 0)
        metrics = {
            "count": valid_metrics.sum(),
            "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
            "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
            "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
            "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
        }

    lm_loss = (loss_head.loss_fn(outputs["logits"], labels, ignore_index=-100, valid_mask=mask) / loss_divisor).sum()
    q_halt_loss = F.binary_cross_entropy_with_logits(
        outputs["q_halt_logits"],
        seq_is_correct.to(outputs["q_halt_logits"].dtype),
        reduction="sum",
    )
    q_continue_loss = torch.zeros((), device=lm_loss.device, dtype=lm_loss.dtype)
    if "target_q_continue" in outputs:
        q_continue_loss = F.binary_cross_entropy_with_logits(
            outputs["q_continue_logits"],
            outputs["target_q_continue"],
            reduction="sum",
        )
    metrics["lm_loss"] = lm_loss.detach()
    metrics["q_halt_loss"] = q_halt_loss.detach()
    if "target_q_continue" in outputs:
        metrics["q_continue_loss"] = q_continue_loss.detach()

    if loss_mode == "token_only":
        zero_q_anchor = outputs["q_halt_logits"].sum() * 0.0
        if "q_continue_logits" in outputs:
            zero_q_anchor = zero_q_anchor + outputs["q_continue_logits"].sum() * 0.0
        loss = lm_loss + zero_q_anchor
    else:
        loss = lm_loss + float(q_loss_weight) * (q_halt_loss + q_continue_loss)

    detached_outputs = {key: outputs[key].detach() for key in return_keys if key in outputs}
    return new_carry, loss, metrics, detached_outputs, new_carry.halted.all()


def forward_loss(
    train_state: Any,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[Any, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, str, str]:
    act_mode, loss_mode = effective_modes(args, train_state.step)
    loss_head = train_state.model
    was_training = loss_head.training
    act_module = loss_head.model
    was_act_training = act_module.training
    loss_head.train()
    if act_mode == "fixed_depth":
        act_module.training = False
    try:
        result = generalized_loss(
            loss_head,
            train_state.carry,
            batch,
            loss_mode=loss_mode,
            q_loss_weight=args.q_loss_weight,
            return_keys=["logits", "preds", "q_halt_logits", "q_continue_logits"],
        )
    finally:
        act_module.training = was_act_training
        loss_head.train(was_training)
    return (*result, act_mode, loss_mode)


def train_optimizer_batch(
    config: Any,
    train_state: Any,
    batch: Any,
    global_batch_size: int,
    args: argparse.Namespace,
    schedule: dict[str, Any],
) -> dict[str, float] | None:
    from pretrain import compute_lr

    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return None

    batch = {key: value.cuda() for key, value in batch.items()}
    if args.carry_mode == "reset_every_step":
        train_state.carry = None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)

    dense_optimizer = train_state.optimizers[-1]
    sparse_optimizer = train_state.optimizers[0] if len(train_state.optimizers) > 1 else None
    if sparse_optimizer is not None:
        sparse_default = compute_lr(args.puzzle_emb_lr, config, train_state)
        sparse_lr = float(schedule_value(schedule, "puzzle_emb_lr", train_state.step, sparse_default))
        for param_group in sparse_optimizer.param_groups:
            param_group["lr"] = sparse_lr
    else:
        sparse_lr = None

    scheduled_values = apply_optimizer_schedule(
        dense_optimizer, schedule, train_state.step, args
    )
    dense_optimizer_name = optimizer_name(dense_optimizer)
    metrics_out: dict[str, float] | None = None

    def closure() -> Any:
        nonlocal metrics_out
        dense_optimizer.zero_grad()
        if sparse_optimizer is not None:
            sparse_optimizer.zero_grad()
        with optimizer_sampling_context(dense_optimizer, train=True):
            train_state.carry, loss, raw_metrics, _outputs, _halted, act_mode, loss_mode = forward_loss(
                train_state,
                batch,
                args,
            )
            ((1 / global_batch_size) * loss).backward()
            if raw_metrics:
                count = max(float(raw_metrics["count"].detach().cpu()), 1.0)
                metrics_out = {
                    f"train/{key}": float(value.detach().cpu()) / (global_batch_size if key.endswith("loss") else count)
                    for key, value in raw_metrics.items()
                }
                metrics_out["train/lr"] = float(scheduled_values["dense_lr"])
                if sparse_lr is not None:
                    metrics_out["train/puzzle_emb_lr"] = sparse_lr
                metrics_out["diag/optimizer_id"] = {
                    "ivon": 1.0,
                    "soap": 2.0,
                    "evon": 3.0,
                }[dense_optimizer_name]
                if dense_optimizer_name == "ivon":
                    metrics_out["train/ivon_ess"] = float(
                        scheduled_values["ivon_ess"]
                    )
                    metrics_out["train/ivon_hess_approx_id"] = (
                        1.0
                        if scheduled_values["ivon_hess_approx"] == "price"
                        else 2.0
                    )
                elif dense_optimizer_name == "evon":
                    metrics_out["train/evon_ess"] = float(
                        scheduled_values["evon_ess"]
                    )
                metrics_out["diag/carry_mode_id"] = 1.0 if args.carry_mode == "persistent" else 2.0
                if dense_optimizer_name == "ivon":
                    metrics_out["diag/ivon_noise_scale"] = float(
                        args.ivon_noise_scale
                    )
                    metrics_out["diag/ivon_update_transform_id"] = {
                        "clip": 1.0,
                        "none": 2.0,
                        "muon_whiten": 3.0,
                    }[args.ivon_update_transform]
                    metrics_out["diag/ivon_muon_ns_steps"] = float(
                        args.ivon_muon_ns_steps
                    )
                metrics_out["diag/requested_act_mode_id"] = mode_id(args.act_mode)
                metrics_out["diag/effective_act_mode_id"] = mode_id(act_mode)
                metrics_out["diag/requested_loss_mode_id"] = mode_id(args.loss_mode)
                metrics_out["diag/effective_loss_mode_id"] = mode_id(loss_mode)
                metrics_out["diag/delay_act_steps"] = float(args.delay_act_steps)
                metrics_out["diag/q_loss_weight"] = float(args.q_loss_weight if loss_mode == "full" else 0.0)
            return (loss / global_batch_size).detach()

    dense_optimizer.step(closure)
    dense_optimizer.zero_grad()
    if sparse_optimizer is not None:
        sparse_optimizer.step()
        sparse_optimizer.zero_grad()
    return metrics_out


def train_ivon_batch(
    config: Any,
    train_state: Any,
    batch: Any,
    global_batch_size: int,
    args: argparse.Namespace,
    schedule: dict[str, Any],
) -> dict[str, float] | None:
    """Backward-compatible alias for older launch code."""
    return train_optimizer_batch(
        config, train_state, batch, global_batch_size, args, schedule
    )


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "torch_cuda_active_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any], device: torch.device, source_device: str | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda_all") or []
    if not cuda_states:
        return

    cuda_states = [cuda_state.cpu() for cuda_state in cuda_states]
    source_index = state.get("torch_cuda_active_device")
    if source_index is None and source_device is not None:
        parsed_source = torch.device(source_device)
        if parsed_source.type == "cuda":
            source_index = 0 if parsed_source.index is None else parsed_source.index
    if source_index is None:
        source_index = 0 if device.index is None else device.index
    source_index = int(source_index)
    if not 0 <= source_index < len(cuda_states):
        raise ValueError(
            f"Checkpoint CUDA RNG state does not contain source device index {source_index}; "
            f"available states: {len(cuda_states)}."
        )
    torch.cuda.set_rng_state(cuda_states[source_index], device=device)


def comparable_resume_args(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in RESUME_ARG_IGNORED_KEYS}


def exact_resume_args(args: argparse.Namespace) -> dict[str, Any]:
    return comparable_resume_args(serializable_args(args))


def validate_resume_metadata(checkpoint: dict[str, Any], args: argparse.Namespace, schedule: dict[str, Any]) -> None:
    format_version = int(checkpoint.get("exact_format_version", -1))
    if format_version not in (3, EXACT_FORMAT_VERSION):
        raise ValueError(f"Unsupported exact checkpoint format: {checkpoint.get('exact_format_version')}")
    checkpoint_args = checkpoint.get("args_for_resume")
    if not isinstance(checkpoint_args, dict):
        raise ValueError("Exact checkpoint is missing args_for_resume metadata.")
    checkpoint_args = dict(checkpoint_args)
    checkpoint_args.setdefault("train_group_count_for_epochs", 1000)
    current_args = exact_resume_args(args)
    if format_version == 3:
        if getattr(args, "optimizer", "ivon") != "ivon":
            raise ValueError("Version 3 exact checkpoints can only resume with IVON.")
        checkpoint_args.setdefault("optimizer", "ivon")
        for key, value in current_args.items():
            if key.startswith(("soap_", "evon_")):
                checkpoint_args.setdefault(key, value)
    if comparable_resume_args(checkpoint_args) != current_args:
        raise ValueError("Resume args do not match exact checkpoint args_for_resume.")
    if checkpoint.get("schedule") != schedule:
        raise ValueError("Resume schedule does not match exact checkpoint schedule.")


def save_public_checkpoint(train_state: Any, args: argparse.Namespace, run_root: Path, name: str, ema_helper: Any | None) -> Path:
    path = run_root / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    dense_optimizer = train_state.optimizers[-1]
    sparse_optimizer = train_state.optimizers[0] if len(train_state.optimizers) > 1 else None
    dense_optimizer_state = dense_optimizer.state_dict()
    payload = {
        "name": "ptrm_optimizer_public",
        "args": serializable_args(args),
        "step": train_state.step,
        "model_state_dict": train_state.model.state_dict(),
        "optimizer_name": optimizer_name(dense_optimizer),
        "optimizer_state_dict": dense_optimizer_state,
        "dense_optimizer_state_dict": dense_optimizer_state,
        "dense_optimizer_current_step": getattr(dense_optimizer, "current_step", None),
        "sparse_optimizer_state_dict": sparse_optimizer.state_dict() if sparse_optimizer is not None else None,
    }
    if ema_helper is not None:
        ema_model = ema_helper.ema_copy(train_state.model)
        payload["ema_model_state_dict"] = ema_model.state_dict()
        payload["ema_shadow_state_dict"] = ema_helper.state_dict()
        del ema_model
    torch.save(payload, path)
    return path


def save_exact_checkpoint(
    train_state: Any,
    batch_stream: ExactTrainBatchStream,
    args: argparse.Namespace,
    schedule: dict[str, Any],
    run_root: Path,
    name: str,
    ema_helper: Any | None,
    last_metrics: dict[str, Any] | None,
) -> Path:
    path = run_root / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    dense_optimizer = train_state.optimizers[-1]
    sparse_optimizer = train_state.optimizers[0] if len(train_state.optimizers) > 1 else None
    dense_optimizer_state = dense_optimizer.state_dict()
    payload = {
        "name": "ptrm_optimizer_exact",
        "exact_format_version": EXACT_FORMAT_VERSION,
        "args": serializable_args(args),
        "args_for_resume": exact_resume_args(args),
        "schedule": schedule,
        "step": train_state.step,
        "total_steps": train_state.total_steps,
        "model_state_dict": train_state.model.state_dict(),
        "optimizer_name": optimizer_name(dense_optimizer),
        "optimizer_state_dict": dense_optimizer_state,
        "dense_optimizer_state_dict": dense_optimizer_state,
        "dense_optimizer_current_step": getattr(dense_optimizer, "current_step", None),
        "sparse_optimizer_state_dict": sparse_optimizer.state_dict() if sparse_optimizer is not None else None,
        "carry": train_state.carry,
        "batch_stream_state": batch_stream.state_dict(),
        "rng_state": capture_rng_state(),
        "last_metrics": last_metrics,
    }
    if ema_helper is not None:
        ema_model = ema_helper.ema_copy(train_state.model)
        payload["ema_model_state_dict"] = ema_model.state_dict()
        payload["ema_shadow_state_dict"] = ema_helper.state_dict()
        del ema_model
    torch.save(payload, path)
    return path


def restore_exact_checkpoint(
    train_state: Any,
    batch_stream: ExactTrainBatchStream,
    ema_helper: Any | None,
    checkpoint_path: Path,
    args: argparse.Namespace,
    schedule: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_resume_metadata(checkpoint, args, schedule)
    train_state.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dense_optimizer = train_state.optimizers[-1]
    saved_optimizer_name = checkpoint.get(
        "optimizer_name", (checkpoint.get("args") or {}).get("optimizer", "ivon")
    )
    if saved_optimizer_name != optimizer_name(dense_optimizer):
        raise ValueError(
            f"Checkpoint optimizer {saved_optimizer_name!r} does not match "
            f"requested optimizer {optimizer_name(dense_optimizer)!r}."
        )
    optimizer_state = checkpoint.get(
        "optimizer_state_dict", checkpoint.get("dense_optimizer_state_dict")
    )
    if not isinstance(optimizer_state, dict):
        raise ValueError("Exact checkpoint is missing optimizer state.")
    dense_optimizer.load_state_dict(optimizer_state)
    move_optimizer_state_to_device(dense_optimizer, device)
    current_step = checkpoint.get("dense_optimizer_current_step")
    if current_step is not None and hasattr(dense_optimizer, "current_step"):
        dense_optimizer.current_step = int(current_step)
    sparse_state = checkpoint.get("sparse_optimizer_state_dict")
    if sparse_state is not None and len(train_state.optimizers) > 1:
        train_state.optimizers[0].load_state_dict(sparse_state)
    train_state.step = int(checkpoint["step"])
    train_state.total_steps = int(checkpoint["total_steps"])
    train_state.carry = checkpoint["carry"]
    batch_stream.load_state_dict(checkpoint["batch_stream_state"])
    if ema_helper is not None:
        ema_state = checkpoint.get("ema_shadow_state_dict")
        if ema_state is None:
            raise ValueError("Resume requested EMA, but checkpoint has no EMA state.")
        ema_helper.load_state_dict(ema_state)
    elif checkpoint.get("ema_shadow_state_dict") is not None:
        raise ValueError("Checkpoint has EMA state, but --ema was not provided.")
    checkpoint_args = checkpoint.get("args")
    source_device = checkpoint_args.get("device") if isinstance(checkpoint_args, dict) else None
    restore_rng_state(checkpoint["rng_state"], device, source_device=source_device)
    return checkpoint


def load_model_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device, state_key: str = "model_state_dict") -> None:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload[state_key] if isinstance(payload, dict) and state_key in payload else payload
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint does not contain a state dict: {checkpoint_path}")
    for prefix in ("_orig_mod.", "module."):
        if state_dict and all(str(key).startswith(prefix) for key in state_dict):
            state_dict = {str(key)[len(prefix) :]: value for key, value in state_dict.items()}
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Could not load model weights from {checkpoint_path}. "
            f"Missing keys: {result.missing_keys}. Unexpected keys: {result.unexpected_keys}."
        )


def init_wandb(args: argparse.Namespace, run_root: Path, schedule: dict[str, Any], job_type: str) -> Any | None:
    if not args.wandb:
        return None
    import wandb

    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        id=args.wandb_run_id or args.run_name,
        resume="allow",
        name=f"{args.wandb_group}/{args.run_name}" if args.wandb_group else args.run_name,
        group=args.wandb_group,
        job_type=job_type,
        mode=args.wandb_mode,
        config={"args": serializable_args(args), "schedule": schedule, "artifact_root": str(run_root)},
    )
