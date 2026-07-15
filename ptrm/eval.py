#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optim.factory import (
    create_optimizer,
    move_optimizer_state_to_device,
    optimizer_name,
    optimizer_sampling_context,
    supports_posterior_sampling,
)
from optim.ivon import IVON
from ptrm import utils


Method = Literal[
    "mean_only",
    "posterior_parameter_sampling",
    "antithetic_parameter_sampling",
    "posterior_mean_q_selection",
    "compare_selection",
]
METHOD_ALIASES = {
    "ivon_parameter_sampling": "posterior_parameter_sampling",
    "ivon_antithetic_parameter_sampling": "antithetic_parameter_sampling",
    "ivon_posterior_mean_q_selection": "posterior_mean_q_selection",
    "ivon_compare_selection": "compare_selection",
}
METHOD_CHOICES = (
    "mean_only",
    "posterior_parameter_sampling",
    "antithetic_parameter_sampling",
    "posterior_mean_q_selection",
    "compare_selection",
    *METHOD_ALIASES.keys(),
)
IGNORE_LABEL_ID = -100


@dataclass
class EvalTotals:
    count: int = 0
    token_count: int = 0
    direct_exact: int = 0
    direct_token_correct: int = 0
    best_q_exact: int = 0
    best_q_token_correct: int = 0
    pass_at_k_exact: int = 0
    posterior_mean_q_exact: int = 0
    posterior_mean_q_token_correct: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PTRM/TRM optimizer checkpoints.")
    parser.add_argument("--trm-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", "--ivon-checkpoint", dest="checkpoint", type=Path, required=True
    )
    parser.add_argument("--model-state-key", choices=("model_state_dict", "ema_model_state_dict"), default="model_state_dict")
    parser.add_argument(
        "--method",
        choices=METHOD_CHOICES,
        default="posterior_parameter_sampling",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument(
        "--posterior-scale",
        "--ivon-posterior-scale",
        dest="posterior_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit-puzzles", type=int, default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def create_eval_loader(dataset_path: Path, seed: int, eval_batch_size: int) -> tuple[DataLoader, Any]:
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=seed,
            dataset_paths=[str(dataset_path)],
            global_batch_size=eval_batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split="test",
    )
    return DataLoader(dataset, batch_size=None, num_workers=0), dataset.metadata


def create_sharded_eval_loader(
    *,
    dataset_path: Path,
    seed: int,
    eval_batch_size: int,
    num_shards: int,
    shard_index: int,
) -> tuple[Iterable[tuple[str, dict[str, torch.Tensor], int]], Any, int]:
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=seed,
            dataset_paths=[str(dataset_path)],
            global_batch_size=eval_batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split="test",
    )
    dataset._lazy_load_dataset()
    assert dataset._data is not None
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")

    set_names = list(dataset._data.keys())
    set_offsets = np.cumsum([0] + [len(dataset._data[set_name]["inputs"]) for set_name in set_names])
    total_examples = int(set_offsets[-1])
    shard_start = total_examples * shard_index // num_shards
    shard_end = total_examples * (shard_index + 1) // num_shards
    shard_size = shard_end - shard_start

    def iter_shard_batches() -> Iterable[tuple[str, dict[str, torch.Tensor], int]]:
        for start in range(shard_start, shard_end, eval_batch_size):
            stop = min(start + eval_batch_size, shard_end)
            global_indices = np.arange(start, stop, dtype=np.int64)
            inputs = []
            labels = []
            puzzle_identifiers = []
            for set_i, set_name in enumerate(set_names):
                set_start = set_offsets[set_i]
                set_end = set_offsets[set_i + 1]
                in_set = global_indices[(global_indices >= set_start) & (global_indices < set_end)]
                if in_set.size == 0:
                    continue
                local_indices = in_set - set_start
                data = dataset._data[set_name]
                puzzle_ids = np.searchsorted(data["puzzle_indices"], local_indices, side="right") - 1
                inputs.append(data["inputs"][local_indices])
                labels.append(data["labels"][local_indices])
                puzzle_identifiers.append(data["puzzle_identifiers"][puzzle_ids])
            raw_count = int(global_indices.size)
            batch = dataset._collate_batch(
                {
                    "inputs": np.concatenate(inputs, axis=0),
                    "labels": np.concatenate(labels, axis=0),
                    "puzzle_identifiers": np.concatenate(puzzle_identifiers, axis=0),
                }
            )
            yield f"shard-{shard_index}", batch, raw_count

    return iter_shard_batches(), dataset.metadata, shard_size


def checkpoint_model_config(payload_args: dict[str, Any], metadata: Any, batch_size: int) -> dict[str, Any]:
    return {
        "batch_size": batch_size,
        "vocab_size": metadata.vocab_size,
        "seq_len": metadata.seq_len,
        "num_puzzle_identifiers": metadata.num_puzzle_identifiers,
        "causal": False,
        "halt_exploration_prob": float(payload_args.get("halt_exploration_prob", 0.1)),
        "halt_max_steps": int(payload_args.get("halt_max_steps", 16)),
        "H_cycles": int(payload_args.get("H_cycles", 3)),
        "L_cycles": int(payload_args.get("L_cycles", 6)),
        "H_layers": int(payload_args.get("H_layers", 0)),
        "L_layers": int(payload_args.get("L_layers", 2)),
        "hidden_size": int(payload_args.get("hidden_size", 512)),
        "num_heads": int(payload_args.get("num_heads", 8)),
        "expansion": float(payload_args.get("expansion", 4.0)),
        "puzzle_emb_ndim": int(payload_args.get("puzzle_emb_ndim", 512)),
        "pos_encodings": str(payload_args.get("pos_encodings", "none")),
        "forward_dtype": str(payload_args.get("forward_dtype", "bfloat16")),
        "mlp_t": bool(payload_args.get("mlp_t", True)),
        "puzzle_emb_len": int(payload_args.get("puzzle_emb_len", 16)),
        "no_ACT_continue": bool(payload_args.get("no_ACT_continue", True)),
    }


def create_wrapped_model_from_checkpoint(payload: dict[str, Any], metadata: Any, batch_size: int, device: torch.device) -> torch.nn.Module:
    from models.losses import ACTLossHead
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

    payload_args = payload.get("args") or {}
    with torch.device(device):
        model = TinyRecursiveReasoningModel_ACTV1(checkpoint_model_config(payload_args, metadata, batch_size))
        model = ACTLossHead(model, loss_type="stablemax_cross_entropy")
    model.eval()
    return model


def dense_optimizer_parameters_from_checkpoint(
    model: torch.nn.Module,
    optimizer_state: dict[str, Any],
) -> list[dict[str, Any]]:
    saved_groups = optimizer_state.get("param_groups")
    if not isinstance(saved_groups, list) or not saved_groups:
        named_parameters = list(model.named_parameters())
        return [
            {
                "params": [parameter for _, parameter in named_parameters],
                "param_names": [name for name, _ in named_parameters],
            }
        ]

    parameters_by_name = dict(model.named_parameters())
    parameter_groups: list[dict[str, Any]] = []
    for saved_group in saved_groups:
        names = saved_group.get("param_names") if isinstance(saved_group, dict) else None
        saved_params = saved_group.get("params") if isinstance(saved_group, dict) else None
        if not isinstance(names, list) or not isinstance(saved_params, list) or len(names) != len(saved_params):
            raise RuntimeError(
                "Checkpoint optimizer groups require param_names for safe reconstruction."
            )
        resolved_parameters = []
        for saved_name in names:
            normalized = str(saved_name)
            for prefix in ("_orig_mod.", "module."):
                normalized = normalized.removeprefix(prefix)
            if normalized not in parameters_by_name:
                raise RuntimeError(f"Checkpoint dense optimizer parameter is missing from the model: {saved_name}")
            resolved_parameters.append(parameters_by_name[normalized])
        parameter_groups.append(
            {"params": resolved_parameters, "param_names": list(names)}
        )
    return parameter_groups


def create_optimizer_from_checkpoint(
    model: torch.nn.Module,
    optimizer_kind: str,
    payload_args: dict[str, Any],
    optimizer_state: dict[str, Any],
) -> Optimizer:
    return create_optimizer(
        optimizer_kind,
        dense_optimizer_parameters_from_checkpoint(model, optimizer_state),
        payload_args,
    )


def normalize_model_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    for prefix in ("_orig_mod.", "module."):
        if state_dict and all(str(key).startswith(prefix) for key in state_dict):
            return {str(key)[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def load_optimizer_state(
    model: torch.nn.Module,
    payload: dict[str, Any],
    checkpoint_path: Path,
    model_state_key: str,
) -> Optimizer:
    if model_state_key not in payload:
        available = sorted(payload.keys())
        raise RuntimeError(f"Expected checkpoint payload with {model_state_key}. Available keys: {available}")
    state_dict = normalize_model_state_dict_keys(payload[model_state_key])
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Could not load model state {checkpoint_path}. "
            f"Missing keys: {result.missing_keys}. Unexpected keys: {result.unexpected_keys}."
        )
    optimizer_state = payload.get(
        "optimizer_state_dict", payload.get("dense_optimizer_state_dict")
    )
    if not isinstance(optimizer_state, dict):
        raise RuntimeError("Checkpoint does not contain optimizer state.")
    payload_args = payload.get("args") or {}
    if not isinstance(payload_args, dict):
        raise RuntimeError("Checkpoint args must be a mapping.")
    optimizer_kind = str(
        payload.get("optimizer_name", payload_args.get("optimizer", "ivon"))
    ).lower()
    optimizer = create_optimizer_from_checkpoint(
        model, optimizer_kind, payload_args, optimizer_state
    )
    optimizer.load_state_dict(optimizer_state)
    move_optimizer_state_to_device(optimizer, next(model.parameters()).device)
    current_step = payload.get("dense_optimizer_current_step")
    if current_step is not None and hasattr(optimizer, "current_step"):
        optimizer.current_step = int(current_step)
    return optimizer


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def trim_batch(batch: dict[str, torch.Tensor], limit: int | None) -> dict[str, torch.Tensor]:
    if limit is None or batch["inputs"].shape[0] <= limit:
        return batch
    return {key: value[:limit] for key, value in batch.items()}


def run_rollout_current_params(
    *,
    inner: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    depth: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = batch["inputs"].shape[0]
    with torch.device(device):
        inner_carry = inner.empty_carry(batch_size)
    reset_flag = torch.ones(batch_size, dtype=torch.bool, device=device)
    inner_carry = inner.reset_carry(reset_flag, inner_carry)

    logits = None
    q_halt_logits = None
    with torch.inference_mode():
        for _ in range(depth):
            inner_carry, logits, (q_halt_logits, _q_continue_logits) = inner(inner_carry, batch)
    assert logits is not None
    assert q_halt_logits is not None
    return torch.argmax(logits, dim=-1), q_halt_logits.to(torch.float32)


def run_optimizer_rollout_once(
    *,
    inner: torch.nn.Module,
    optimizer: Optimizer,
    batch: dict[str, torch.Tensor],
    depth: int,
    device: torch.device,
    posterior_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with optimizer_sampling_context(
        optimizer, train=False, posterior_scale=posterior_scale
    ):
        return run_rollout_current_params(inner=inner, batch=batch, depth=depth, device=device)


def assign_ivon_noise_sample(
    optimizer: IVON,
    *,
    param_avg: torch.Tensor,
    noise: torch.Tensor,
    posterior_scale: float,
    sign: float,
) -> None:
    offset = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p is None:
                continue
            p_slice = slice(offset, offset + p.numel())
            p.data = (param_avg[p_slice] + sign * posterior_scale * noise[p_slice]).view(p.shape)
            offset += p.numel()
    assert offset == optimizer._numel


def run_ivon_antithetic_rollout_pair(
    *,
    inner: torch.nn.Module,
    optimizer: IVON,
    batch: dict[str, torch.Tensor],
    depth: int,
    device: torch.device,
    posterior_scale: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    param_avg, noise = optimizer._sample_params()
    try:
        outputs = []
        for sign in (1.0, -1.0):
            assign_ivon_noise_sample(optimizer, param_avg=param_avg, noise=noise, posterior_scale=posterior_scale, sign=sign)
            outputs.append(run_rollout_current_params(inner=inner, batch=batch, depth=depth, device=device))
        return outputs
    finally:
        optimizer._restore_param_average(False, param_avg, noise)


def exact_and_token_correct(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded_labels = labels.unsqueeze(1)
    expanded_mask = mask.unsqueeze(1)
    token_correct = ((preds == expanded_labels) & expanded_mask).sum(dim=-1)
    exact = ((preds == expanded_labels) | ~expanded_mask).all(dim=-1) & valid.unsqueeze(1)
    return exact, token_correct


def posterior_mean_q_indices(
    *,
    preds: torch.Tensor,
    q_scores: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    selected = torch.zeros((preds.shape[0],), dtype=torch.long, device=preds.device)
    preds_cpu = preds.detach().cpu()
    q_cpu = q_scores.detach().cpu()
    mask_cpu = mask.detach().cpu()
    valid_cpu = valid.detach().cpu()
    for row in range(preds_cpu.shape[0]):
        if not bool(valid_cpu[row]):
            continue
        active_mask = mask_cpu[row].bool()
        groups: dict[tuple[int, ...], list[tuple[int, float]]] = {}
        for candidate_id in range(preds_cpu.shape[1]):
            answer = tuple(int(x) for x in preds_cpu[row, candidate_id][active_mask].tolist())
            groups.setdefault(answer, []).append((candidate_id, float(q_cpu[row, candidate_id])))

        def group_key(item: tuple[tuple[int, ...], list[tuple[int, float]]]) -> tuple[float, float]:
            _answer, members = item
            scores = [score for _candidate_id, score in members]
            return (sum(scores) / len(scores), max(scores))

        _answer, best_members = max(groups.items(), key=group_key)
        best_candidate_id, _best_q = max(best_members, key=lambda x: x[1])
        selected[row] = best_candidate_id
    return selected


def write_progress_json(
    progress_json: Path | None,
    *,
    args: argparse.Namespace,
    totals: EvalTotals,
    start_time: float,
    expected_count: int | None,
) -> None:
    if progress_json is None:
        return
    elapsed_s = time.time() - start_time
    count = max(totals.count, 1)
    token_count = max(totals.token_count, 1)
    has_posterior_mean_selection = args.method in (
        "posterior_mean_q_selection",
        "compare_selection",
    )
    selected_exact = (
        totals.posterior_mean_q_exact
        if args.method == "posterior_mean_q_selection"
        else totals.best_q_exact
    )
    selected_token_correct = (
        totals.posterior_mean_q_token_correct
        if args.method == "posterior_mean_q_selection"
        else totals.best_q_token_correct
    )
    utils.write_json(
        progress_json,
        {
            "method": args.method,
            "count": totals.count,
            "expected_count": expected_count,
            "elapsed_s": elapsed_s,
            "puzzles_per_s": totals.count / elapsed_s if elapsed_s > 0 else None,
            "direct_exact_accuracy": totals.direct_exact / count,
            "selected_exact_accuracy": selected_exact / count,
            "best_q_exact_accuracy": totals.best_q_exact / count,
            "pass_at_k_exact_accuracy": totals.pass_at_k_exact / count,
            "direct_token_accuracy": totals.direct_token_correct / token_count,
            "selected_token_accuracy": selected_token_correct / token_count,
            "best_q_token_accuracy": totals.best_q_token_correct / token_count,
            "posterior_mean_q_exact_accuracy": (
                totals.posterior_mean_q_exact / count if has_posterior_mean_selection else None
            ),
            "posterior_mean_q_token_accuracy": (
                totals.posterior_mean_q_token_correct / token_count if has_posterior_mean_selection else None
            ),
            "updated_at": time.time(),
        },
    )


def evaluate(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: Optimizer,
    loader: Iterable[tuple[str, dict[str, torch.Tensor], int]],
    device: torch.device,
    expected_count: int | None,
) -> tuple[EvalTotals, float]:
    inner = model.model.inner
    totals = EvalTotals()
    start_time = time.time()
    iterator = tqdm(loader, desc=f"{args.method}:shard{args.shard_index}", disable=not args.progress)
    for _set_name, cpu_batch, _global_batch_size in iterator:
        remaining = None if args.limit_puzzles is None else args.limit_puzzles - totals.count
        if remaining is not None and remaining <= 0:
            break
        batch = move_batch(trim_batch(cpu_batch, remaining), device)
        labels = batch["labels"]
        mask = labels != IGNORE_LABEL_ID
        valid = mask.sum(dim=-1) > 0
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            continue

        batch_size = labels.shape[0]
        best_q = torch.full((batch_size,), -math.inf, dtype=torch.float32, device=device)
        best_exact = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        best_token_correct = torch.zeros((batch_size,), dtype=torch.long, device=device)
        pass_exact = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        direct_exact = None
        direct_token_correct = None
        posterior_exact = None
        posterior_token_correct = None
        all_preds = []
        all_q_scores = []

        sample_id = 0
        sample_count = 1 if args.method == "mean_only" else args.k
        posterior_scale = (
            0.0 if args.method == "mean_only" else args.posterior_scale
        )
        while sample_id < sample_count:
            if args.method == "antithetic_parameter_sampling":
                assert isinstance(optimizer, IVON)
                rollout_outputs = run_ivon_antithetic_rollout_pair(
                    inner=inner,
                    optimizer=optimizer,
                    batch=batch,
                    depth=args.depth,
                    device=device,
                    posterior_scale=posterior_scale,
                )
            else:
                rollout_outputs = [
                    run_optimizer_rollout_once(
                        inner=inner,
                        optimizer=optimizer,
                        batch=batch,
                        depth=args.depth,
                        device=device,
                        posterior_scale=posterior_scale,
                    )
                ]
            for pred, q_score in rollout_outputs:
                if sample_id >= sample_count:
                    break
                preds = pred.unsqueeze(1)
                q_scores = q_score.unsqueeze(1)
                exact, token_correct = exact_and_token_correct(preds, labels, mask, valid)
                exact = exact[:, 0]
                token_correct = token_correct[:, 0]
                if sample_id == 0:
                    direct_exact = exact
                    direct_token_correct = token_correct
                better = q_score > best_q
                best_q = torch.where(better, q_score, best_q)
                best_exact = torch.where(better, exact, best_exact)
                best_token_correct = torch.where(better, token_correct, best_token_correct)
                pass_exact |= exact
                if args.method in (
                    "posterior_mean_q_selection",
                    "compare_selection",
                ):
                    all_preds.append(pred)
                    all_q_scores.append(q_score)
                sample_id += 1

        if all_preds:
            preds_stack = torch.stack(all_preds, dim=1)
            q_stack = torch.stack(all_q_scores, dim=1)
            selected = posterior_mean_q_indices(preds=preds_stack, q_scores=q_stack, mask=mask, valid=valid)
            posterior_exact_all, posterior_token_all = exact_and_token_correct(preds_stack, labels, mask, valid)
            rows = torch.arange(batch_size, device=device)
            posterior_exact = posterior_exact_all[rows, selected]
            posterior_token_correct = posterior_token_all[rows, selected]

        assert direct_exact is not None
        assert direct_token_correct is not None
        token_count = int(mask.sum().item())
        totals.count += valid_count
        totals.token_count += token_count
        totals.direct_exact += int((direct_exact & valid).sum().item())
        totals.direct_token_correct += int((direct_token_correct * valid.to(direct_token_correct.dtype)).sum().item())
        totals.best_q_exact += int((best_exact & valid).sum().item())
        totals.best_q_token_correct += int((best_token_correct * valid.to(best_token_correct.dtype)).sum().item())
        totals.pass_at_k_exact += int((pass_exact & valid).sum().item())
        if posterior_exact is not None and posterior_token_correct is not None:
            totals.posterior_mean_q_exact += int((posterior_exact & valid).sum().item())
            totals.posterior_mean_q_token_correct += int(
                (posterior_token_correct * valid.to(posterior_token_correct.dtype)).sum().item()
            )
        write_progress_json(args.progress_json, args=args, totals=totals, start_time=start_time, expected_count=expected_count)

    return totals, time.time() - start_time


def summarize(totals: EvalTotals, elapsed_s: float, args: argparse.Namespace, expected_count: int | None) -> dict[str, Any]:
    count = max(totals.count, 1)
    token_count = max(totals.token_count, 1)
    has_posterior_mean_selection = args.method in (
        "posterior_mean_q_selection",
        "compare_selection",
    )
    selected_exact = (
        totals.posterior_mean_q_exact
        if args.method == "posterior_mean_q_selection"
        else totals.best_q_exact
    )
    selected_token_correct = (
        totals.posterior_mean_q_token_correct
        if args.method == "posterior_mean_q_selection"
        else totals.best_q_token_correct
    )
    return {
        "method": args.method,
        "count": totals.count,
        "expected_count": expected_count,
        "token_count": totals.token_count,
        "direct_exact_count": totals.direct_exact,
        "direct_token_correct": totals.direct_token_correct,
        "best_q_exact_count": totals.best_q_exact,
        "best_q_token_correct": totals.best_q_token_correct,
        "selected_exact_count": selected_exact,
        "selected_token_correct": selected_token_correct,
        "pass_at_k_exact_count": totals.pass_at_k_exact,
        "posterior_mean_q_exact_count": totals.posterior_mean_q_exact,
        "posterior_mean_q_token_correct": totals.posterior_mean_q_token_correct,
        "elapsed_s": elapsed_s,
        "config": {
            "checkpoint": str(args.checkpoint),
            "model_state_key": args.model_state_key,
            "k": args.k,
            "depth": args.depth,
            "limit_puzzles": args.limit_puzzles,
            "posterior_scale": args.posterior_scale,
        },
        "direct_exact_accuracy": totals.direct_exact / count,
        "direct_token_accuracy": totals.direct_token_correct / token_count,
        "selected_exact_accuracy": selected_exact / count,
        "selected_token_accuracy": selected_token_correct / token_count,
        "best_q_exact_accuracy": totals.best_q_exact / count,
        "best_q_token_accuracy": totals.best_q_token_correct / token_count,
        "pass_at_k_exact_accuracy": totals.pass_at_k_exact / count,
        "posterior_mean_q_exact_accuracy": (
            totals.posterior_mean_q_exact / count if has_posterior_mean_selection else None
        ),
        "posterior_mean_q_token_accuracy": (
            totals.posterior_mean_q_token_correct / token_count if has_posterior_mean_selection else None
        ),
    }


def main() -> None:
    args = parse_args()
    args.method = METHOD_ALIASES.get(args.method, args.method)
    if args.k <= 0 or args.depth <= 0:
        raise ValueError("--k and --depth must be positive.")
    if args.posterior_scale < 0:
        raise ValueError("--posterior-scale must be non-negative.")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")
    utils.import_reference_code(args.trm_repo)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)

    if args.num_shards == 1:
        loader, metadata = create_eval_loader(args.dataset, args.seed, args.eval_batch_size)
        expected_count = args.limit_puzzles
    else:
        loader, metadata, shard_size = create_sharded_eval_loader(
            dataset_path=args.dataset,
            seed=args.seed,
            eval_batch_size=args.eval_batch_size,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
        expected_count = min(shard_size, args.limit_puzzles) if args.limit_puzzles is not None else shard_size

    raw_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(raw_payload, dict):
        raise RuntimeError(f"Expected dict checkpoint payload: {args.checkpoint}")
    model = create_wrapped_model_from_checkpoint(raw_payload, metadata, args.eval_batch_size, device)
    optimizer = load_optimizer_state(
        model, raw_payload, args.checkpoint, args.model_state_key
    )
    loaded_optimizer_name = optimizer_name(optimizer)
    if args.method != "mean_only" and not supports_posterior_sampling(optimizer):
        raise ValueError(
            f"{loaded_optimizer_name.upper()} checkpoints only support --method mean_only."
        )
    if args.method == "antithetic_parameter_sampling" and not isinstance(
        optimizer, IVON
    ):
        raise ValueError("Antithetic parameter sampling is currently IVON-only.")
    totals, elapsed_s = evaluate(args=args, model=model, optimizer=optimizer, loader=loader, device=device, expected_count=expected_count)
    summary = summarize(totals, elapsed_s, args, expected_count)
    summary["optimizer"] = loaded_optimizer_name
    summary["checkpoint_args"] = raw_payload.get("args", {})
    print(json.dumps(summary, indent=2, sort_keys=True))
    utils.write_json(args.output_json, summary)


if __name__ == "__main__":
    main()
