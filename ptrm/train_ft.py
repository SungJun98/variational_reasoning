#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptrm import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a PTRM/TRM checkpoint with IVON.")
    parser.add_argument("--trm-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-state-key", default="model_state_dict")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", default="ptrm_ivon_ft")
    parser.add_argument("--schedule-json", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--stop-step", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--train-group-count-for-epochs", type=int, default=1000)
    parser.add_argument("--status-interval", type=int, default=50)
    parser.add_argument("--history-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--disable-compile", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min-ratio", type=float, default=1.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=100)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--puzzle-emb-lr", type=float, default=1e-2)
    parser.add_argument("--puzzle-emb-weight-decay", type=float, default=0.1)

    parser.add_argument("--ivon-ess", type=float, default=3e5)
    parser.add_argument("--ivon-hess-init", type=float, default=3.0)
    parser.add_argument("--ivon-beta2", type=float, default=0.99999)
    parser.add_argument("--ivon-weight-decay", type=float, default=0.225)
    parser.add_argument("--ivon-clip-radius", type=float, default=float("inf"))
    parser.add_argument("--ivon-hess-approx", choices=("price", "gradsq"), default="price")
    parser.add_argument("--ivon-mc-samples", type=int, default=1)
    parser.add_argument("--no-ivon-debias", action="store_true")
    parser.add_argument("--no-ivon-rescale-lr", action="store_true")
    parser.add_argument("--ivon-noise-scale", type=float, default=1.0)
    parser.add_argument("--ivon-update-transform", choices=("clip", "none", "muon_whiten"), default="clip")
    parser.add_argument("--ivon-muon-whiten-eps", type=float, default=1e-8)
    parser.add_argument("--ivon-muon-ns-steps", type=int, default=5)

    parser.add_argument("--carry-mode", choices=("persistent", "reset_every_step"), default="persistent")
    parser.add_argument("--h-cycles", dest="H_cycles", type=int, default=3)
    parser.add_argument("--l-cycles", dest="L_cycles", type=int, default=6)
    parser.add_argument("--h-layers", dest="H_layers", type=int, default=0)
    parser.add_argument("--l-layers", dest="L_layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--expansion", type=float, default=4.0)
    parser.add_argument("--puzzle-emb-ndim", type=int, default=512)
    parser.add_argument("--puzzle-emb-len", type=int, default=16)
    parser.add_argument("--pos-encodings", choices=("none", "rope", "learned"), default="none")
    parser.add_argument("--forward-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--mlp-t", dest="mlp_t", action="store_true", default=True)
    parser.add_argument("--no-mlp-t", dest="mlp_t", action="store_false")
    parser.add_argument("--halt-max-steps", type=int, default=16)
    parser.add_argument("--halt-exploration-prob", type=float, default=0.1)
    parser.add_argument("--no-act-continue", dest="no_ACT_continue", action="store_true", default=True)
    parser.add_argument("--act-continue", dest="no_ACT_continue", action="store_false")
    parser.add_argument("--act-mode", choices=("normal", "fixed_depth", "delayed_fixed_depth"), default="normal")
    parser.add_argument("--loss-mode", choices=("full", "token_only", "delayed_token_only"), default="full")
    parser.add_argument("--delay-act-steps", type=int, default=0)
    parser.add_argument("--q-loss-weight", type=float, default=0.35)

    parser.add_argument("--ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="ema", action="store_false")
    parser.add_argument("--ema-rate", type=float, default=0.999)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="ptrm-ivon-fine-tuning")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-mode", default="online")
    return parser.parse_args()


def prepare_run(args: argparse.Namespace, schedule: dict[str, Any]) -> tuple[Path, Path]:
    run_root = args.output_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.md"
    status_path.write_text("# PTRM IVON Fine-Tuning\n\n")
    utils.write_json(run_root / "configs" / "args.json", utils.serializable_args(args))
    utils.write_json(run_root / "configs" / "schedule.json", schedule)
    command = os.environ.get("VR_LAUNCH_COMMAND", " ".join([sys.executable] + sys.argv))
    (run_root / "command.txt").write_text(command + "\n")
    return run_root, status_path


def setup_device_and_seed(args: argparse.Namespace) -> torch.device:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("PTRM IVON training expects a CUDA device.")
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return device


def main() -> None:
    args = parse_args()
    if args.ivon_mc_samples != 1:
        raise ValueError("This recurrent IVON runner supports --ivon-mc-samples 1 only.")
    if args.q_loss_weight < 0:
        raise ValueError("--q-loss-weight must be non-negative.")
    for name in ("history_interval", "status_interval", "checkpoint_interval"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    stop_step = args.train_steps if args.stop_step is None else args.stop_step
    if stop_step <= 0 or stop_step > args.train_steps:
        raise ValueError(f"--stop-step must be in [1, train_steps], got {stop_step}.")
    if args.disable_compile:
        os.environ["DISABLE_COMPILE"] = "1"

    schedule = utils.load_schedule(args.schedule_json)
    utils.import_reference_code(args.trm_repo)
    from models.ema import EMAHelper
    from pretrain import init_train_state

    run_root, status_path = prepare_run(args, schedule)
    device = setup_device_and_seed(args)

    config = utils.make_config(args)
    batch_stream = utils.ExactTrainBatchStream(config)
    train_state = init_train_state(config, batch_stream.metadata, rank=0, world_size=1)
    train_state.total_steps = args.train_steps
    utils.load_model_weights(train_state.model, args.base_checkpoint, device, state_key=args.base_state_key)
    utils.replace_dense_optimizer(train_state, args)
    utils.patch_ivon_noise_scale(train_state.optimizers[-1], args.ivon_noise_scale)
    train_state.model.train()

    ema_helper = None
    if args.ema:
        ema_helper = EMAHelper(mu=args.ema_rate)
        ema_helper.register(train_state.model)

    if args.resume_checkpoint is not None:
        checkpoint = utils.restore_exact_checkpoint(
            train_state,
            batch_stream,
            ema_helper,
            args.resume_checkpoint,
            args,
            schedule,
            device,
        )
        utils.write_json(
            run_root / "resume.json",
            {
                "resume_checkpoint": str(args.resume_checkpoint),
                "resume_checkpoint_step": int(checkpoint["step"]),
                "target_stop_step": stop_step,
                "target_train_steps": args.train_steps,
                "exact_resume": True,
            },
        )
        if train_state.step >= stop_step:
            raise ValueError(f"Checkpoint step {train_state.step} is already >= stop_step {stop_step}.")

    wandb_run = utils.init_wandb(args, run_root, schedule, job_type="ptrm_ivon_ft")
    history_path = run_root / "history.jsonl"
    start_time = time.time()
    start_step = train_state.step
    last_metrics = None
    utils.update_status(
        status_path,
        f"Started fine-tuning from {args.base_checkpoint} at step {start_step}; target stop_step={stop_step}.",
    )

    try:
        while train_state.step < stop_step:
            _set_name, batch, global_batch_size = batch_stream.next_batch()
            last_metrics = utils.train_ivon_batch(config, train_state, batch, global_batch_size, args, schedule)
            if ema_helper is not None:
                ema_helper.update(train_state.model)

            if (
                train_state.step <= 10
                or (args.history_interval > 0 and train_state.step % args.history_interval == 0)
                or train_state.step == stop_step
            ):
                with history_path.open("a") as handle:
                    handle.write(json.dumps({"step": train_state.step, "metrics": last_metrics}, sort_keys=True) + "\n")
                if wandb_run is not None:
                    wandb_run.log(utils.numeric_metrics(last_metrics), step=train_state.step)

            if (
                train_state.step <= 10
                or (args.status_interval > 0 and train_state.step % args.status_interval == 0)
                or train_state.step == stop_step
            ):
                elapsed_s = time.time() - start_time
                utils.write_json(
                    run_root / "progress.json",
                    {
                        "step": train_state.step,
                        "start_step": start_step,
                        "stop_step": stop_step,
                        "train_steps": args.train_steps,
                        "elapsed_s": elapsed_s,
                        "steps_per_s": (train_state.step - start_step) / elapsed_s if elapsed_s > 0 else None,
                        "last_metrics": last_metrics,
                    },
                )
                utils.update_status(status_path, f"step {train_state.step}/{stop_step}; metrics={last_metrics}.")

            if args.checkpoint_interval and train_state.step % args.checkpoint_interval == 0:
                utils.save_public_checkpoint(train_state, args, run_root, f"step_{train_state.step}.pt", ema_helper)
                utils.save_exact_checkpoint(
                    train_state,
                    batch_stream,
                    args,
                    schedule,
                    run_root,
                    f"exact_step_{train_state.step}.pt",
                    ema_helper,
                    last_metrics,
                )

        public_checkpoint = utils.save_public_checkpoint(train_state, args, run_root, "checkpoint.pt", ema_helper)
        exact_checkpoint = utils.save_exact_checkpoint(
            train_state,
            batch_stream,
            args,
            schedule,
            run_root,
            "exact.pt",
            ema_helper,
            last_metrics,
        )
        summary = {
            "run_name": args.run_name,
            "stage": args.stage,
            "base_checkpoint": str(args.base_checkpoint),
            "base_state_key": args.base_state_key,
            "checkpoint": str(public_checkpoint),
            "exact_checkpoint": str(exact_checkpoint),
            "start_step": start_step,
            "stop_step": stop_step,
            "train_steps": args.train_steps,
            "elapsed_s": time.time() - start_time,
            "final_step": train_state.step,
            "last_metrics": last_metrics,
            "args": utils.serializable_args(args),
            "schedule": schedule,
            "exact_format_version": utils.EXACT_FORMAT_VERSION,
        }
        utils.write_json(run_root / "summary.json", summary)
        utils.update_status(status_path, f"Completed fine-tuning. Saved checkpoint: {public_checkpoint}.")
        if wandb_run is not None:
            for key, value in utils.numeric_metrics(last_metrics).items():
                wandb_run.summary[f"final/{key}"] = value
            wandb_run.summary["elapsed_s"] = summary["elapsed_s"]
            wandb_run.summary["final_step"] = train_state.step
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
