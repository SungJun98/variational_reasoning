from __future__ import annotations

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from optim.factory import create_optimizer, optimizer_name
from ptrm import utils
from ptrm.eval import load_optimizer_state


def _soap_config() -> dict[str, object]:
    return {
        "optimizer": "soap",
        "lr": 0.01,
        "beta1": 0.9,
        "soap_beta2": 0.95,
        "soap_max_precond_dim": 16,
    }


def test_optimizer_schedule_handles_soap_and_evon() -> None:
    parameter = torch.nn.Parameter(torch.randn(2, 3))
    soap = create_optimizer("soap", [("weight", parameter)], _soap_config())
    defaults = SimpleNamespace(lr=0.01, ivon_ess=100.0, evon_ess=200.0)
    values = utils.apply_optimizer_schedule(
        soap, {"dense_lr": {"type": "constant", "value": 0.02}}, 1, defaults
    )
    assert values == {"dense_lr": 0.02, "optimizer": "soap"}
    assert soap.param_groups[0]["lr"] == 0.02

    pytest.importorskip("evon")
    evon = create_optimizer(
        "evon",
        [("weight", torch.nn.Parameter(torch.randn(2, 3)))],
        {
            "lr": 0.01,
            "evon_ess": 200.0,
            "evon_hess_init": 1.0,
            "evon_max_precond_dim": 16,
            "evon_no_whiten_prec_grad": True,
        },
    )
    values = utils.apply_optimizer_schedule(
        evon,
        {
            "dense_lr": 0.03,
            "evon_ess": {"type": "constant", "value": 300.0},
        },
        1,
        defaults,
    )
    assert values["evon_ess"] == 300.0
    assert evon.param_groups[0]["ess"] == 300.0


def test_public_checkpoint_records_generic_and_legacy_optimizer_keys(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = create_optimizer("soap", list(model.named_parameters()), _soap_config())
    state = SimpleNamespace(model=model, optimizers=[optimizer], step=4)
    path = utils.save_public_checkpoint(
        state,
        argparse.Namespace(optimizer="soap"),
        tmp_path,
        "checkpoint.pt",
        None,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["optimizer_name"] == "soap"
    assert payload["optimizer_state_dict"] == payload["dense_optimizer_state_dict"]


def test_evaluator_restores_soap_optimizer_checkpoint() -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = create_optimizer("soap", list(model.named_parameters()), _soap_config())
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    payload = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_name": "soap",
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "args": _soap_config(),
    }

    restored_model = torch.nn.Linear(3, 2)
    restored = load_optimizer_state(
        restored_model, payload, Path("memory.pt"), "model_state_dict"
    )
    assert optimizer_name(restored) == "soap"
    assert len(restored.state) == len(optimizer.state)


def test_evaluator_preserves_multiple_optimizer_parameter_groups() -> None:
    model = torch.nn.Linear(3, 2)
    groups = [
        {"params": [model.weight], "param_names": ["weight"]},
        {"params": [model.bias], "param_names": ["bias"]},
    ]
    optimizer = create_optimizer(
        "ivon",
        groups,
        {"lr": 0.01, "ivon_ess": 100.0, "ivon_hess_init": 1.0},
    )
    payload = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_name": "ivon",
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "args": {"lr": 0.01, "ivon_ess": 100.0, "ivon_hess_init": 1.0},
    }
    restored = load_optimizer_state(
        torch.nn.Linear(3, 2), payload, Path("memory.pt"), "model_state_dict"
    )
    assert len(restored.param_groups) == 2
    assert [group["param_names"] for group in restored.param_groups] == [
        ["weight"],
        ["bias"],
    ]


def test_version_three_exact_checkpoint_remains_ivon_compatible() -> None:
    args = argparse.Namespace(
        optimizer="ivon",
        train_group_count_for_epochs=1000,
        soap_beta2=0.95,
        evon_beta2=0.9999,
        device="cuda:0",
    )
    checkpoint_args = utils.exact_resume_args(args)
    checkpoint_args.pop("optimizer")
    checkpoint_args.pop("soap_beta2")
    checkpoint_args.pop("evon_beta2")
    checkpoint = {
        "exact_format_version": 3,
        "args_for_resume": checkpoint_args,
        "schedule": {},
    }
    utils.validate_resume_metadata(checkpoint, args, {})

    args.optimizer = "soap"
    with pytest.raises(ValueError, match="only resume with IVON"):
        utils.validate_resume_metadata(checkpoint, args, {})
