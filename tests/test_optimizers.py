from __future__ import annotations

import argparse
import copy

import pytest
import torch

from optim.factory import (
    add_optimizer_arguments,
    create_optimizer,
    optimizer_name,
    optimizer_sampling_context,
    supports_posterior_sampling,
)
from optim.soap import SOAP


def _linear_batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return torch.randn(8, 5, generator=generator), torch.randint(
        0, 2, (8,), generator=generator
    )


def _soap_train_step(
    model: torch.nn.Module,
    optimizer: SOAP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()
    optimizer.step()


def test_soap_first_step_initializes_then_second_step_updates() -> None:
    torch.manual_seed(1)
    model = torch.nn.Linear(5, 2)
    optimizer = SOAP(model.parameters(), lr=0.1, precondition_frequency=2)
    inputs, targets = _linear_batch()
    initial = model.weight.detach().clone()

    _soap_train_step(model, optimizer, inputs, targets)
    torch.testing.assert_close(model.weight, initial)
    _soap_train_step(model, optimizer, inputs, targets)
    assert not torch.equal(model.weight, initial)


def test_soap_projection_round_trip_and_orthogonal_basis() -> None:
    parameter = torch.nn.Parameter(torch.randn(3, 4))
    optimizer = SOAP([parameter], max_precond_dim=16)
    state: dict = {"step": 0, "exp_avg": torch.zeros_like(parameter), "exp_avg_sq": torch.zeros_like(parameter)}
    optimizer._init_preconditioner(
        torch.randn_like(parameter),
        state,
        precondition_frequency=2,
        shampoo_beta=0.95,
        max_precond_dim=16,
        precondition_1d=False,
        merge_dims=False,
    )
    optimizer._update_preconditioner(
        torch.randn_like(parameter),
        state,
        max_precond_dim=16,
        merge_dims=False,
        precondition_1d=False,
    )
    value = torch.randn_like(parameter)
    projected = optimizer._project(value, state, merge_dims=False, max_precond_dim=16)
    restored = optimizer._project_back(projected, state, merge_dims=False, max_precond_dim=16)
    torch.testing.assert_close(restored, value, atol=2e-5, rtol=2e-5)
    for basis in state["Q"]:
        torch.testing.assert_close(
            basis.T @ basis,
            torch.eye(basis.shape[1]),
            atol=2e-5,
            rtol=2e-5,
        )


@pytest.mark.parametrize("shape,max_precond_dim", [((4, 4), 16), ((1, 2, 3), 4)])
def test_soap_merged_dimensions_keep_moments_aligned(
    shape: tuple[int, ...], max_precond_dim: int
) -> None:
    parameter = torch.nn.Parameter(torch.randn(shape))
    optimizer = SOAP(
        [parameter],
        merge_dims=True,
        max_precond_dim=max_precond_dim,
        precondition_frequency=2,
    )
    for _ in range(4):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    assert optimizer.state[parameter]["exp_avg_sq"].shape == parameter.shape


def test_soap_closure_and_state_dict_round_trip() -> None:
    torch.manual_seed(2)
    model1 = torch.nn.Linear(5, 2)
    model2 = copy.deepcopy(model1)
    optimizer1 = SOAP(model1.parameters(), lr=0.03, precondition_frequency=2)
    optimizer2 = SOAP(model2.parameters(), lr=0.03, precondition_frequency=2)
    inputs, targets = _linear_batch()

    def closure() -> torch.Tensor:
        optimizer1.zero_grad()
        loss = torch.nn.functional.cross_entropy(model1(inputs), targets)
        loss.backward()
        return loss

    assert optimizer1.step(closure) is not None
    assert optimizer1.step(closure) is not None
    model2.load_state_dict(model1.state_dict())
    optimizer2.load_state_dict(copy.deepcopy(optimizer1.state_dict()))

    _soap_train_step(model1, optimizer1, inputs, targets)
    _soap_train_step(model2, optimizer2, inputs, targets)
    for left, right in zip(model1.parameters(), model2.parameters()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_factory_capabilities() -> None:
    parameter = torch.nn.Parameter(torch.randn(2, 3))
    soap = create_optimizer("soap", [("weight", parameter)], {"lr": 0.1})
    assert optimizer_name(soap) == "soap"
    assert not supports_posterior_sampling(soap)
    with optimizer_sampling_context(soap, train=True):
        pass
    with optimizer_sampling_context(soap, train=False, posterior_scale=0.0):
        pass
    with pytest.raises(ValueError, match="no posterior scale"):
        with optimizer_sampling_context(soap, train=False, posterior_scale=0.5):
            pass


def test_factory_registers_evon_bias_correction_switch() -> None:
    parser = argparse.ArgumentParser()
    add_optimizer_arguments(parser)
    args = parser.parse_args(["--optimizer", "evon", "--evon-no-correct-bias"])
    assert args.optimizer == "evon"
    assert args.evon_no_correct_bias


def test_soap_mixed_precision_state_resume_uses_work_dtype() -> None:
    parameter = torch.nn.Parameter(torch.randn(3, 4, dtype=torch.float16))
    optimizer = SOAP(
        [parameter], cast_dtype=torch.float32, max_precond_dim=16
    )
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = SOAP(
        [resumed_parameter], cast_dtype=torch.float32, max_precond_dim=16
    )
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    state = resumed.state[resumed_parameter]
    assert state["exp_avg"].dtype == torch.float32
    assert all(matrix.dtype == torch.float32 for matrix in state["GG"])
    assert all(matrix.dtype == torch.float32 for matrix in state["Q"])

    resumed_parameter.grad = torch.randn_like(resumed_parameter)
    resumed.step()


def test_evon_adapter_sampling_and_checkpoint_state() -> None:
    pytest.importorskip("evon")
    parameter = torch.nn.Parameter(torch.randn(2, 3))
    optimizer = create_optimizer(
        "evon",
        [("weight", parameter)],
        {
            "lr": 0.1,
            "evon_ess": 10.0,
            "evon_hess_init": 1.0,
            "evon_no_whiten_prec_grad": True,
        },
    )
    assert optimizer_name(optimizer) == "evon"
    assert supports_posterior_sampling(optimizer)
    mean = parameter.detach().clone()
    original_ess = optimizer.param_groups[0]["ess"]
    with optimizer_sampling_context(
        optimizer, train=False, posterior_scale=0.5
    ):
        assert not torch.equal(parameter, mean)
        assert optimizer.param_groups[0]["ess"] == original_ess / 0.25
    torch.testing.assert_close(parameter, mean, atol=0, rtol=0)
    assert optimizer.param_groups[0]["ess"] == original_ess
    assert "_mean_buf" in optimizer.state[parameter]
    assert all("_mean_buf" not in state for state in optimizer.state_dict()["state"].values())


def test_evon_price_hessian_update_matches_formula() -> None:
    pytest.importorskip("evon")
    from optim.evon import EVON

    hessian = torch.tensor([1.0, 2.0])
    average_noise_gradient = torch.tensor([0.1, -0.05])
    expected = hessian.clone()
    ess, weight_decay, eps, beta2 = 10.0, 0.2, 1e-8, 0.9
    raw = average_noise_gradient * (expected + weight_decay) * ess
    correction = (
        0.5
        * (1 - beta2) ** 2
        * (expected - raw).square()
        / (expected + weight_decay + eps)
    )
    expected = beta2 * expected + (1 - beta2) * raw + correction
    EVON._price_hess_update(
        hessian,
        average_noise_gradient,
        ess=ess,
        wd=weight_decay,
        eps=eps,
        beta2=beta2,
    )
    torch.testing.assert_close(hessian, expected)


def test_evon_mixed_precision_state_resume_uses_work_dtype() -> None:
    pytest.importorskip("evon")
    from optim.evon import EVON

    parameter = torch.nn.Parameter(torch.randn(3, 4, dtype=torch.float16))
    optimizer = EVON(
        [parameter],
        ess=10.0,
        hess_init=1.0,
        cast_dtype=torch.float32,
        max_precond_dim=16,
        whiten_prec_grad=False,
    )
    with optimizer.sampled_params(train=True):
        parameter.float().square().sum().backward()
    optimizer.step()

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = EVON(
        [resumed_parameter],
        ess=10.0,
        hess_init=1.0,
        cast_dtype=torch.float32,
        max_precond_dim=16,
        whiten_prec_grad=False,
    )
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    state = resumed.state[resumed_parameter]
    assert state["exp_avg"].dtype == torch.float32
    assert state["h_mom"].dtype == torch.float32
    assert all(matrix.dtype == torch.float32 for matrix in state["GG"])
    assert all(matrix.dtype == torch.float32 for matrix in state["Q"])

    with resumed.sampled_params(train=True):
        resumed_parameter.float().square().sum().backward()
    resumed.step()
