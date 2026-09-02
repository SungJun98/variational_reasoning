from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from rrm import ptrm
from rrm.arc import (
    ArcDataset,
    ArcPrediction,
    aggregate_arc_predictions,
    crop_grid,
    inverse_augmentation,
    load_arc_dataset,
)


def _write_arc_dataset(
    root: Path,
    *,
    write_identifiers: bool = True,
    puzzle_indices: np.ndarray | None = None,
    puzzle_identifiers: np.ndarray | None = None,
) -> Path:
    test_root = root / "test"
    test_root.mkdir(parents=True)
    metadata = {
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "vocab_size": 12,
        "seq_len": 900,
        "num_puzzle_identifiers": 2,
        "total_groups": 1,
        "mean_puzzle_examples": 2.0,
        "total_puzzles": 1,
        "sets": ["all"],
    }
    (test_root / "dataset.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    np.save(test_root / "all__inputs.npy", np.zeros((2, 900), dtype=np.uint8))
    np.save(test_root / "all__labels.npy", np.ones((2, 900), dtype=np.uint8))
    np.save(
        test_root / "all__puzzle_identifiers.npy",
        (
            np.array([1], dtype=np.int32)
            if puzzle_identifiers is None
            else puzzle_identifiers
        ),
    )
    np.save(
        test_root / "all__puzzle_indices.npy",
        (
            np.array([0, 2], dtype=np.int32)
            if puzzle_indices is None
            else puzzle_indices
        ),
    )
    np.save(
        test_root / "all__group_indices.npy",
        np.array([0, 1], dtype=np.int32),
    )
    if write_identifiers:
        (root / "identifiers.json").write_text(
            json.dumps(["<blank>", "task|||t0|||0123456789"]),
            encoding="utf-8",
        )
    (root / "test_puzzles.json").write_text(
        json.dumps(
            {
                "task": {
                    "train": [],
                    "test": [{"input": [[1]], "output": [[2]]}],
                }
            }
        ),
        encoding="utf-8",
    )
    return root


def test_load_arc_dataset_preserves_memory_mapped_arrays(tmp_path: Path) -> None:
    dataset = load_arc_dataset(_write_arc_dataset(tmp_path / "arc"))

    assert isinstance(dataset, ArcDataset)
    assert isinstance(dataset.inputs, np.memmap)
    assert isinstance(dataset.labels, np.memmap)
    assert dataset.inputs.shape == (2, 900)
    assert dataset.puzzle_identifiers.tolist() == [1]
    assert dataset.metadata.num_puzzle_identifiers == 2
    assert dataset.identifier_map == ("<blank>", "task|||t0|||0123456789")


def test_load_arc_dataset_rejects_a_missing_identifier_sidecar(
    tmp_path: Path,
) -> None:
    root = _write_arc_dataset(tmp_path / "arc", write_identifiers=False)

    with pytest.raises(FileNotFoundError, match="identifiers.json"):
        load_arc_dataset(root)


def test_load_arc_dataset_rejects_malformed_puzzle_boundaries(
    tmp_path: Path,
) -> None:
    root = _write_arc_dataset(
        tmp_path / "arc",
        puzzle_indices=np.array([1, 2], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="puzzle_indices"):
        load_arc_dataset(root)


def test_load_arc_dataset_rejects_identifier_outside_metadata_range(
    tmp_path: Path,
) -> None:
    root = _write_arc_dataset(
        tmp_path / "arc",
        puzzle_identifiers=np.array([2], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="puzzle identifier"):
        load_arc_dataset(root)


def _encoded_grid(grid: list[list[int]]) -> np.ndarray:
    source = np.asarray(grid, dtype=np.uint8)
    encoded = np.zeros((30, 30), dtype=np.uint8)
    rows, columns = source.shape
    encoded[:rows, :columns] = source + 2
    if rows < 30:
        encoded[rows, :columns] = 1
    if columns < 30:
        encoded[:rows, columns] = 1
    return encoded.reshape(-1)


def _fixture_grid_hash(grid: list[list[int]]) -> str:
    array = np.asarray(grid, dtype=np.uint8)
    payload = bytes(array.shape) + array.tobytes()
    return hashlib.sha256(payload).hexdigest()


def test_crop_grid_recovers_the_literal_arc_rectangle() -> None:
    encoded = _encoded_grid([[0, 1], [2, 3]])

    cropped = crop_grid(encoded)

    assert cropped.dtype == np.uint8
    assert cropped.tolist() == [[0, 1], [2, 3]]


def test_inverse_augmentation_restores_rotation_and_color_mapping() -> None:
    original_name, inverse = inverse_augmentation(
        "task|||t1|||0213456789"
    )
    transformed = np.array([[1, 0], [2, 3]], dtype=np.uint8)

    restored = inverse(transformed)

    assert original_name == "task"
    assert restored.tolist() == [[1, 2], [3, 0]]


def test_aggregate_arc_predictions_uses_official_task_normalized_pass_at_k(
) -> None:
    test_puzzles = {
        "alpha": {
            "test": [
                {"input": [[0]], "output": [[1]]},
                {"input": [[2]], "output": [[3]]},
            ]
        },
        "beta": {"test": [{"input": [[4]], "output": [[5]]}]},
    }
    predictions = [
        ArcPrediction("alpha", _fixture_grid_hash([[0]]), np.array([[0]], dtype=np.uint8), 2.0),
        ArcPrediction("alpha", _fixture_grid_hash([[0]]), np.array([[0]], dtype=np.uint8), -1.0),
        ArcPrediction("alpha", _fixture_grid_hash([[0]]), np.array([[1]], dtype=np.uint8), 3.0),
        ArcPrediction("alpha", _fixture_grid_hash([[2]]), np.array([[3]], dtype=np.uint8), 0.0),
        ArcPrediction("beta", _fixture_grid_hash([[4]]), np.array([[5]], dtype=np.uint8), 0.0),
    ]

    result = aggregate_arc_predictions(
        predictions,
        test_puzzles=test_puzzles,
        pass_ks=(1, 2),
        submission_k=2,
    )

    assert result.metrics == {"ARC/pass@1": 0.75, "ARC/pass@2": 1.0}
    assert result.submission["alpha"][0] == {
        "attempt_1": [[0]],
        "attempt_2": [[1]],
    }
    assert result.submission["alpha"][1] == {
        "attempt_1": [[3]],
        "attempt_2": [[3]],
    }


def test_aggregate_arc_predictions_breaks_vote_ties_by_mean_sigmoid_q() -> None:
    test_puzzles = {
        "task": {"test": [{"input": [[0]], "output": [[2]]}]}
    }
    input_hash = _fixture_grid_hash([[0]])
    predictions = [
        ArcPrediction("task", input_hash, np.array([[1]], dtype=np.uint8), -2.0),
        ArcPrediction("task", input_hash, np.array([[2]], dtype=np.uint8), 2.0),
    ]

    result = aggregate_arc_predictions(
        predictions,
        test_puzzles=test_puzzles,
        pass_ks=(1,),
        submission_k=2,
    )

    assert result.metrics == {"ARC/pass@1": 1.0}
    assert result.submission["task"][0]["attempt_1"] == [[2]]


def test_aggregate_arc_predictions_preserves_first_observation_on_exact_tie(
) -> None:
    test_puzzles = {
        "task": {"test": [{"input": [[0]], "output": [[1]]}]}
    }
    input_hash = _fixture_grid_hash([[0]])
    predictions = [
        ArcPrediction("task", input_hash, np.array([[1]], dtype=np.uint8), 0.0),
        ArcPrediction("task", input_hash, np.array([[2]], dtype=np.uint8), 0.0),
    ]

    result = aggregate_arc_predictions(
        predictions,
        test_puzzles=test_puzzles,
        pass_ks=(1,),
        submission_k=2,
    )

    assert result.submission["task"][0]["attempt_1"] == [[1]]


def test_aggregate_arc_predictions_rejects_missing_test_pair() -> None:
    test_puzzles = {
        "task": {"test": [{"input": [[0]], "output": [[1]]}]}
    }

    with pytest.raises(ValueError, match="no predictions"):
        aggregate_arc_predictions(
            [],
            test_puzzles=test_puzzles,
            pass_ks=(1,),
            submission_k=2,
        )


def _tiny_arc_ptrm_config() -> dict[str, object]:
    return {
        "batch_size": 1,
        "seq_len": 900,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1,
        "vocab_size": 12,
        "H_cycles": 1,
        "L_cycles": 1,
        "H_layers": 0,
        "L_layers": 1,
        "hidden_size": 8,
        "expansion": 1.0,
        "num_heads": 2,
        "pos_encodings": "none",
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.0,
        "forward_dtype": "float32",
        "mlp_t": False,
        "puzzle_emb_len": 0,
        "no_ACT_continue": True,
    }


@pytest.mark.parametrize("task", ("arc-agi-1", "arc-agi-2"))
def test_ptrm_adapter_accepts_arc_task_names(task: str) -> None:
    model = ptrm.ADAPTER.build_model(
        task,
        "ptrm",
        {
            "arch": _tiny_arc_ptrm_config(),
            "candidate_count": 25,
            "latent_noise_sigma": 0.2,
            "inference_depth": 16,
        },
    )

    assert model._rrm_candidate_count == 25
    assert model._rrm_latent_noise_sigma == 0.2
    assert model._rrm_inference_depth == 16


def test_ptrm_adapter_still_rejects_unknown_task_name() -> None:
    with pytest.raises(ValueError, match="does not support task"):
        ptrm.ADAPTER.build_model(
            "unknown", "ptrm", {"arch": _tiny_arc_ptrm_config()}
        )


def _bank_model() -> torch.nn.Linear:
    model = torch.nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0]]))
        model.bias.copy_(torch.tensor([0.5]))
    return model


def _parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [parameter.detach().reshape(-1) for parameter in model.parameters()]
    )


def test_parameter_perturbation_bank_reuses_fixed_candidates_and_restores() -> None:
    model = _bank_model()
    original = _parameter_vector(model).clone()
    bank = ptrm.ParameterPerturbationBank.sample(
        model,
        candidate_count=3,
        relative_scale=0.3,
        generator=torch.Generator().manual_seed(17),
    )

    bank.apply(0)
    candidate_zero = _parameter_vector(model).clone()
    bank.apply(1)
    candidate_one = _parameter_vector(model).clone()
    bank.apply(0)
    repeated_zero = _parameter_vector(model).clone()
    bank.restore()

    assert not torch.equal(candidate_zero, original)
    assert not torch.equal(candidate_zero, candidate_one)
    assert torch.equal(repeated_zero, candidate_zero)
    assert torch.equal(_parameter_vector(model), original)


def test_parameter_perturbation_bank_is_deterministic_for_a_fixed_seed() -> None:
    first_model = _bank_model()
    second_model = _bank_model()
    first = ptrm.ParameterPerturbationBank.sample(
        first_model,
        candidate_count=2,
        relative_scale=0.3,
        generator=torch.Generator().manual_seed(23),
    )
    second = ptrm.ParameterPerturbationBank.sample(
        second_model,
        candidate_count=2,
        relative_scale=0.3,
        generator=torch.Generator().manual_seed(23),
    )

    first.apply(1)
    second.apply(1)

    assert torch.equal(
        _parameter_vector(first_model), _parameter_vector(second_model)
    )


@pytest.mark.parametrize(
    ("candidate_count", "relative_scale", "message"),
    ((0, 0.3, "candidate count"), (2, -0.1, "relative scale")),
)
def test_parameter_perturbation_bank_rejects_invalid_protocol(
    candidate_count: int, relative_scale: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ptrm.ParameterPerturbationBank.sample(
            _bank_model(),
            candidate_count=candidate_count,
            relative_scale=relative_scale,
            generator=torch.Generator().manual_seed(0),
        )
