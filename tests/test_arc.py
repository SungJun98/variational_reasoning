from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from rrm import arc, ptrm
from rrm.arc import (
    ArcDataset,
    ArcPrediction,
    aggregate_arc_predictions,
    crop_grid,
    inverse_augmentation,
    load_arc_dataset,
)
from rrm.utils import DistributedContext


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
    (test_root / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")
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
    original_name, inverse = inverse_augmentation("task|||t1|||0213456789")
    transformed = np.array([[1, 0], [2, 3]], dtype=np.uint8)

    restored = inverse(transformed)

    assert original_name == "task"
    assert restored.tolist() == [[1, 2], [3, 0]]


def test_aggregate_arc_predictions_uses_official_task_normalized_pass_at_k() -> None:
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
        ArcPrediction(
            "alpha", _fixture_grid_hash([[0]]), np.array([[0]], dtype=np.uint8), 2.0
        ),
        ArcPrediction(
            "alpha", _fixture_grid_hash([[0]]), np.array([[0]], dtype=np.uint8), -1.0
        ),
        ArcPrediction(
            "alpha", _fixture_grid_hash([[0]]), np.array([[1]], dtype=np.uint8), 3.0
        ),
        ArcPrediction(
            "alpha", _fixture_grid_hash([[2]]), np.array([[3]], dtype=np.uint8), 0.0
        ),
        ArcPrediction(
            "beta", _fixture_grid_hash([[4]]), np.array([[5]], dtype=np.uint8), 0.0
        ),
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
    test_puzzles = {"task": {"test": [{"input": [[0]], "output": [[2]]}]}}
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


def test_aggregate_arc_predictions_preserves_first_observation_on_exact_tie() -> None:
    test_puzzles = {"task": {"test": [{"input": [[0]], "output": [[1]]}]}}
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
    test_puzzles = {"task": {"test": [{"input": [[0]], "output": [[1]]}]}}

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
        ptrm.ADAPTER.build_model("unknown", "ptrm", {"arch": _tiny_arc_ptrm_config()})


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

    assert torch.equal(_parameter_vector(first_model), _parameter_vector(second_model))


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


def _arc_cli_argv(
    tmp_path: Path,
    *,
    sampling_args: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "all_config.yaml").write_text("arch: {}\n", encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    return [
        "--task",
        "arc-agi-1",
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        str(dataset),
        "--output",
        str(tmp_path / "output"),
        "--candidate-count",
        "25",
        "--depth",
        "16",
        "--global-batch-size",
        "32",
        *(sampling_args or ["--latent-noise-scale", "0.2"]),
        *(extra_args or []),
    ]


def test_arc_config_uses_sibling_yaml_and_resolves_frozen_protocol(
    tmp_path: Path,
) -> None:
    args = arc.build_parser().parse_args(_arc_cli_argv(tmp_path))

    config = arc.config_from_args(args, world_size=8)

    assert config.task == "arc-agi-1"
    assert config.config == (tmp_path / "all_config.yaml").resolve()
    assert config.candidate_count == 25
    assert config.depth == 16
    assert config.global_batch_size == 32
    assert config.latent_noise_scale == 0.2
    assert config.parameter_perturbation_scale is None
    assert config.seed == 0


def test_arc_parser_rejects_two_sampling_methods(tmp_path: Path) -> None:
    argv = _arc_cli_argv(
        tmp_path,
        sampling_args=[
            "--latent-noise-scale",
            "0.2",
            "--parameter-perturbation-scale",
            "0.3",
        ],
    )

    with pytest.raises(SystemExit):
        arc.build_parser().parse_args(argv)


@pytest.mark.parametrize(
    ("extra_args", "world_size", "message"),
    (
        (["--candidate-count", "0"], 8, "candidate count"),
        (["--depth", "0"], 8, "depth"),
        (["--global-batch-size", "31"], 8, "divisible"),
        (["--seed", "-1"], 8, "seed"),
    ),
)
def test_arc_config_rejects_invalid_protocol(
    tmp_path: Path,
    extra_args: list[str],
    world_size: int,
    message: str,
) -> None:
    args = arc.build_parser().parse_args(_arc_cli_argv(tmp_path, extra_args=extra_args))

    with pytest.raises(ValueError, match=message):
        arc.config_from_args(args, world_size=world_size)


def test_arc_config_rejects_nonempty_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.json").write_text("{}\n", encoding="utf-8")
    args = arc.build_parser().parse_args(_arc_cli_argv(tmp_path))

    with pytest.raises(FileExistsError, match="non-empty output"):
        arc.config_from_args(args, world_size=8)


def _partition_dataset(tmp_path: Path) -> ArcDataset:
    inputs = np.zeros((10, 900), dtype=np.int32)
    inputs[:, 0] = np.arange(10)
    labels = np.full((10, 900), 2, dtype=np.int32)
    return ArcDataset(
        root=tmp_path,
        metadata=arc.ArcDatasetMetadata(
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            vocab_size=12,
            seq_len=900,
            num_puzzle_identifiers=4,
            sets=("all",),
        ),
        inputs=inputs,
        labels=labels,
        puzzle_identifiers=np.array([1, 2, 3], dtype=np.int32),
        puzzle_indices=np.array([0, 2, 5, 10], dtype=np.int32),
        group_indices=np.array([0, 3], dtype=np.int32),
        identifier_map=("<blank>", "a", "b", "c"),
        test_puzzles={"task": {"test": [{"input": [[0]], "output": [[0]]}]}},
    )


def _evaluation_config(
    tmp_path: Path,
    *,
    checkpoint: Path | None = None,
    config_path: Path | None = None,
    global_batch_size: int = 6,
) -> arc.ArcEvaluationConfig:
    return arc.ArcEvaluationConfig(
        task="arc-agi-1",
        checkpoint=checkpoint or tmp_path / "checkpoint.pt",
        config=config_path or tmp_path / "all_config.yaml",
        dataset=tmp_path,
        output=tmp_path / "output",
        candidate_count=2,
        depth=2,
        global_batch_size=global_batch_size,
        seed=0,
        device="cpu",
        latent_noise_scale=0.2,
        parameter_perturbation_scale=None,
        max_batches=None,
    )


def test_iter_rank_batches_covers_rows_once_and_pads_the_final_rank(
    tmp_path: Path,
) -> None:
    dataset = _partition_dataset(tmp_path)
    config = _evaluation_config(tmp_path)

    rank_zero = list(arc.iter_rank_batches(dataset, config, rank=0, world_size=2))
    rank_one = list(arc.iter_rank_batches(dataset, config, rank=1, world_size=2))

    assert [batch.row_indices.tolist() for batch in rank_zero] == [[0, 1, 2], [6, 7, 8]]
    assert [batch.row_indices.tolist() for batch in rank_one] == [[3, 4, 5], [9]]
    covered = np.concatenate(
        [batch.row_indices for batches in (rank_zero, rank_one) for batch in batches]
    )
    assert sorted(covered.tolist()) == list(range(10))
    assert rank_one[-1].inputs.shape == (3, 900)
    assert rank_one[-1].inputs[:, 0].tolist() == [9, 0, 0]
    assert rank_one[-1].labels[1:].unique().tolist() == [-100]
    assert rank_one[-1].puzzle_identifiers.tolist() == [3, 0, 0]


def test_validate_checkpoint_identifier_rows_rejects_dataset_mismatch() -> None:
    state = {
        "_orig_mod.model.inner.puzzle_emb.weights": torch.zeros(2, 8),
    }

    with pytest.raises(ValueError, match="identifier rows 2 != dataset rows 3"):
        arc.validate_checkpoint_identifier_rows(state, expected_rows=3)


def test_load_arc_model_strict_loads_the_released_checkpoint_shape(
    tmp_path: Path,
) -> None:
    arch = {
        **_tiny_arc_ptrm_config(),
        "puzzle_emb_ndim": 8,
        "num_puzzle_identifiers": 2,
    }
    source = ptrm.TinyRecursiveReasoningModel(arch)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            f"_orig_mod.model.{name}": value.detach().clone()
            for name, value in source.state_dict().items()
        },
        checkpoint,
    )
    config_path = tmp_path / "all_config.yaml"
    config_path.write_text(
        "arch:\n"
        "  H_cycles: 1\n"
        "  H_layers: 0\n"
        "  L_cycles: 1\n"
        "  L_layers: 1\n"
        "  expansion: 1.0\n"
        "  forward_dtype: float32\n"
        "  halt_exploration_prob: 0.0\n"
        "  halt_max_steps: 16\n"
        "  hidden_size: 8\n"
        "  mlp_t: false\n"
        "  no_ACT_continue: true\n"
        "  num_heads: 2\n"
        "  pos_encodings: none\n"
        "  puzzle_emb_len: 0\n"
        "  puzzle_emb_ndim: 8\n",
        encoding="utf-8",
    )
    dataset = _partition_dataset(tmp_path)
    dataset = replace(
        dataset,
        metadata=replace(dataset.metadata, num_puzzle_identifiers=2),
        identifier_map=("<blank>", "a"),
    )
    config = _evaluation_config(
        tmp_path,
        checkpoint=checkpoint,
        config_path=config_path,
        global_batch_size=1,
    )

    loaded = arc.load_arc_model(config, dataset, device=torch.device("cpu"))

    for name, value in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)


def _write_synthetic_evaluation_dataset(root: Path) -> Path:
    test_root = root / "test"
    test_root.mkdir(parents=True)
    metadata = {
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "vocab_size": 12,
        "seq_len": 900,
        "num_puzzle_identifiers": 3,
        "total_groups": 1,
        "mean_puzzle_examples": 1.0,
        "total_puzzles": 2,
        "sets": ["all"],
    }
    (test_root / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")
    inputs = np.stack([_encoded_grid([[0]]), _encoded_grid([[0]])])
    labels = np.stack([_encoded_grid([[1]]), _encoded_grid([[1]])])
    np.save(test_root / "all__inputs.npy", inputs)
    np.save(test_root / "all__labels.npy", labels)
    np.save(
        test_root / "all__puzzle_identifiers.npy",
        np.array([1, 2], dtype=np.int32),
    )
    np.save(
        test_root / "all__puzzle_indices.npy",
        np.array([0, 1, 2], dtype=np.int32),
    )
    np.save(
        test_root / "all__group_indices.npy",
        np.array([0, 2], dtype=np.int32),
    )
    (root / "identifiers.json").write_text(
        json.dumps(["<blank>", "task", "task|||t0|||0123456789"]),
        encoding="utf-8",
    )
    (root / "test_puzzles.json").write_text(
        json.dumps(
            {
                "task": {
                    "train": [],
                    "test": [{"input": [[0]], "output": [[1]]}],
                }
            }
        ),
        encoding="utf-8",
    )
    return root


class _LiteralCandidateAdapter:
    def __init__(self) -> None:
        self.parameter_vectors: list[torch.Tensor] = []

    def evaluation_candidates(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        *,
        candidate_count: int,
        latent_noise_sigma: float,
        inference_depth: int,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del latent_noise_sigma, inference_depth, generator
        self.parameter_vectors.append(_parameter_vector(model).cpu().clone())
        batch_size = int(batch["inputs"].shape[0])
        wrong = torch.from_numpy(_encoded_grid([[2]]).astype(np.int64)).to(
            batch["inputs"].device
        )
        correct = torch.from_numpy(_encoded_grid([[1]]).astype(np.int64)).to(
            batch["inputs"].device
        )
        if candidate_count == 1:
            predictions = correct.view(1, 1, -1).expand(batch_size, 1, -1)
            score = float(self.parameter_vectors[-1].sum().item())
            scores = torch.full((batch_size, 1), score, device=batch["inputs"].device)
            return predictions, scores
        predictions = torch.stack((wrong, correct), dim=0)
        predictions = predictions.view(1, 2, -1).expand(batch_size, 2, -1)
        scores = torch.tensor([[0.0, 1.0]], device=batch["inputs"].device).expand(
            batch_size, 2
        )
        return predictions, scores


def test_run_arc_evaluation_writes_official_outputs_from_real_dataset(
    tmp_path: Path,
) -> None:
    root = _write_synthetic_evaluation_dataset(tmp_path / "dataset")
    dataset = load_arc_dataset(root)
    config = arc.ArcEvaluationConfig(
        task="arc-agi-1",
        checkpoint=tmp_path / "unused.pt",
        config=tmp_path / "unused.yaml",
        dataset=root,
        output=tmp_path / "output",
        candidate_count=2,
        depth=16,
        global_batch_size=2,
        seed=0,
        device="cpu",
        latent_noise_scale=0.2,
        parameter_perturbation_scale=None,
        max_batches=None,
    )
    model = _bank_model()
    adapter = _LiteralCandidateAdapter()

    result = arc.run_arc_evaluation(
        config,
        dataset,
        model=model,
        adapter=adapter,
        context=DistributedContext(0, 1, 0, torch.device("cpu")),
    )

    assert result is not None
    assert result.metrics == {
        "ARC/pass@1": 1.0,
        "ARC/pass@2": 1.0,
        "ARC/pass@5": 1.0,
        "ARC/pass@10": 1.0,
        "ARC/pass@100": 1.0,
        "ARC/pass@1000": 1.0,
    }
    assert json.loads((config.output / "metrics.json").read_text()) == result.metrics
    assert json.loads((config.output / "submission.json").read_text()) == {
        "task": [{"attempt_1": [[1]], "attempt_2": [[1]]}]
    }
    resolved = json.loads((config.output / "resolved_config.json").read_text())
    assert resolved["method"] == "ptrm"
    assert resolved["candidate_count"] == 2
    assert (config.output / "rank_predictions/rank_0_predictions.pkl").is_file()
    assert (config.output / "rank_0_runtime.json").is_file()


def test_run_arc_evaluation_reuses_one_w_ptrm_bank_across_batches(
    tmp_path: Path,
) -> None:
    root = _write_synthetic_evaluation_dataset(tmp_path / "dataset")
    dataset = load_arc_dataset(root)
    config = arc.ArcEvaluationConfig(
        task="arc-agi-2",
        checkpoint=tmp_path / "unused.pt",
        config=tmp_path / "unused.yaml",
        dataset=root,
        output=tmp_path / "output",
        candidate_count=2,
        depth=16,
        global_batch_size=1,
        seed=0,
        device="cpu",
        latent_noise_scale=None,
        parameter_perturbation_scale=0.3,
        max_batches=None,
    )
    model = _bank_model()
    original = _parameter_vector(model).clone()
    adapter = _LiteralCandidateAdapter()

    result = arc.run_arc_evaluation(
        config,
        dataset,
        model=model,
        adapter=adapter,
        context=DistributedContext(0, 1, 0, torch.device("cpu")),
    )

    assert result is not None
    assert len(adapter.parameter_vectors) == 4
    assert torch.equal(adapter.parameter_vectors[0], adapter.parameter_vectors[2])
    assert torch.equal(adapter.parameter_vectors[1], adapter.parameter_vectors[3])
    assert not torch.equal(adapter.parameter_vectors[0], adapter.parameter_vectors[1])
    assert torch.equal(_parameter_vector(model), original)
