"""ARC-AGI data contracts, evaluation, and official augmentation voting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ARC_MAX_GRID_SIZE = 30
ARC_PUZZLE_ID_SEPARATOR = "|||"
_DIHEDRAL_INVERSE = (0, 3, 2, 1, 4, 5, 6, 7)


@dataclass(frozen=True)
class ArcDatasetMetadata:
    """Metadata required to evaluate a preprocessed ARC split."""

    pad_id: int
    ignore_label_id: int | None
    blank_identifier_id: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    sets: tuple[str, ...]


@dataclass(frozen=True)
class ArcDataset:
    """Memory-mapped ARC test arrays plus official voting sidecars."""

    root: Path
    metadata: ArcDatasetMetadata
    inputs: np.ndarray
    labels: np.ndarray
    puzzle_identifiers: np.ndarray
    puzzle_indices: np.ndarray
    group_indices: np.ndarray
    identifier_map: tuple[str, ...]
    test_puzzles: Mapping[str, Any]


@dataclass(frozen=True)
class ArcPrediction:
    """One selected prediction restored to its original ARC coordinates."""

    task_name: str
    input_hash: str
    grid: np.ndarray
    q_logit: float


@dataclass(frozen=True)
class ArcEvaluationResult:
    """Official ARC metrics and two-attempt submission payload."""

    metrics: dict[str, float]
    submission: dict[str, list[dict[str, list[list[int]]]]]


def _dihedral_transform(array: np.ndarray, transform_id: int) -> np.ndarray:
    if transform_id == 0:
        return array
    if transform_id == 1:
        return np.rot90(array, k=1)
    if transform_id == 2:
        return np.rot90(array, k=2)
    if transform_id == 3:
        return np.rot90(array, k=3)
    if transform_id == 4:
        return np.fliplr(array)
    if transform_id == 5:
        return np.flipud(array)
    if transform_id == 6:
        return array.T
    if transform_id == 7:
        return np.fliplr(np.rot90(array, k=1))
    raise ValueError(f"Unknown ARC dihedral transform: {transform_id}")


def crop_grid(sequence: np.ndarray) -> np.ndarray:
    """Crop one tokenized 30-by-30 ARC sequence to its decoded color grid."""

    encoded = np.asarray(sequence)
    if encoded.shape not in ((ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE,), (30, 30)):
        raise ValueError("ARC grid sequence must contain exactly 900 tokens")
    encoded = encoded.reshape(ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE)
    maximum_area = 0
    maximum_shape = (0, 0)
    column_limit = ARC_MAX_GRID_SIZE
    for row_count in range(1, ARC_MAX_GRID_SIZE + 1):
        for column in range(1, column_limit + 1):
            token = int(encoded[row_count - 1, column - 1])
            if token < 2 or token > 11:
                column_limit = column - 1
                break
        area = row_count * column_limit
        if area > maximum_area:
            maximum_area = area
            maximum_shape = (row_count, column_limit)
    rows, columns = maximum_shape
    return (encoded[:rows, :columns].astype(np.int16) - 2).astype(np.uint8)


def inverse_augmentation(name: str) -> tuple[str, Callable[[np.ndarray], np.ndarray]]:
    """Return the original ARC task name and inverse augmentation function."""

    if ARC_PUZZLE_ID_SEPARATOR not in name:
        return name, lambda grid: np.asarray(grid, dtype=np.uint8)
    parts = name.split(ARC_PUZZLE_ID_SEPARATOR)
    if len(parts) < 3:
        raise ValueError(f"Malformed ARC augmented identifier: {name!r}")
    original_name = ARC_PUZZLE_ID_SEPARATOR.join(parts[:-2])
    transform_text, permutation_text = parts[-2:]
    if (
        len(transform_text) != 2
        or not transform_text.startswith("t")
        or not transform_text[1].isdigit()
    ):
        raise ValueError(f"Malformed ARC transform identifier: {name!r}")
    transform_id = int(transform_text[1])
    if transform_id not in range(8):
        raise ValueError(f"ARC transform ID must be in [0, 7]: {name!r}")
    if (
        len(permutation_text) != 10
        or not permutation_text.isdigit()
        or set(permutation_text) != set("0123456789")
    ):
        raise ValueError(f"Malformed ARC color permutation: {name!r}")
    inverse_permutation = np.argsort(
        np.asarray([int(value) for value in permutation_text], dtype=np.uint8)
    ).astype(np.uint8)
    inverse_transform_id = _DIHEDRAL_INVERSE[transform_id]

    def restore(grid: np.ndarray) -> np.ndarray:
        values = np.asarray(grid)
        if values.ndim != 2 or np.any(values < 0) or np.any(values > 9):
            raise ValueError("ARC grids must be two-dimensional colors in [0, 9]")
        untransformed = _dihedral_transform(values, inverse_transform_id)
        return inverse_permutation[untransformed].astype(np.uint8, copy=False)

    return original_name, restore


def grid_hash(grid: np.ndarray) -> str:
    """Hash an ARC grid with its two-dimensional shape."""

    values = np.asarray(grid)
    if (
        values.ndim != 2
        or values.shape[0] > 255
        or values.shape[1] > 255
        or np.any(values < 0)
        or np.any(values > 9)
    ):
        raise ValueError("ARC grids must be two-dimensional colors in [0, 9]")
    values = values.astype(np.uint8, copy=False)
    payload = bytes(values.shape) + values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _arc_grid(payload: Any) -> np.ndarray:
    values = np.asarray(payload)
    if (
        values.ndim != 2
        or values.shape[0] > ARC_MAX_GRID_SIZE
        or values.shape[1] > ARC_MAX_GRID_SIZE
        or values.shape[0] < 1
        or values.shape[1] < 1
        or np.any(values < 0)
        or np.any(values > 9)
    ):
        raise ValueError("Official ARC grids must be non-empty colors in [0, 9]")
    return values.astype(np.uint8)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def aggregate_arc_predictions(
    predictions: Sequence[ArcPrediction],
    *,
    test_puzzles: Mapping[str, Any],
    pass_ks: Sequence[int] = (1, 2, 5, 10, 100, 1000),
    submission_k: int = 2,
) -> ArcEvaluationResult:
    """Apply the official count-first, mean-Q ARC augmentation vote."""

    if not pass_ks or any(isinstance(value, bool) or value < 1 for value in pass_ks):
        raise ValueError("ARC Pass@K values must be positive integers")
    if submission_k < 1:
        raise ValueError("ARC submission candidate count must be positive")
    if not isinstance(test_puzzles, Mapping) or not test_puzzles:
        raise ValueError("ARC test puzzles must be a non-empty mapping")

    observations: dict[
        str,
        dict[str, dict[str, list[Any]]],
    ] = {}
    for prediction in predictions:
        if prediction.task_name not in test_puzzles:
            raise ValueError(f"Prediction references unknown ARC task {prediction.task_name!r}")
        if not isinstance(prediction.input_hash, str) or not prediction.input_hash:
            raise ValueError("ARC prediction input hash must be non-empty")
        q_logit = float(prediction.q_logit)
        if not math.isfinite(q_logit):
            raise ValueError("ARC prediction Q logit must be finite")
        grid = _arc_grid(prediction.grid)
        prediction_hash = grid_hash(grid)
        task = observations.setdefault(prediction.task_name, {})
        input_predictions = task.setdefault(prediction.input_hash, {})
        stats = input_predictions.setdefault(
            prediction_hash,
            [0, 0.0, grid.copy()],
        )
        stats[0] += 1
        stats[1] += _sigmoid(q_logit)

    correct_totals = [0.0 for _ in pass_ks]
    submission: dict[str, list[dict[str, list[list[int]]]]] = {}
    for task_name, puzzle in test_puzzles.items():
        if not isinstance(task_name, str) or not isinstance(puzzle, Mapping):
            raise ValueError("ARC test puzzle entries must map string names to objects")
        test_pairs = puzzle.get("test")
        if not isinstance(test_pairs, list) or not test_pairs:
            raise ValueError(f"ARC task {task_name!r} has no test pairs")
        task_correct = [0 for _ in pass_ks]
        task_submission: list[dict[str, list[list[int]]]] = []
        for pair in test_pairs:
            if not isinstance(pair, Mapping) or "input" not in pair or "output" not in pair:
                raise ValueError(f"ARC task {task_name!r} has a malformed test pair")
            input_hash = grid_hash(_arc_grid(pair["input"]))
            label_hash = grid_hash(_arc_grid(pair["output"]))
            candidates = observations.get(task_name, {}).get(input_hash)
            if not candidates:
                raise ValueError(
                    f"ARC task {task_name!r} input {input_hash} has no predictions"
                )
            ranked = sorted(
                candidates.items(),
                key=lambda item: (
                    int(item[1][0]),
                    float(item[1][1]) / int(item[1][0]),
                ),
                reverse=True,
            )
            for index, pass_k in enumerate(pass_ks):
                task_correct[index] += any(
                    prediction_hash == label_hash
                    for prediction_hash, _stats in ranked[:pass_k]
                )
            selected_grids = [stats[2] for _hash, stats in ranked[:submission_k]]
            while len(selected_grids) < submission_k:
                selected_grids.append(selected_grids[0])
            task_submission.append(
                {
                    f"attempt_{index + 1}": grid.tolist()
                    for index, grid in enumerate(selected_grids)
                }
            )
        submission[task_name] = task_submission
        for index, count in enumerate(task_correct):
            correct_totals[index] += count / len(test_pairs)

    task_count = len(test_puzzles)
    metrics = {
        f"ARC/pass@{pass_k}": correct_totals[index] / task_count
        for index, pass_k in enumerate(pass_ks)
    }
    return ArcEvaluationResult(metrics=metrics, submission=submission)


def _metadata_integer(
    payload: Mapping[str, Any], name: str, *, nullable: bool = False
) -> int | None:
    value = payload.get(name)
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"ARC metadata {name!r} must be an integer")
    return value


def _load_metadata(path: Path) -> tuple[ArcDatasetMetadata, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("ARC dataset metadata must be a JSON object")
    sets = payload.get("sets")
    if sets != ["all"]:
        raise ValueError("ARC evaluation requires dataset sets=['all']")
    pad_id = _metadata_integer(payload, "pad_id")
    ignore_label_id = _metadata_integer(payload, "ignore_label_id", nullable=True)
    blank_identifier_id = _metadata_integer(payload, "blank_identifier_id")
    vocab_size = _metadata_integer(payload, "vocab_size")
    seq_len = _metadata_integer(payload, "seq_len")
    num_puzzle_identifiers = _metadata_integer(payload, "num_puzzle_identifiers")
    assert pad_id is not None
    assert blank_identifier_id is not None
    assert vocab_size is not None
    assert seq_len is not None
    assert num_puzzle_identifiers is not None
    if seq_len != 900 or vocab_size != 12 or num_puzzle_identifiers < 1:
        raise ValueError(
            "ARC metadata requires seq_len=900, vocab_size=12, and positive identifiers"
        )
    return (
        ArcDatasetMetadata(
            pad_id=pad_id,
            ignore_label_id=ignore_label_id,
            blank_identifier_id=blank_identifier_id,
            vocab_size=vocab_size,
            seq_len=seq_len,
            num_puzzle_identifiers=num_puzzle_identifiers,
            sets=("all",),
        ),
        payload,
    )


def _validate_boundaries(
    name: str, values: np.ndarray, *, expected_stop: int
) -> None:
    boundaries = np.asarray(values)
    if (
        boundaries.ndim != 1
        or boundaries.size < 2
        or not np.issubdtype(boundaries.dtype, np.integer)
        or int(boundaries[0]) != 0
        or int(boundaries[-1]) != expected_stop
        or np.any(np.diff(boundaries.astype(np.int64, copy=False)) <= 0)
    ):
        raise ValueError(
            f"ARC {name} must be strictly increasing boundaries from 0 to {expected_stop}"
        )


def load_arc_dataset(path: Path) -> ArcDataset:
    """Load and validate one official-format preprocessed ARC test split."""

    root = path.expanduser().resolve(strict=True)
    test_root = root / "test"
    metadata_path = test_root / "dataset.json"
    identifiers_path = root / "identifiers.json"
    test_puzzles_path = root / "test_puzzles.json"
    required = (metadata_path, identifiers_path, test_puzzles_path)
    for required_path in required:
        if not required_path.is_file():
            raise FileNotFoundError(f"ARC dataset file is missing: {required_path}")

    metadata, raw_metadata = _load_metadata(metadata_path)
    array_paths = {
        name: test_root / f"all__{name}.npy"
        for name in (
            "inputs",
            "labels",
            "puzzle_identifiers",
            "puzzle_indices",
            "group_indices",
        )
    }
    missing = [str(array_path) for array_path in array_paths.values() if not array_path.is_file()]
    if missing:
        raise FileNotFoundError(f"ARC test split is missing required arrays: {missing}")
    arrays = {
        name: np.load(array_path, mmap_mode="r")
        for name, array_path in array_paths.items()
    }
    inputs = arrays["inputs"]
    labels = arrays["labels"]
    puzzle_identifiers = arrays["puzzle_identifiers"]
    puzzle_indices = arrays["puzzle_indices"]
    group_indices = arrays["group_indices"]
    if inputs.ndim != 2 or inputs.shape[1] != metadata.seq_len or inputs.shape[0] < 1:
        raise ValueError(
            f"ARC inputs must have shape [N, {metadata.seq_len}] with N positive"
        )
    if labels.shape != inputs.shape:
        raise ValueError("ARC labels must match the input array shape")
    _validate_boundaries(
        "puzzle_indices", puzzle_indices, expected_stop=int(inputs.shape[0])
    )
    puzzle_count = int(puzzle_indices.size - 1)
    if puzzle_identifiers.shape != (puzzle_count,) or not np.issubdtype(
        puzzle_identifiers.dtype, np.integer
    ):
        raise ValueError("ARC puzzle_identifiers must contain one integer per puzzle")
    identifier_values = np.asarray(puzzle_identifiers, dtype=np.int64)
    if np.any(identifier_values < 0) or np.any(
        identifier_values >= metadata.num_puzzle_identifiers
    ):
        raise ValueError("ARC puzzle identifier is outside the metadata range")
    _validate_boundaries(
        "group_indices", group_indices, expected_stop=puzzle_count
    )

    identifier_payload = json.loads(identifiers_path.read_text(encoding="utf-8"))
    if (
        not isinstance(identifier_payload, list)
        or len(identifier_payload) != metadata.num_puzzle_identifiers
        or not all(isinstance(value, str) for value in identifier_payload)
    ):
        raise ValueError("ARC identifiers.json does not match metadata identifiers")
    test_puzzles = json.loads(test_puzzles_path.read_text(encoding="utf-8"))
    if not isinstance(test_puzzles, Mapping) or not test_puzzles:
        raise ValueError("ARC test_puzzles.json must contain a non-empty task object")
    total_groups = raw_metadata.get("total_groups")
    if isinstance(total_groups, bool) or not isinstance(total_groups, int):
        raise ValueError("ARC metadata 'total_groups' must be an integer")
    if total_groups != int(group_indices.size - 1) or total_groups != len(test_puzzles):
        raise ValueError("ARC task count differs across metadata, groups, and sidecar")

    return ArcDataset(
        root=root,
        metadata=metadata,
        inputs=inputs,
        labels=labels,
        puzzle_identifiers=puzzle_identifiers,
        puzzle_indices=puzzle_indices,
        group_indices=group_indices,
        identifier_map=tuple(identifier_payload),
        test_puzzles=test_puzzles,
    )
