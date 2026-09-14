"""Saved-array atom-response suppression diagnostic for the completed ETT run.

This script performs no model sampling, training, trajectory generation, or native
environment stepping.  It derives two diagnostic arrays from the two saved B
evaluation arrays and reuses the original paired tuples and episode split.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
SOURCE = ROOT / "outputs" / "supervised_stochastic_ett_20260914_v1"
PAIRED = Path(
    r"C:\Users\trhua\Documents\Codex\2026-09-08\f\outputs"
    r"\supervised_ett_native_probe_v1\paired_transitions.npz"
)
COLLECTOR = Path(
    r"C:\Users\trhua\Documents\Codex\2026-09-08\f\work"
    r"\supervised-ett-audit\pointmaze_supervised_probe.py"
)
TARGET_COMMIT = "54b36620ddd6818ad91679ec5f587ff9f59e2474"
STATIONARY_THRESHOLD = 1e-7
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 614000000

# Frozen native PointMaze occupancy used by the corrected original evaluator.
POINTMAZE_WALLS = np.array(
    [
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
    ],
    dtype=np.int32,
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(name: str, value) -> None:
    (OUT / name).write_text(
        json.dumps(plain(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def full_energy_score(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Full unbiased energy-score U-statistic, including every j,k term.

    Diagonal pair distances are zero but are deliberately retained in the sum;
    the denominator is 2*K*(K-1), as in the completed experiment.
    """
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rows, draws, width = samples.shape
    assert width == 2 and target.shape == (rows, 2) and draws > 1
    result = np.empty(rows, dtype=np.float64)
    for start in range(0, rows, 128):
        stop = min(start + 128, rows)
        block = samples[start:stop]
        target_term = np.linalg.norm(block - target[start:stop, None], axis=-1).mean(1)
        all_pairs = np.linalg.norm(block[:, :, None] - block[:, None, :], axis=-1)
        pair_term = all_pairs.sum((1, 2)) / (2.0 * draws * (draws - 1))
        result[start:stop] = target_term - pair_term
    return result


def native_legal(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy)
    floor = np.floor(xy).astype(np.int64)
    bounded = (
        (xy[..., 0] >= 0)
        & (xy[..., 0] <= 9)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] <= 5)
    )
    cell = POINTMAZE_WALLS[
        np.clip(floor[..., 0], 0, 8), np.clip(floor[..., 1], 0, 4)
    ]
    return bounded & (cell == 0)


def stationary(xy: np.ndarray, current_xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xy - current_xy, axis=-1) <= STATIONARY_THRESHOLD


def gate_features(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.float64)
    motion = np.sqrt(
        np.mean(
            np.sum(np.diff(state.reshape(-1, 4, 2), axis=1) ** 2, axis=-1),
            axis=-1,
        )
    )
    x = state[:, 0]
    x_centers = np.repeat(np.array([0.5, 2.5, 4.5, 7.5]), 2)
    motion_centers = np.tile(np.array([0.0, 0.7]), 4)
    logits = (
        -0.5 * ((x[:, None] - x_centers) / 2.0) ** 2
        - 0.5 * ((motion[:, None] - motion_centers) / 0.35) ** 2
    )
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.column_stack([x, motion]), weights


def response_from_checkpoint(
    theta: np.ndarray, weights: np.ndarray, action_difference: np.ndarray
) -> np.ndarray:
    raw = np.asarray(theta[16:], dtype=np.float64).reshape(8, 2, 2)
    norm = np.sqrt(np.sum(raw**2, axis=(1, 2), keepdims=True))
    component = raw / np.maximum(1.0, norm)
    matrix = np.einsum("bj,jkl->bkl", weights, component)
    return np.einsum("bij,bj->bi", matrix, action_difference)


def q(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"available": False, "reason": "zero rows"}
    return {
        "min": float(values.min()),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def nearest_opposite(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.sqrt(np.sum((left[:, None] - right[None, :]) ** 2, axis=-1))
    return distance.min(axis=1), distance.min(axis=0)


def bootstrap_difference(
    modified: np.ndarray,
    original: np.ndarray,
    episode: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    transform=None,
) -> dict:
    """Modified minus original under paired whole-development-episode resampling."""
    if not mask.any():
        return {
            "available": False,
            "reason": "zero rows",
            "direction": "modified_minus_original",
            "replicates": 0,
        }
    episodes = np.arange(72, 96)
    count = np.array([np.sum(mask & (episode == item)) for item in episodes])
    modified_sum = np.array([modified[mask & (episode == item)].sum() for item in episodes])
    original_sum = np.array([original[mask & (episode == item)].sum() for item in episodes])
    denominator = weights @ count
    valid = denominator > 0
    m = (weights[valid] @ modified_sum) / denominator[valid]
    o = (weights[valid] @ original_sum) / denominator[valid]
    if transform is not None:
        m, o = transform(m), transform(o)
    point_m = modified[mask].mean()
    point_o = original[mask].mean()
    if transform is not None:
        point_m, point_o = transform(point_m), transform(point_o)
    difference = m - o
    return {
        "direction": "modified_minus_original",
        "difference": float(point_m - point_o),
        "ci95": np.quantile(difference, [0.025, 0.975]),
        "contributing_episodes": int(np.sum(count > 0)),
        "empty_bootstrap_replicates": int(np.sum(~valid)),
        "replicates": int(len(weights)),
    }


def summarize_version(
    samples: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
    atom: np.ndarray,
    es: np.ndarray,
) -> tuple[dict, dict]:
    predicted_mean = samples.mean(axis=1)
    squared_error = np.sum((predicted_mean - target) ** 2, axis=-1)
    euclidean_error = np.linalg.norm(predicted_mean - target, axis=-1)
    is_stationary = stationary(samples, current[:, None, :])
    legal = native_legal(samples)
    mean_legal = native_legal(predicted_mean)
    displacement = np.linalg.norm(samples - current[:, None, :], axis=-1)
    row = {
        "energy_score": es,
        "squared_mean_xy_error": squared_error,
        "mean_xy_euclidean_error": euclidean_error,
        "sample_stationary_fraction": is_stationary.mean(axis=1),
        "sample_moving_fraction": (~is_stationary).mean(axis=1),
        "mean_sample_displacement": displacement.mean(axis=1),
        "illegal_sample_fraction": (~legal).mean(axis=1),
        "illegal_mean_indicator": ~mean_legal,
    }
    aggregate = {
        "energy_score": float(es.mean()),
        "mean_xy_rmse": float(np.sqrt(squared_error.mean())),
        "mean_xy_euclidean_error": float(euclidean_error.mean()),
        "sample_stationary_fraction": float(is_stationary.mean()),
        "sample_moving_fraction": float((~is_stationary).mean()),
        "mean_sample_displacement": float(displacement.mean()),
        "illegal_sample_fraction": float((~legal).mean()),
        "illegal_mean_fraction": float((~mean_legal).mean()),
        "atom_fraction": float(atom.mean()),
        "atom_stationary_fraction": float(is_stationary[atom].mean()),
        "atom_moving_fraction": float((~is_stationary[atom]).mean()),
        "atom_mean_displacement": float(displacement[atom].mean()),
    }
    return row, aggregate


def empty_aggregate() -> dict:
    return {
        "available": False,
        "reason": "zero rows",
        "energy_score": None,
        "mean_xy_rmse": None,
        "mean_xy_euclidean_error": None,
        "sample_stationary_fraction": None,
        "sample_moving_fraction": None,
        "mean_sample_displacement": None,
        "illegal_sample_fraction": None,
        "illegal_mean_fraction": None,
        "atom_fraction": None,
        "atom_stationary_fraction": None,
        "atom_moving_fraction": None,
        "atom_mean_displacement": None,
    }


def source_contract_checks() -> dict:
    kernel_path = ROOT / "ett" / "pointmaze_region_pilot.py"
    emitter_path = ROOT / "ett" / "convex_action_transition.py"
    diagonal_path = ROOT / "ett" / "diagonal_transition.py"
    texts = {
        "kernel": kernel_path.read_text(encoding="utf-8"),
        "emitter": emitter_path.read_text(encoding="utf-8"),
        "diagonal": diagonal_path.read_text(encoding="utf-8"),
        "collector": COLLECTOR.read_text(encoding="utf-8"),
    }
    snippets = {
        "raw_atom_sets_zero_delta": "return jnp.where(stationary[..., None], 0.0, delta), stationary",
        "kernel_passes_projected_anchor_to_emitter": "y, diag = emit(theta[16:], s, a, xp, anchor[..., :2], 1.)",
        "kernel_saves_raw_atom_flag": "diag['stationary_atom'] = atom",
        "response_is_one_vector_per_row": "response=jnp.einsum('bij,bj->bi',matrix,action-xp)",
        "response_is_broadcast_across_anchor_draws": "proposal=anchor+response[:,None,:]",
        "collector_uses_same_snapshot_for_each_query": "branch = copy.deepcopy(env)",
        "collector_records_actual_queried_outcome": "y=next_obs[:8].copy()",
        "collector_query_zero_is_diagonal": "actions = [xb, forward, -xb, query_rng.uniform(-1, 1, 2).astype(np.float32)]",
    }
    locations = {
        "raw_atom_sets_zero_delta": "diagonal",
        "kernel_passes_projected_anchor_to_emitter": "kernel",
        "kernel_saves_raw_atom_flag": "kernel",
        "response_is_one_vector_per_row": "emitter",
        "response_is_broadcast_across_anchor_draws": "emitter",
        "collector_uses_same_snapshot_for_each_query": "collector",
        "collector_records_actual_queried_outcome": "collector",
        "collector_query_zero_is_diagonal": "collector",
    }
    found = {name: snippet in texts[locations[name]] for name, snippet in snippets.items()}
    assert all(found.values())
    return {
        "checks": found,
        "conclusion": (
            "On a saved raw-atom draw the diagonal sampler's raw displacement is exactly zero, "
            "so its projected anchor is current XY.  The emitter computes one response per row "
            "and broadcasts it across anchor draws.  Replacing flagged draws by current XY is "
            "therefore exactly zeroing that response on raw atom draws, after emission, without "
            "reconstructing non-atom anchors."
        ),
    }


def main() -> None:
    worktree_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert worktree_head == TARGET_COMMIT
    required = [
        SOURCE / "REPORT.md",
        SOURCE / "run.py",
        SOURCE / "provenance.json",
        SOURCE / "split.npz",
        SOURCE / "B_s0.npz",
        SOURCE / "B_s1.npz",
        SOURCE / "B_s0_evaluation.npz",
        SOURCE / "B_s1_evaluation.npz",
        PAIRED,
        COLLECTOR,
        ROOT / "ett" / "pointmaze_region_pilot.py",
        ROOT / "ett" / "convex_action_transition.py",
        ROOT / "ett" / "diagonal_transition.py",
    ]
    assert all(path.is_file() for path in required)
    original_hashes = {str(path): sha256(path) for path in required}
    original_provenance = json.loads((SOURCE / "provenance.json").read_text(encoding="utf-8"))
    for path in [
        PAIRED,
        COLLECTOR,
        ROOT / "ett" / "pointmaze_region_pilot.py",
        ROOT / "ett" / "convex_action_transition.py",
        ROOT / "ett" / "diagonal_transition.py",
    ]:
        assert original_provenance["sources"][str(path)] == sha256(path)

    with np.load(PAIRED, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    with np.load(SOURCE / "split.npz", allow_pickle=False) as loaded:
        split = {name: loaded[name] for name in loaded.files}

    assert data["s"].shape == (19200, 8)
    assert np.array_equal(data["s"][:, :6], data["y"][:, 2:])
    assert np.array_equal(split["train_episodes"], np.arange(72))
    assert np.array_equal(split["dev_episodes"], np.arange(72, 96))
    expected_rows = np.flatnonzero(data["episode"] >= 72)
    assert np.array_equal(split["dev_rows"], expected_rows)
    assert len(expected_rows) == 4800

    # Verify the collector's four-row same-context pairing and locate query=0.
    for episode in range(72, 96):
        for time in range(50):
            ids = np.flatnonzero((data["episode"] == episode) & (data["time"] == time))
            assert len(ids) == 4 and set(data["query"][ids]) == {0, 1, 2, 3}
            assert np.all(data["s"][ids] == data["s"][ids[0]])
            assert np.all(data["xb"][ids] == data["xb"][ids[0]])
            diagonal_id = ids[data["query"][ids] == 0][0]
            assert np.array_equal(data["xq"][diagonal_id], data["xb"][diagonal_id])

    rows = expected_rows
    state = data["s"][rows]
    target = data["y"][rows, :2]
    current = state[:, :2]
    episode = data["episode"][rows]
    exact_diagonal = np.all(data["xq"][rows] == data["xb"][rows], axis=1)
    actual_stationary = stationary(target, current)
    alive = ~data["dead_before"][rows]
    fatal_onset = alive & data["dead_after"][rows]

    # Query=0 is the collector's recorded diagonal.  Map its actual outcome to
    # all four rows from the same (episode,time) context without using labels.
    recorded_diagonal_stationary = np.empty(len(rows), dtype=bool)
    for offset in range(0, len(rows), 4):
        block = np.arange(offset, offset + 4)
        assert len(set(zip(episode[block], data["time"][rows[block]]))) == 1
        q0 = block[data["query"][rows[block]] == 0][0]
        recorded_diagonal_stationary[block] = actual_stationary[q0]

    groups = OrderedDict(
        [
            ("all_off_diagonal", ~exact_diagonal),
            ("already_dead_off_diagonal", (~exact_diagonal) & (~alive)),
            (
                "alive_off_diagonal_actual_stationary",
                (~exact_diagonal) & alive & actual_stationary,
            ),
            (
                "alive_off_diagonal_actual_moving",
                (~exact_diagonal) & alive & (~actual_stationary),
            ),
            (
                "alive_recorded_diagonal_stationary_paired_off_diagonal_moving",
                (~exact_diagonal)
                & alive
                & recorded_diagonal_stationary
                & (~actual_stationary),
            ),
            ("fatal_onset_off_diagonal", (~exact_diagonal) & fatal_onset),
            ("all_diagonal", exact_diagonal),
        ]
    )
    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_weights = bootstrap_rng.multinomial(
        24, np.full(24, 1.0 / 24), size=BOOTSTRAP_REPLICATES
    )

    verification = {
        "target_commit": TARGET_COMMIT,
        "source_contract": source_contract_checks(),
        "split": {
            "train_episodes": [0, 71],
            "development_episodes": [72, 95],
            "development_episode_count": 24,
            "development_rows": len(rows),
            "saved_split_exact": True,
            "paired_four_query_contexts_verified": 1200,
        },
        "seeds": {},
    }
    all_results = {}
    per_row = {"rows": rows, "episode": episode}
    features, weights = gate_features(state)
    assert np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=2e-15)

    response_summary = {}
    for seed in [0, 1]:
        name = f"B_s{seed}"
        with np.load(SOURCE / f"{name}_evaluation.npz", allow_pickle=False) as loaded:
            saved = {key: loaded[key] for key in loaded.files}
        assert np.array_equal(saved["rows"], rows)
        samples = saved["samples_xy"]
        atom = saved["atom"]
        assert samples.shape == (4800, 64, 2)
        assert atom.shape == (4800, 64) and atom.dtype == np.bool_

        reconstructed_es = full_energy_score(samples, target)
        reconstructed_stationary = stationary(samples, current[:, None, :])
        reconstructed_mean = samples.mean(axis=1)
        reconstructed_squared_error = np.sum((reconstructed_mean - target) ** 2, axis=-1)
        reconstructed_legal = native_legal(samples)
        reconstructed_mean_legal = native_legal(reconstructed_mean)

        # On exact diagonal rows response is structurally zero.  Atom draws must
        # therefore expose the zero-displacement anchor at current XY exactly.
        diagonal_atom = exact_diagonal[:, None] & atom
        diagonal_atom_error = np.abs(samples - current[:, None, :])[diagonal_atom].max()

        # All atom draws in one row have the same anchor (current XY) and receive
        # the same row response; their emitted XY must match one another.
        within_row_spread = 0.0
        rows_with_two_atoms = 0
        for index in np.flatnonzero(atom.sum(axis=1) >= 2):
            atom_xy = samples[index, atom[index]]
            within_row_spread = max(
                within_row_spread, float(np.abs(atom_xy - atom_xy[0]).max())
            )
            rows_with_two_atoms += 1

        verification["seeds"][name] = {
            "rows_exactly_aligned_to_saved_split": True,
            "atom_flag_shape": list(atom.shape),
            "atom_flag_dtype": str(atom.dtype),
            "atom_true_draws": int(atom.sum()),
            "atom_false_draws": int((~atom).sum()),
            "stationary_flag_exactly_reconstructed": bool(
                np.array_equal(saved["stationary"], reconstructed_stationary)
            ),
            "native_sample_legality_exactly_reconstructed": bool(
                np.array_equal(saved["legal"], reconstructed_legal)
            ),
            "native_mean_legality_exactly_reconstructed": bool(
                np.array_equal(saved["mean_legal"], reconstructed_mean_legal)
            ),
            "max_abs_saved_mean_difference": float(
                np.abs(saved["mean"] - reconstructed_mean).max()
            ),
            "max_abs_saved_squared_error_difference": float(
                np.abs(saved["squared_error"] - reconstructed_squared_error).max()
            ),
            "max_abs_saved_energy_score_difference": float(
                np.abs(saved["es"] - reconstructed_es).max()
            ),
            "full_u_statistic_reconstructed": True,
            "diagonal_atom_draw_max_abs_current_xy_error": float(diagonal_atom_error),
            "rows_with_at_least_two_atom_draws": rows_with_two_atoms,
            "atom_output_max_within_row_spread": within_row_spread,
        }
        assert verification["seeds"][name]["stationary_flag_exactly_reconstructed"]
        assert verification["seeds"][name]["native_sample_legality_exactly_reconstructed"]
        assert verification["seeds"][name]["native_mean_legality_exactly_reconstructed"]
        assert verification["seeds"][name]["max_abs_saved_mean_difference"] < 2e-6
        assert verification["seeds"][name]["max_abs_saved_squared_error_difference"] < 2e-6
        assert verification["seeds"][name]["max_abs_saved_energy_score_difference"] < 2e-12
        assert diagonal_atom_error == 0.0 and within_row_spread == 0.0

        modified_samples = samples.copy()
        modified_samples[atom] = np.broadcast_to(current[:, None, :], samples.shape)[atom]
        # Selection is solely the saved raw atom flag.  Labels are used only in
        # the subsequent descriptive masks.
        modified_es = full_energy_score(modified_samples, target)

        original_row, _ = summarize_version(samples, target, current, atom, reconstructed_es)
        modified_row, _ = summarize_version(modified_samples, target, current, atom, modified_es)
        result = {}
        for label, mask in groups.items():
            if mask.any():
                _, original_aggregate = summarize_version(
                    samples[mask], target[mask], current[mask], atom[mask], reconstructed_es[mask]
                )
                _, modified_aggregate = summarize_version(
                    modified_samples[mask], target[mask], current[mask], atom[mask], modified_es[mask]
                )
            else:
                original_aggregate = empty_aggregate()
                modified_aggregate = empty_aggregate()
            intervals = {
                "energy_score": bootstrap_difference(
                    modified_row["energy_score"],
                    original_row["energy_score"],
                    episode,
                    mask,
                    bootstrap_weights,
                ),
                "mean_xy_rmse": bootstrap_difference(
                    modified_row["squared_mean_xy_error"],
                    original_row["squared_mean_xy_error"],
                    episode,
                    mask,
                    bootstrap_weights,
                    np.sqrt,
                ),
                "mean_xy_euclidean_error": bootstrap_difference(
                    modified_row["mean_xy_euclidean_error"],
                    original_row["mean_xy_euclidean_error"],
                    episode,
                    mask,
                    bootstrap_weights,
                ),
                "sample_stationary_fraction": bootstrap_difference(
                    modified_row["sample_stationary_fraction"],
                    original_row["sample_stationary_fraction"],
                    episode,
                    mask,
                    bootstrap_weights,
                ),
                "mean_sample_displacement": bootstrap_difference(
                    modified_row["mean_sample_displacement"],
                    original_row["mean_sample_displacement"],
                    episode,
                    mask,
                    bootstrap_weights,
                ),
                "illegal_sample_fraction": bootstrap_difference(
                    modified_row["illegal_sample_fraction"],
                    original_row["illegal_sample_fraction"],
                    episode,
                    mask,
                    bootstrap_weights,
                ),
                "illegal_mean_fraction": bootstrap_difference(
                    modified_row["illegal_mean_indicator"].astype(float),
                    original_row["illegal_mean_indicator"].astype(float),
                    episode,
                    mask,
                    bootstrap_weights,
                ),
            }
            result[label] = {
                "rows": int(mask.sum()),
                "episodes": int(len(np.unique(episode[mask]))),
                "actual_outcome_stationary_fraction": (
                    float(actual_stationary[mask].mean()) if mask.any() else None
                ),
                "actual_outcome_moving_fraction": (
                    float((~actual_stationary[mask]).mean()) if mask.any() else None
                ),
                "original": original_aggregate,
                "modified": modified_aggregate,
                "paired_episode_bootstrap": intervals,
                "relative_energy_score_change": (
                    float(
                        (modified_aggregate["energy_score"] - original_aggregate["energy_score"])
                        / original_aggregate["energy_score"]
                    )
                    if mask.any()
                    else None
                ),
            }
        all_results[name] = result

        checkpoint = np.load(SOURCE / f"{name}.npz", allow_pickle=False)["theta"]
        response = response_from_checkpoint(
            checkpoint, weights, data["xq"][rows] - data["xb"][rows]
        )
        response_norm = np.linalg.norm(response, axis=1)
        response_summary[name] = {
            label: {
                "rows": int(mask.sum()),
                "pre_projection_response_norm": q(response_norm[mask]),
                "action_difference_norm": q(
                    np.linalg.norm(
                        data["xq"][rows[mask]] - data["xb"][rows[mask]], axis=1
                    )
                ),
            }
            for label, mask in groups.items()
        }

        mean_modified = modified_samples.mean(axis=1)
        np.savez_compressed(
            OUT / f"{name}_atom_response_suppressed_evaluation.npz",
            rows=rows,
            samples_xy=modified_samples,
            es=modified_es,
            mean=mean_modified,
            squared_error=np.sum((mean_modified - target) ** 2, axis=-1),
            legal=native_legal(modified_samples),
            mean_legal=native_legal(mean_modified),
            atom=atom,
            intervened_atom=atom,
            stationary=stationary(modified_samples, current[:, None, :]),
        )
        per_row.update(
            {
                f"{name}_original_es": reconstructed_es,
                f"{name}_modified_es": modified_es,
                f"{name}_original_mean_xy": reconstructed_mean,
                f"{name}_modified_mean_xy": mean_modified,
            }
        )

    # The alive-stationary stratum is retained above even if empty.  For the
    # response-gate overlap question, compare every observed stationary
    # off-diagonal consequence with every observed moving alive consequence.
    stationary_mask = (~exact_diagonal) & actual_stationary
    moving_mask = groups["alive_off_diagonal_actual_moving"]
    normalized_features = np.column_stack([features[:, 0] / 2.0, features[:, 1] / 0.35])
    stationary_nearest, moving_nearest = nearest_opposite(
        normalized_features[stationary_mask], normalized_features[moving_mask]
    )
    gate_stationary_nearest, gate_moving_nearest = nearest_opposite(
        weights[stationary_mask], weights[moving_mask]
    )

    # Same-context outcome heterogeneity among alive off-diagonal queries.
    mixed_contexts = []
    for offset in range(0, len(rows), 4):
        block = np.arange(offset, offset + 4)
        off_alive = block[(~exact_diagonal[block]) & alive[block]]
        if len(off_alive) and actual_stationary[off_alive].any() and (~actual_stationary[off_alive]).any():
            mixed_contexts.append(block[0])

    feature_overlap = {
        "definition": {
            "horizontal_position": "state[:,0]",
            "f4_motion_summary": (
                "sqrt(mean over three frame differences of squared 2D displacement))"
            ),
            "gate_scaled_distance": "Euclidean distance after dividing x by 2 and motion by 0.35",
        },
        "off_diagonal_actual_stationary": {
            "rows": int(stationary_mask.sum()),
            "episodes": int(len(np.unique(episode[stationary_mask]))),
            "horizontal_position": q(features[stationary_mask, 0]),
            "f4_motion_summary": q(features[stationary_mask, 1]),
            "nearest_moving_gate_scaled_distance": q(stationary_nearest),
            "fraction_with_moving_neighbor_within": {
                str(radius): float((stationary_nearest <= radius).mean())
                for radius in [0.1, 0.25, 0.5]
            },
            "nearest_moving_gate_weight_l2_distance": q(gate_stationary_nearest),
        },
        "alive_off_diagonal_actual_moving": {
            "rows": int(moving_mask.sum()),
            "episodes": int(len(np.unique(episode[moving_mask]))),
            "horizontal_position": q(features[moving_mask, 0]),
            "f4_motion_summary": q(features[moving_mask, 1]),
            "nearest_stationary_gate_scaled_distance": q(moving_nearest),
            "fraction_with_stationary_neighbor_within": {
                str(radius): float((moving_nearest <= radius).mean())
                for radius in [0.1, 0.25, 0.5]
            },
            "nearest_stationary_gate_weight_l2_distance": q(gate_moving_nearest),
        },
        "same_context_checks": {
            "alive_contexts_with_both_stationary_and_moving_off_diagonal_actual_outcomes": len(
                mixed_contexts
            ),
            "episodes_for_those_mixed_contexts": int(
                len(np.unique(episode[mixed_contexts])) if mixed_contexts else 0
            ),
            "recorded_diagonal_stationary_paired_off_diagonal_moving_rows": int(
                groups[
                    "alive_recorded_diagonal_stationary_paired_off_diagonal_moving"
                ].sum()
            ),
            "exact_feature_identity_within_each_paired_context": True,
        },
        "checkpoint_response": response_summary,
        "interpretation_limit": (
            "Overlap concerns this eight-gate response component only. It does not prove that the "
            "complete model family cannot fit the conditional law."
        ),
    }

    for label, mask in groups.items():
        per_row[f"group__{label}"] = mask
    np.savez_compressed(OUT / "per_row_results.npz", **per_row)
    np.savez_compressed(
        OUT / "group_masks.npz",
        rows=rows,
        actual_stationary=actual_stationary,
        recorded_diagonal_stationary=recorded_diagonal_stationary,
        **{label: mask for label, mask in groups.items()},
    )
    write_json("results.json", all_results)
    write_json("feature_overlap.json", feature_overlap)
    write_json("verification.json", verification)

    assert all(original_hashes[str(path)] == sha256(path) for path in required)
    write_json(
        "provenance.json",
        {
            "target_commit": TARGET_COMMIT,
            "analysis_worktree_head": worktree_head,
            "source_experiment_head": original_provenance["head"],
            "analysis_code_sha256": sha256(Path(__file__)),
            "input_sha256": original_hashes,
            "source_provenance_hash_matches": True,
            "source_hashes_unchanged_during_analysis": True,
            "stationary_threshold": STATIONARY_THRESHOLD,
            "native_legality": (
                "inclusive bounds x in [0,9], y in [0,5], floor-cell lookup with clipped indices"
            ),
            "diagonal_definition": "exact equality of both raw action coordinates",
            "bootstrap": {
                "unit": "original development episode",
                "episodes": list(range(72, 96)),
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "pairing": "modified and original use the same resampled episode multiplicities",
            },
            "selection_for_intervention": "saved raw atom flag only; no death/outcome labels",
            "model_successor_calls": 0,
            "training_updates": 0,
            "native_steps": 0,
            "new_trajectories": 0,
            "non_atom_anchor_reconstruction": False,
        },
    )
    write_json(
        "completion.json",
        {
            "status": "complete",
            "source_artifacts_modified": False,
            "diagnostic_variants": 2,
            "saved_arrays_only": True,
        },
    )


if __name__ == "__main__":
    main()
