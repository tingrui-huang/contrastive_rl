"""Read-only critic/MC contrast agreement audit from saved arrays only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ett.rollout_return import GOAL, task_reward


CONTINUATION = Path("artifacts/pointmaze_region_pilot/continuation_precision_s01_v1")
RESPONSE = Path("artifacts/pointmaze_region_pilot/response_search_s01_v1")
OUT = Path("artifacts/pointmaze_region_pilot/contrast_agreement_v1")
NOTE = Path("notes/pointmaze_contrast_agreement.md")
PAIRS = ("s0_k2", "s1_k0")
BOOTSTRAP_REPLICATES = 2000
CONTINUATION_SEED = 190000000
RESPONSE_SEED = 190000001


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _interval(values: np.ndarray) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def _weighted_midrank(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    sorted_weights = weights[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], len(values)]
    result = np.empty(len(values), dtype=np.float64)
    below = 0.0
    for start, end in zip(starts, ends):
        group_weight = float(sorted_weights[start:end].sum())
        result[order[start:end]] = below + 0.5 * group_weight
        below += group_weight
    return result


def _weighted_spearman(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> float | None:
    keep = np.asarray(weights) > 0
    x = np.asarray(x)[keep]
    y = np.asarray(y)[keep]
    weights = np.asarray(weights, dtype=np.float64)[keep]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rx = _weighted_midrank(x, weights)
    ry = _weighted_midrank(y, weights)
    total = weights.sum()
    mx = np.sum(weights * rx) / total
    my = np.sum(weights * ry) / total
    dx = rx - mx
    dy = ry - my
    denominator = np.sqrt(np.sum(weights * dx * dx) * np.sum(weights * dy * dy))
    if denominator == 0:
        return None
    return float(np.sum(weights * dx * dy) / denominator)


def _root_quadratic_components(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    root_positions: np.ndarray,
    root_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Root-aggregated pair sums for weighted Kendall tau-b."""
    weights = np.asarray(weights, dtype=np.float64)
    sx = np.sign(np.subtract.outer(x, x))
    sy = np.sign(np.subtract.outer(y, y))
    pair_weight = 0.5 * np.multiply.outer(weights, weights)
    membership = np.zeros((len(x), root_count), dtype=np.float64)
    membership[np.arange(len(x)), root_positions] = 1.0
    numerator = membership.T @ (pair_weight * sx * sy) @ membership
    comparable_x = membership.T @ (pair_weight * (sx != 0)) @ membership
    comparable_y = membership.T @ (pair_weight * (sy != 0)) @ membership
    return numerator, comparable_x, comparable_y


def _quadratic_rows(counts: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("bi,ij,bj->b", counts, matrix, counts, optimize=True)


def _weighted_kendall_from_components(
    counts: np.ndarray,
    components: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    numerator = _quadratic_rows(counts, components[0])
    comparable_x = _quadratic_rows(counts, components[1])
    comparable_y = _quadratic_rows(counts, components[2])
    denominator = np.sqrt(comparable_x * comparable_y)
    result = np.full(len(counts), np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return np.clip(result, -1.0, 1.0)


def _weighted_rank_rows(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted midranks for many bootstrap weight rows and fixed values."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    sorted_weights = weights[:, order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], len(values)]
    result_sorted = np.empty_like(sorted_weights, dtype=np.float64)
    below = np.zeros(len(weights), dtype=np.float64)
    for start, end in zip(starts, ends):
        group_weight = sorted_weights[:, start:end].sum(axis=1)
        result_sorted[:, start:end] = (below + 0.5 * group_weight)[:, None]
        below += group_weight
    result = np.empty_like(result_sorted)
    result[:, order] = result_sorted
    return result


def _weighted_spearman_rows(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    rx = _weighted_rank_rows(x, weights)
    ry = _weighted_rank_rows(y, weights)
    total = weights.sum(axis=1)
    mx = np.divide(
        np.sum(weights * rx, axis=1), total, out=np.zeros_like(total), where=total > 0
    )
    my = np.divide(
        np.sum(weights * ry, axis=1), total, out=np.zeros_like(total), where=total > 0
    )
    dx = rx - mx[:, None]
    dy = ry - my[:, None]
    numerator = np.sum(weights * dx * dy, axis=1)
    denominator = np.sqrt(
        np.sum(weights * dx * dx, axis=1) * np.sum(weights * dy * dy, axis=1)
    )
    result = np.full(len(weights), np.nan, dtype=np.float64)
    valid = (total > 0) & (denominator > 0)
    result[valid] = numerator[valid] / denominator[valid]
    return np.clip(result, -1.0, 1.0)


def _root_bootstrap_counts(rng: np.random.Generator, root_count: int) -> np.ndarray:
    sampled = rng.integers(root_count, size=(BOOTSTRAP_REPLICATES, root_count))
    counts = np.zeros((BOOTSTRAP_REPLICATES, root_count), dtype=np.int16)
    rows = np.repeat(np.arange(BOOTSTRAP_REPLICATES), root_count)
    np.add.at(counts, (rows, sampled.ravel()), 1)
    return counts


def _continuation_arrays(name: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    folder = CONTINUATION / name
    with np.load(folder / "queries.npz") as archive:
        queries = {key: archive[key] for key in archive.files}
    critic_parts = []
    mc_parts = []
    changed_parts = []
    for batch_number in range(9):
        with np.load(folder / f"batch{batch_number}.npz") as archive:
            indices = archive["query_index"]
            expected = np.arange(batch_number * 256, min((batch_number + 1) * 256, 2304))
            np.testing.assert_array_equal(indices, expected)
            critic = archive["candidate_critic"] - archive["reference_critic"]
            mc = archive["candidate_mc"] - archive["reference_mc"]
            first_changed = archive["candidate_reward"] != archive["reference_reward"]
            later_changed = np.any(
                archive["candidate_continuation_rewards"]
                != archive["reference_continuation_rewards"],
                axis=-1,
            )
            critic_parts.append(critic.mean(axis=(1, 2)))
            mc_parts.append(mc.mean(axis=(1, 2)))
            changed_parts.append(np.any(first_changed[:, :, None] | later_changed, axis=(1, 2)))
    critic = np.concatenate(critic_parts)
    mc = np.concatenate(mc_parts)
    changed = np.concatenate(changed_parts)
    assert len(critic) == len(mc) == len(changed) == len(queries["root"]) == 2304
    return queries, critic, mc, changed


def _validate_published_means(
    name: str, queries: dict, critic: np.ndarray, mc: np.ndarray, published: dict
) -> dict:
    mass = np.asarray(queries["mass"], dtype=np.float64)
    np.testing.assert_allclose(mass.sum(), 1.0, rtol=0, atol=1e-12)
    reconstructed = {
        "critic": float(np.sum(mass * critic)),
        "mc": float(np.sum(mass * mc)),
    }
    expected = {
        key: float(published["comparisons"][name]["overall"][key]["mean"])
        for key in ("critic", "mc")
    }
    for key in reconstructed:
        np.testing.assert_allclose(
            reconstructed[key], expected[key], rtol=1e-12, atol=1e-12
        )
    return {
        "n_queries": int(len(critic)),
        "reconstructed": reconstructed,
        "published": expected,
        "absolute_error": {
            key: float(abs(reconstructed[key] - expected[key])) for key in reconstructed
        },
        "tolerance": {"relative": 1e-12, "absolute": 1e-12},
        "passed": True,
    }


def _stratum_statistics(
    critic: np.ndarray,
    mc: np.ndarray,
    design_weight: np.ndarray,
    roots: np.ndarray,
    root_labels: np.ndarray,
    mask: np.ndarray,
    changed: np.ndarray,
    bootstrap_counts: np.ndarray,
) -> dict:
    critic = np.asarray(critic)[mask]
    mc = np.asarray(mc)[mask]
    design_weight = np.asarray(design_weight, dtype=np.float64)[mask]
    roots = np.asarray(roots)[mask]
    changed = np.asarray(changed, dtype=bool)[mask]
    root_positions = np.searchsorted(root_labels, roots)
    assert np.array_equal(root_labels[root_positions], roots)
    n = len(critic)
    total_weight = float(design_weight.sum())
    degenerate = n < 2 or np.ptp(critic) == 0 or np.ptp(mc) == 0
    result = {
        "n_queries": int(n),
        "target_mass": total_weight,
        "degenerate": bool(degenerate),
        "critic_mean": {
            "estimate": float(np.sum(design_weight * critic) / total_weight),
            "uncertainty": "point estimate",
        },
        "mc_mean": {
            "estimate": float(np.sum(design_weight * mc) / total_weight),
            "uncertainty": "point estimate",
        },
        "exact_zero_counts": {
            "critic": int(np.sum(np.sign(critic) == 0)),
            "mc": int(np.sum(np.sign(mc) == 0)),
            "both": int(np.sum((np.sign(critic) == 0) & (np.sign(mc) == 0))),
        },
    }
    query_multiplicity = bootstrap_counts[:, root_positions].astype(np.float64)
    if degenerate:
        result.update(
            sign_agreement=None,
            weighted_spearman=None,
            weighted_kendall_tau_b=None,
            unavailable_reason="zero variance or fewer than two queries",
        )
    else:
        agreement = (np.sign(critic) == np.sign(mc)).astype(np.float64)
        agreement_denominator = query_multiplicity.sum(axis=1)
        agreement_bootstrap = np.divide(
            np.sum(query_multiplicity * agreement[None, :], axis=1),
            agreement_denominator,
            out=np.full(BOOTSTRAP_REPLICATES, np.nan),
            where=agreement_denominator > 0,
        )
        bootstrap_weights = query_multiplicity * design_weight[None, :]
        spearman_bootstrap = _weighted_spearman_rows(critic, mc, bootstrap_weights)
        components = _root_quadratic_components(
            critic, mc, design_weight, root_positions, len(root_labels)
        )
        kendall_bootstrap = _weighted_kendall_from_components(
            bootstrap_counts.astype(np.float64), components
        )
        point_counts = np.ones((1, len(root_labels)), dtype=np.float64)
        result.update(
            sign_agreement={
                "estimate": float(agreement.mean()),
                "ci95": _interval(agreement_bootstrap),
                "bootstrap_valid": int(np.isfinite(agreement_bootstrap).sum()),
                "bootstrap_degenerate": int(np.isnan(agreement_bootstrap).sum()),
            },
            weighted_spearman={
                "estimate": _weighted_spearman(critic, mc, design_weight),
                "ci95": _interval(spearman_bootstrap),
                "bootstrap_valid": int(np.isfinite(spearman_bootstrap).sum()),
                "bootstrap_degenerate": int(np.isnan(spearman_bootstrap).sum()),
            },
            weighted_kendall_tau_b={
                "estimate": float(
                    _weighted_kendall_from_components(point_counts, components)[0]
                ),
                "ci95": _interval(kendall_bootstrap),
                "bootstrap_valid": int(np.isfinite(kendall_bootstrap).sum()),
                "bootstrap_degenerate": int(np.isnan(kendall_bootstrap).sum()),
            },
        )

    changed_critic = critic[changed]
    changed_mc = mc[changed]
    changed_multiplicity = query_multiplicity[:, changed]
    changed_degenerate = (
        len(changed_critic) < 2
        or np.ptp(changed_critic) == 0
        or np.ptp(changed_mc) == 0
    )
    changed_result = {
        "n_queries": int(changed.sum()),
        "n_unchanged_queries_excluded": int((~changed).sum()),
        "degenerate": bool(changed_degenerate),
    }
    if changed_degenerate:
        changed_result.update(
            sign_agreement=None,
            unavailable_reason="zero variance or fewer than two changed queries",
        )
    else:
        changed_agreement = (
            np.sign(changed_critic) == np.sign(changed_mc)
        ).astype(np.float64)
        denominator = changed_multiplicity.sum(axis=1)
        values = np.divide(
            np.sum(changed_multiplicity * changed_agreement[None, :], axis=1),
            denominator,
            out=np.full(BOOTSTRAP_REPLICATES, np.nan),
            where=denominator > 0,
        )
        changed_result["sign_agreement"] = {
            "estimate": float(changed_agreement.mean()),
            "ci95": _interval(values),
            "bootstrap_valid": int(np.isfinite(values).sum()),
            "bootstrap_degenerate": int(np.isnan(values).sum()),
        }
    result["sequence_changed_only"] = changed_result
    return result


def _analyze_continuation() -> dict:
    published = _read(CONTINUATION / "results.json")
    loaded = {}
    validations = {}
    # Acceptance gate: complete all unstratified checks before stratified outcomes.
    for name in PAIRS:
        queries, critic, mc, changed = _continuation_arrays(name)
        validations[name] = _validate_published_means(
            name, queries, critic, mc, published
        )
        loaded[name] = (queries, critic, mc, changed)

    rng = np.random.default_rng(CONTINUATION_SEED)
    comparisons = {}
    for name in PAIRS:
        queries, critic, mc, changed = loaded[name]
        root_labels = np.unique(queries["root"])
        assert len(root_labels) == 36
        bootstrap_counts = _root_bootstrap_counts(rng, len(root_labels))
        comparisons[name] = {
            "label": f"{name}_u24 minus {name}_u0",
            "unstratified_reproduction": validations[name],
            "root_clusters": int(len(root_labels)),
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": CONTINUATION_SEED,
                "paired_by": "root",
                "interval": "percentile 95%",
            },
            "strata": {
                "pre_entry": _stratum_statistics(
                    critic,
                    mc,
                    queries["mass"],
                    queries["root"],
                    root_labels,
                    queries["pre_entry"],
                    changed,
                    bootstrap_counts,
                ),
                "entered": _stratum_statistics(
                    critic,
                    mc,
                    queries["mass"],
                    queries["root"],
                    root_labels,
                    ~queries["pre_entry"],
                    changed,
                    bootstrap_counts,
                ),
            },
        }
    return {
        "published_mean_acceptance_gate": {"passed": True, "comparisons": validations},
        "comparisons": comparisons,
    }


def _load_visitation(path: Path) -> list[dict]:
    with np.load(path) as archive:
        buckets = sorted({int(key.split("_")[0][1:]) for key in archive.files})
        return [
            {
                key.split("_", 1)[1]: archive[key]
                for key in archive.files
                if key.startswith(f"b{bucket}_")
            }
            for bucket in buckets
        ]


def _reconstruct_pre_entry(check: Path, matched: dict) -> np.ndarray:
    records = _load_visitation(check / "visitation.npz")
    paths = [
        (bucket, record, index)
        for bucket, record in enumerate(records)
        for index in range(len(record["root"]))
    ]
    recomputed_rewards = []
    for record in records:
        next_states = record["states"][:, 1:]
        goals = np.broadcast_to(GOAL, next_states.shape)
        rewards = np.asarray(task_reward(next_states, goals))
        np.testing.assert_array_equal(rewards, record["reward"])
        recomputed_rewards.append(rewards)

    flags = []
    for query_index, path_index in enumerate(matched["query_path"]):
        bucket, record, record_index = paths[int(path_index)]
        time = int(matched["query_t"][query_index])
        np.testing.assert_array_equal(
            matched["query_state"][query_index], record["states"][record_index, time]
        )
        np.testing.assert_array_equal(
            matched["query_action"][query_index], record["action"][record_index, time]
        )
        assert int(matched["query_length"][query_index]) == record["action"].shape[1]
        assert int(matched["query_root"][query_index]) == int(record["root"][record_index])
        flags.append(not bool(recomputed_rewards[bucket][record_index, :time].any()))
    return np.asarray(flags, dtype=bool)


def _response_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def _response_aggregate(
    critic: np.ndarray,
    mc: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict:
    degenerate = len(critic) < 2 or np.ptp(critic) == 0 or np.ptp(mc) == 0
    if degenerate:
        return {
            "n_comparisons": int(len(critic)),
            "degenerate": True,
            "sign_agreement": None,
            "spearman": None,
            "unavailable_reason": "zero variance or fewer than two comparisons",
        }
    boot = np.asarray(
        [
            _response_spearman(critic[indices], mc[indices])
            for indices in bootstrap_indices
        ],
        dtype=np.float64,
    )
    signs = np.sign(critic) == np.sign(mc)
    return {
        "n_comparisons": int(len(critic)),
        "degenerate": False,
        "sign_agreement": {
            "count": int(signs.sum()),
            "denominator": int(len(signs)),
            "rate": float(signs.mean()),
            "uncertainty": "point estimate",
        },
        "spearman": {
            "estimate": _response_spearman(critic, mc),
            "ci95": _interval(boot),
            "bootstrap_valid": int(np.isfinite(boot).sum()),
            "bootstrap_degenerate": int(np.isnan(boot).sum()),
        },
    }


def _analyze_response() -> dict:
    checks = sorted(
        path
        for path in (RESPONSE / "checks").iterdir()
        if (path / "matched_mc.npz").exists() and (path / "matched_mc.json").exists()
    )
    assert len(checks) == 18
    rows = []
    reconstruction_errors = []
    for check in checks:
        with np.load(check / "matched_mc.npz") as archive:
            matched = {key: archive[key] for key in archive.files}
        critic_query = (
            matched["proposed_critic"] - matched["before_critic"]
        ).mean(axis=(1, 2))
        mc_query = (matched["proposed_mc"] - matched["before_mc"]).mean(axis=(1, 2))
        saved = _read(check / "matched_mc.json")
        critic_delta = float(critic_query.mean())
        mc_delta = float(mc_query.mean())
        np.testing.assert_allclose(
            critic_delta, saved["critic"]["mean"], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            mc_delta, saved["mc"]["mean"], rtol=1e-12, atol=1e-12
        )
        row = {
            "name": check.name,
            "baseline": int(check.name[1]),
            "start": int(check.name.split("_k", 1)[1].split("_", 1)[0]),
            "update": int(check.name.rsplit("_u", 1)[1]),
            "n_queries": int(len(critic_query)),
            "critic_delta": {"estimate": critic_delta, "uncertainty": "point estimate"},
            "mc_delta": {"estimate": mc_delta, "uncertainty": "point estimate"},
            "sign_agreement": {
                "estimate": bool(np.sign(critic_delta) == np.sign(mc_delta)),
                "uncertainty": "point estimate",
            },
            "saved_json_point_estimates_reproduced": True,
        }
        try:
            pre_entry = _reconstruct_pre_entry(check, matched)
            if pre_entry.any():
                row["pre_entry"] = {
                    "flag_reconstructed": True,
                    "delta_estimable": True,
                    "n_queries": int(pre_entry.sum()),
                    "critic_delta": {
                        "estimate": float(critic_query[pre_entry].mean()),
                        "uncertainty": "point estimate",
                    },
                    "mc_delta": {
                        "estimate": float(mc_query[pre_entry].mean()),
                        "uncertainty": "point estimate",
                    },
                }
            else:
                row["pre_entry"] = {
                    "flag_reconstructed": True,
                    "delta_estimable": False,
                    "n_queries": 0,
                    "critic_delta": None,
                    "mc_delta": None,
                    "reason": "no pre-entry query was sampled in this saved check",
                }
        except Exception as error:  # Explicitly skip rather than fabricate a flag.
            reconstruction_errors.append(f"{check.name}: {type(error).__name__}: {error}")
            row["pre_entry"] = {
                "flag_reconstructed": False,
                "delta_estimable": False,
                "reason": f"{type(error).__name__}: {error}",
            }
        rows.append(row)

    critic = np.asarray([row["critic_delta"]["estimate"] for row in rows])
    mc = np.asarray([row["mc_delta"]["estimate"] for row in rows])
    rng = np.random.default_rng(RESPONSE_SEED)
    bootstrap_indices = rng.integers(len(rows), size=(BOOTSTRAP_REPLICATES, len(rows)))
    result = {
        "comparisons": rows,
        "all_comparisons": _response_aggregate(critic, mc, bootstrap_indices),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": RESPONSE_SEED,
            "paired_by": "comparison",
            "interval": "percentile 95%",
        },
        "saved_json_point_estimates_reproduced": True,
    }
    if reconstruction_errors:
        result["pre_entry_reconstruction"] = {
            "available": False,
            "method": "saved visitation states mapped by query_path/query_t; task_reward recomputation",
            "errors": reconstruction_errors,
        }
        result["pre_entry_comparisons"] = None
        result["pre_entry_estimable_subset"] = None
    else:
        estimable = [row for row in rows if row["pre_entry"]["delta_estimable"]]
        pre_critic = np.asarray(
            [row["pre_entry"]["critic_delta"]["estimate"] for row in estimable]
        )
        pre_mc = np.asarray(
            [row["pre_entry"]["mc_delta"]["estimate"] for row in estimable]
        )
        result["pre_entry_reconstruction"] = {
            "available": True,
            "method": "saved visitation states mapped by query_path/query_t; task_reward recomputed and matched saved rewards",
            "n_comparisons": len(rows),
            "n_estimable_comparisons": len(estimable),
            "n_zero_query_comparisons": len(rows) - len(estimable),
            "total_pre_entry_queries": int(
                sum(row["pre_entry"]["n_queries"] for row in rows)
            ),
        }
        if len(estimable) == len(rows):
            result["pre_entry_comparisons"] = _response_aggregate(
                pre_critic, pre_mc, bootstrap_indices
            )
            result["pre_entry_estimable_subset"] = result["pre_entry_comparisons"]
        else:
            result["pre_entry_comparisons"] = {
                "available": False,
                "n_requested_comparisons": len(rows),
                "n_estimable_comparisons": len(estimable),
                "reason": "at least one saved check contains zero pre-entry queries",
            }
            subset_indices = rng.integers(
                len(estimable), size=(BOOTSTRAP_REPLICATES, len(estimable))
            )
            result["pre_entry_estimable_subset"] = _response_aggregate(
                pre_critic, pre_mc, subset_indices
            )
    return result


def _fmt(value: float | None) -> str:
    return "undefined" if value is None else f"{value:+.4f}"


def _fmt_rate(value: float | None) -> str:
    return "undefined" if value is None else f"{100 * value:.1f}%"


def _fmt_ci(interval: list[float] | None) -> str:
    if interval is None:
        return "undefined"
    return f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"


def _fmt_rate_ci(interval: list[float] | None) -> str:
    if interval is None:
        return "undefined"
    return f"[{100 * interval[0]:.1f}%, {100 * interval[1]:.1f}%]"


def _observed_pattern(continuation: dict) -> str:
    pre = [
        continuation["comparisons"][name]["strata"]["pre_entry"] for name in PAIRS
    ]
    systematic = all(
        row["sign_agreement"] is not None
        and row["sign_agreement"]["ci95"][1] < 0.5
        and row["weighted_spearman"]["ci95"][1] < 0
        and row["weighted_kendall_tau_b"]["ci95"][1] < 0
        for row in pre
    )
    if systematic:
        return (
            "Both tested pre-entry pairs show sign agreement below one half and "
            "negative rank association with intervals excluding the reversal boundaries; "
            "the observed pattern is systematic sign reversal within these two pairs."
        )
    concordant = all(
        row["sign_agreement"] is not None
        and row["sign_agreement"]["ci95"][0] > 0.5
        and row["weighted_spearman"]["ci95"][0] > 0
        and row["weighted_kendall_tau_b"]["ci95"][0] > 0
        for row in pre
    )
    if concordant:
        return (
            "The critic is not systematically sign-reversed in either tested pre-entry "
            "pair: both have above-half sign agreement and positive rank association with "
            "root-bootstrap intervals excluding the reversal boundaries. The s0_k2 "
            "opposite weighted means therefore reflect contrast magnitudes, not pervasive "
            "per-query order reversal."
        )
    directions = [
        (
            row["sign_agreement"]["estimate"] if row["sign_agreement"] else None,
            row["weighted_spearman"]["estimate"] if row["weighted_spearman"] else None,
        )
        for row in pre
    ]
    if directions[0][1] is not None and directions[1][1] is not None and np.sign(
        directions[0][1]
    ) != np.sign(directions[1][1]):
        return (
            "The two tested pre-entry pairs have opposite rank-association directions, "
            "so the observed pattern is mixed/noisy rather than a shared systematic reversal."
        )
    return (
        "The joint pre-entry intervals do not support a shared systematic sign reversal; "
        "the observed ordering is unresolved/noisy at this sample size."
    )


def _render_report(results: dict) -> str:
    continuation = results["continuation_precision"]
    response = results["response_search"]
    lines = [
        "# Read-only critic/MC contrast agreement audit",
        "",
        _observed_pattern(continuation),
        "",
        "All intervals below are descriptive percentile 95% intervals. Means and "
        "per-comparison deltas are explicitly point estimates.",
        "",
        "## Continuation precision",
        "",
    ]
    for name in PAIRS:
        comparison = continuation["comparisons"][name]
        reproduction = comparison["unstratified_reproduction"]
        lines.extend(
            [
                f"### {comparison['label']}",
                "",
                f"The unstratified acceptance gate passed at n={reproduction['n_queries']} queries: "
                f"critic {_fmt(reproduction['reconstructed']['critic'])} and MC "
                f"{_fmt(reproduction['reconstructed']['mc'])} (point estimates), reproducing "
                "the published means within absolute and relative tolerance 1e-12.",
                "",
            ]
        )
        for stratum_name in ("pre_entry", "entered"):
            row = comparison["strata"][stratum_name]
            changed = row["sequence_changed_only"]
            if row["degenerate"]:
                lines.append(
                    f"- `{stratum_name}`: n={row['n_queries']}; degenerate, so agreement "
                    "and rank correlations are unavailable."
                )
                continue
            lines.append(
                f"- `{stratum_name}`: n={row['n_queries']}; sign agreement "
                f"{_fmt_rate(row['sign_agreement']['estimate'])}, 95% CI "
                f"{_fmt_rate_ci(row['sign_agreement']['ci95'])}; weighted Spearman "
                f"{_fmt(row['weighted_spearman']['estimate'])}, 95% CI "
                f"{_fmt_ci(row['weighted_spearman']['ci95'])}; weighted Kendall tau-b "
                f"{_fmt(row['weighted_kendall_tau_b']['estimate'])}, 95% CI "
                f"{_fmt_ci(row['weighted_kendall_tau_b']['ci95'])}; weighted critic mean "
                f"{_fmt(row['critic_mean']['estimate'])} and weighted MC mean "
                f"{_fmt(row['mc_mean']['estimate'])} (point estimates). Changed-sequence "
                f"agreement uses n={changed['n_queries']} changed queries after excluding "
                f"n={changed['n_unchanged_queries_excluded']} unchanged queries: "
                + (
                    f"{_fmt_rate(changed['sign_agreement']['estimate'])}, 95% CI "
                    f"{_fmt_rate_ci(changed['sign_agreement']['ci95'])}."
                    if changed["sign_agreement"] is not None
                    else "unavailable because that subset is degenerate."
                )
            )
        lines.append("")

    aggregate = response["all_comparisons"]
    lines.extend(
        [
            "## Response search",
            "",
            f"Across n={aggregate['n_comparisons']} saved comparisons, strict sign "
            f"agreement is {aggregate['sign_agreement']['count']}/"
            f"{aggregate['sign_agreement']['denominator']} "
            f"({_fmt_rate(aggregate['sign_agreement']['rate'])}; point estimate). "
            f"Spearman is {_fmt(aggregate['spearman']['estimate'])}, 95% CI "
            f"{_fmt_ci(aggregate['spearman']['ci95'])}.",
            "",
        ]
    )
    if response["pre_entry_reconstruction"]["available"]:
        pre = response["pre_entry_estimable_subset"]
        lines.extend(
            [
                f"Exact prefix-entry status was reconstructed for all n="
                f"{response['pre_entry_reconstruction']['n_comparisons']} comparisons from "
                "saved trajectory states only. The restricted per-check "
                f"deltas use n={response['pre_entry_reconstruction']['total_pre_entry_queries']} "
                f"pre-entry queries in total (point count). One check has n=0 pre-entry "
                f"queries, so the requested all-18 aggregate is unavailable; across the "
                f"n={pre['n_comparisons']} estimable comparisons, sign agreement is "
                f"{pre['sign_agreement']['count']}/{pre['sign_agreement']['denominator']} "
                f"({_fmt_rate(pre['sign_agreement']['rate'])}; point estimate), and Spearman "
                f"is {_fmt(pre['spearman']['estimate'])}, 95% CI "
                f"{_fmt_ci(pre['spearman']['ci95'])}.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Pre-entry response-search stratification is unavailable; the exact saved-state "
                "reconstruction failure is recorded in results.json.",
                "",
            ]
        )
    lines.append("The 18 all-query and pre-entry-restricted deltas are:")
    lines.append("")
    for row in response["comparisons"]:
        text = (
            f"- `{row['name']}`: all-query n={row['n_queries']}, critic "
            f"{_fmt(row['critic_delta']['estimate'])}, MC {_fmt(row['mc_delta']['estimate'])} "
            "(point estimates)"
        )
        if row["pre_entry"]["delta_estimable"]:
            text += (
                f"; pre-entry n={row['pre_entry']['n_queries']}, critic "
                f"{_fmt(row['pre_entry']['critic_delta']['estimate'])}, MC "
                f"{_fmt(row['pre_entry']['mc_delta']['estimate'])} (point estimates)."
            )
        else:
            text += (
                f"; pre-entry n={row['pre_entry'].get('n_queries', 0)}, deltas "
                "unavailable (point count)."
            )
        lines.append(text)

    lines.extend(
        [
            "",
            "## Audit constraints and interpretation",
            "",
            "Computed new transitions: 0 (point count). New rollouts: 0 (point count). "
            "Checkpoint loads for inference: 0 (point count). Parameter updates: 0 "
            "(point count). Only saved arrays and JSON records were read; no pre-existing "
            "input artifact was overwritten, moved, or deleted.",
            "",
            "Two candidate pairs and 18 comparisons cannot establish a general property of "
            "the critic. This audit can only distinguish systematically reversed in the "
            "leverage region from unordered noise, and it is underpowered for the latter. "
            "It has no authority to conclude that the critic is fine.",
            "",
            "Protocol: `PROTOCOL.md`. Machine-readable values and all bootstrap-validity "
            "counts: `results.json`. Concise interpretation: `SUMMARY.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_summary(results: dict) -> str:
    continuation = results["continuation_precision"]
    response = results["response_search"]
    lines = [
        "# Contrast agreement summary",
        "",
        _observed_pattern(continuation),
        "",
    ]
    for name in PAIRS:
        row = continuation["comparisons"][name]["strata"]["pre_entry"]
        changed = row["sequence_changed_only"]
        lines.append(
            f"- `{name}` pre-entry: n={row['n_queries']}; agreement "
            f"{_fmt_rate(row['sign_agreement']['estimate'])} "
            f"({_fmt_rate_ci(row['sign_agreement']['ci95'])}); weighted Spearman "
            f"{_fmt(row['weighted_spearman']['estimate'])} "
            f"({_fmt_ci(row['weighted_spearman']['ci95'])}); weighted Kendall tau-b "
            f"{_fmt(row['weighted_kendall_tau_b']['estimate'])} "
            f"({_fmt_ci(row['weighted_kendall_tau_b']['ci95'])}); changed-sequence-only "
            f"n={changed['n_queries']}, agreement "
            f"{_fmt_rate(changed['sign_agreement']['estimate'])} "
            f"({_fmt_rate_ci(changed['sign_agreement']['ci95'])})."
        )
    all_response = response["all_comparisons"]
    lines.extend(
        [
            "",
            f"Response search all-query (n={all_response['n_comparisons']} comparisons): "
            f"agreement {all_response['sign_agreement']['count']}/"
            f"{all_response['sign_agreement']['denominator']} (point estimate); Spearman "
            f"{_fmt(all_response['spearman']['estimate'])} "
            f"({_fmt_ci(all_response['spearman']['ci95'])}).",
        ]
    )
    if response["pre_entry_reconstruction"]["available"]:
        pre = response["pre_entry_estimable_subset"]
        lines.append(
            f"Response search pre-entry estimable subset (n={pre['n_comparisons']} of 18 "
            f"comparisons; "
            f"n={response['pre_entry_reconstruction']['total_pre_entry_queries']} queries): "
            f"agreement {pre['sign_agreement']['count']}/{pre['sign_agreement']['denominator']} "
            f"(point estimate); Spearman {_fmt(pre['spearman']['estimate'])} "
            f"({_fmt_ci(pre['spearman']['ci95'])})."
        )
    else:
        lines.append("Response-search pre-entry analysis was unavailable and was skipped.")
    lines.extend(
        [
            "",
            "Computed new transitions, rollouts, checkpoint inference loads, and parameter "
            "updates are each 0 (point counts).",
            "",
            "Two candidate pairs and 18 comparisons cannot establish a general property of "
            "the critic. This audit can only distinguish systematically reversed in the "
            "leverage region from unordered noise, and it is underpowered for the latter. "
            "It has no authority to conclude that the critic is fine.",
            "",
            _observed_pattern(continuation),
            "",
        ]
    )
    return "\n".join(lines)


def run(*, replace_generated_draft: bool = False) -> dict:
    protocol = OUT / "PROTOCOL.md"
    outputs = [OUT / name for name in ("results.json", "REPORT.md", "SUMMARY.md", "audit.json", "manifest.json")]
    if not protocol.exists():
        raise FileNotFoundError("PROTOCOL.md must exist before analysis")
    if protocol.read_bytes() != NOTE.read_bytes():
        raise AssertionError("Artifact protocol must match the repository note")
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not replace_generated_draft:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    continuation = _analyze_continuation()
    response = _analyze_response()
    results = {
        "audit": {
            "read_only": True,
            "computed_new_transitions": 0,
            "new_rollouts": 0,
            "checkpoint_loads_for_inference": 0,
            "parameter_updates": 0,
            "pre_existing_input_artifacts_modified": 0,
        },
        "methods": {
            "continuation_bootstrap_seed": CONTINUATION_SEED,
            "response_bootstrap_seed": RESPONSE_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "interval_level": 0.95,
            "multiplicity_correction": "none; descriptive intervals",
        },
        "continuation_precision": continuation,
        "response_search": response,
        "interpretation": {
            "observed_pattern": _observed_pattern(continuation),
            "limits": (
                "Two candidate pairs and 18 comparisons cannot establish a general property "
                "of the critic. This audit only distinguishes systematic reversal in the "
                "leverage region from unordered noise and is underpowered for the latter; "
                "it cannot conclude that the critic is fine."
            ),
        },
    }
    report = _render_report(results)
    summary = _render_summary(results)
    audit = {
        "passed": True,
        "read_only": True,
        "computed_new_transitions": 0,
        "published_unstratified_means_reproduced": True,
        "response_json_point_estimates_reproduced": True,
        "response_pre_entry_reconstructed_from_saved_states": bool(
            response["pre_entry_reconstruction"]["available"]
        ),
        "checkpoint_loads_for_inference": 0,
        "parameter_updates": 0,
    }

    (OUT / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "SUMMARY.md").write_text(summary, encoding="utf-8")
    (OUT / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_files = [protocol, OUT / "results.json", OUT / "REPORT.md", OUT / "SUMMARY.md", OUT / "audit.json", NOTE, Path(__file__), Path("scripts/test_contrast_agreement.py")]
    manifest = {str(path).replace("\\", "/"): _sha(path) for path in manifest_files}
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results


if __name__ == "__main__":
    run()
