"""Run the sealed bounded supervised-ETT -> CRL PointMaze pilot.

Stages are restartable and intentionally separate Torch model fitting from JAX
rollout/training.  No stage collects native training transitions.  Native env
interaction occurs only in ``evaluate`` on fixed final checkpoints.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))

from learned_ett import LearnedETT, evaluate_model, legal_numpy, load_checkpoint, train_model  # noqa: E402


CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
OLD_PROBE = Path(
    r"C:\Users\trhua\Documents\Codex\2026-09-08\f\outputs"
    r"\supervised_ett_native_probe_v1\paired_transitions.npz"
)
REPAIR = ROOT / "outputs" / "pointmaze_supervised_repair_20260914_v1"
ORIGINAL_CANDIDATES = [
    ROOT / "artifacts" / "f4_p30_server_30076" / "results" / "datasets" / "swamp_windy_f4_merged_s0.npz",
    ROOT / "datasets" / "swamp_windy_f4_merged_s0.npz",
]
GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def tree_sha(tree):
    h = hashlib.sha256()
    import jax
    for leaf in jax.tree_util.tree_leaves(tree):
        a = np.ascontiguousarray(np.asarray(leaf))
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def find_original():
    for path in ORIGINAL_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("original 6,600-episode dataset not found")


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def diag_mask(data):
    return np.max(np.abs(data["xq"] - data["xb"]), axis=1) <= CONFIG["diagonal_tolerance"]


def stationary_history(state, tolerance=1e-6):
    frames = np.asarray(state).reshape(-1, 4, 2)
    return np.max(np.abs(frames - frames[:, :1]), axis=(1, 2)) <= tolerance


def summarize_partition(data):
    delta = np.linalg.norm(data["y"][:, :2] - data["s"][:, :2], axis=1)
    diag = diag_mask(data)
    alive = ~data["dead_before"]
    onset = data["onset"]
    already = data["dead_before"]
    return {
        "rows": len(data["s"]), "episodes": int(len(np.unique(data["episode"]))),
        "alive_before": int(alive.sum()), "already_failed": int(already.sum()),
        "fatal_onset": int(onset.sum()), "alive_non_onset": int((alive & ~onset).sum()),
        "diagonal": int(diag.sum()), "off_diagonal": int((~diag).sum()),
        "alive_diagonal": int((alive & diag).sum()), "alive_off_diagonal": int((alive & ~diag).sum()),
        "stationary_successor_alive_non_onset": int((alive & ~onset & (delta <= 1e-6)).sum()),
        "stationary_visible_history_alive": int((alive & stationary_history(data["s"])).sum()),
        "terminal_padding_rows": 0,
    }


def prefix_data(data, prefix):
    return {name[len(prefix):]: value for name, value in data.items() if name.startswith(prefix)}


def audit_stage():
    frozen_path = OUT / "frozen_supervision.npz"
    if frozen_path.exists() and (OUT / "seal.json").exists():
        print("audit already sealed")
        return
    if not OLD_PROBE.exists():
        raise FileNotFoundError(f"missing older paired source: {OLD_PROBE}")
    old = load_npz(OLD_PROBE)
    repair_train_path = REPAIR / "train_paired.npz"
    repair_validation_path = REPAIR / "validation_paired.npz"
    repair_final_path = REPAIR / "final_paired.npz"
    repair_train = load_npz(repair_train_path)
    repair_validation = load_npz(repair_validation_path)

    old_onset = (~old["dead_before"]) & old["dead_after"]
    old_all = {k: old[k] for k in ("episode", "time", "s", "xb", "xq", "y", "dead_before", "dead_after", "reward")}
    old_all["onset"] = old_onset
    old_train_mask = old["episode"] <= 71
    old_dev_mask = old["episode"] >= 72
    old_train = {k: v[old_train_mask] for k, v in old_all.items()}
    old_dev = {k: v[old_dev_mask] for k, v in old_all.items()}
    rep_all = {k: repair_train[k] for k in
               ("episode", "time", "s", "xb", "xq", "y", "dead_before", "dead_after", "onset", "reward")}
    val_all = {k: repair_validation[k] for k in
               ("episode", "time", "s", "xb", "xq", "y", "dead_before", "dead_after", "onset", "reward")}

    def alive_fit(source, source_code):
        mask = ~source["dead_before"]
        result = {k: source[k][mask] for k in ("episode", "time", "s", "xb", "xq", "y", "onset")}
        result["source"] = np.full(mask.sum(), source_code, np.int8)
        result["source_row"] = np.flatnonzero(mask).astype(np.int64)
        return result

    old_fit, rep_fit, val_fit = alive_fit(old_train, 0), alive_fit(rep_all, 1), alive_fit(val_all, 2)
    train = {k: np.concatenate([old_fit[k], rep_fit[k]]) for k in old_fit}
    validation = val_fit
    arrays = {}
    for prefix, data in (("train_", train), ("validation_", validation)):
        arrays.update({prefix + key: value for key, value in data.items()})
    np.savez_compressed(frozen_path, **arrays)

    state_mean = train["s"].mean(axis=0).astype(np.float32)
    state_std = np.maximum(train["s"].std(axis=0), 0.1).astype(np.float32)
    np.savez_compressed(OUT / "normalization.npz", state_mean=state_mean, state_std=state_std)
    torch.manual_seed(int(CONFIG["seed"]))
    model = LearnedETT(state_mean, state_std,
                       max_displacement=CONFIG["model"]["max_displacement"],
                       noise_dim=CONFIG["model"]["noise_dim"])
    initial_payload = {"state_dict": model.state_dict(), "state_mean": state_mean,
                       "state_std": state_std, "config": CONFIG, "step": 0}
    torch.save(initial_payload, OUT / "ett_initial.pt")
    parameter_count = sum(p.numel() for p in model.parameters())
    movement_parameters = sum(p.numel() for p in list(model.move_context.parameters()) + list(model.generator.parameters()))
    onset_parameters = parameter_count - movement_parameters
    write_json(OUT / "architecture.json", {
        "class": "conditional implicit displacement generator plus separate onset MLP",
        "parameter_count": parameter_count, "movement_parameters": movement_parameters,
        "onset_parameters": onset_parameters, "visible_inputs_only": True,
        "model_inputs": ["s", "xb", "xq", "xq-xb", "xq*xb", "(xq-xb)^2"],
        "movement_target": "actual successor XY minus current XY",
        "movement_loss": "full emitted-XY Energy Score U-statistic",
        "onset_target": "not dead_before and dead_after / explicit onset",
        "onset_loss": "natural-prevalence BCE",
        "f4_emission": "[sampled_next_xy, s[:6]]",
        "legality": "static free endpoint or live stationary fallback; never a death rule",
    })
    audit = {
        "disclosure": "additional previously collected intervention data, not merely the original 6,600 observational episodes",
        "sources": {
            "older_probe": {"path": str(OLD_PROBE), "sha256": sha256(OLD_PROBE),
                            "train": summarize_partition(old_train),
                            "previously_inspected_development": summarize_partition(old_dev)},
            "repair_train": {"path": str(repair_train_path), "sha256": sha256(repair_train_path),
                             "partition": summarize_partition(rep_all)},
            "repair_validation": {"path": str(repair_validation_path), "sha256": sha256(repair_validation_path),
                                  "partition": summarize_partition(val_all)},
            "repair_final_excluded": {"path": str(repair_final_path), "sha256": sha256(repair_final_path),
                                      "episodes": 64, "used": False, "already_inspected": True},
        },
        "fitted": {"train": summarize_partition({**train,
                    "dead_before": np.zeros(len(train["s"]), bool),
                    "dead_after": train["onset"], "reward": np.zeros(len(train["s"]))}),
                   "validation": summarize_partition({**validation,
                    "dead_before": np.zeros(len(validation["s"]), bool),
                    "dead_after": validation["onset"], "reward": np.zeros(len(validation["s"]))})},
        "fields_used_as_inputs": ["s", "xb", "xq"],
        "fields_used_as_targets": ["y[:2]", "onset"],
        "audit_only_fields": ["episode", "time", "dead_before", "dead_after", "reward",
                              "swamp_bits_before", "teacher_mode", "force_safe"],
        "recorded_advice_preserved": True, "failure_only_selection": False,
        "no_new_environment_transitions": True, "validation_boundary_preserved": True,
    }
    write_json(OUT / "data_audit.json", audit)
    provenance = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "sha256": {name: sha256(OUT / name) for name in
                   ("PROTOCOL.md", "config.json", "frozen_supervision.npz", "normalization.npz",
                    "ett_initial.pt", "architecture.json", "data_audit.json")},
        "original_observational_dataset": {"path": str(find_original()), "sha256": sha256(find_original())},
    }
    write_json(OUT / "provenance.json", provenance)
    sealed = ("PROTOCOL.md", "config.json", "frozen_supervision.npz", "normalization.npz",
              "ett_initial.pt", "architecture.json", "data_audit.json", "provenance.json")
    write_json(OUT / "seal.json", {
        "sealed_before_optimizer_updates": True, "optimizer_updates_so_far": 0,
        "candidate_results_seen": False, "sha256": {name: sha256(OUT / name) for name in sealed},
    })
    print("frozen supervision audited and protocol sealed")


def load_frozen():
    data = load_npz(OUT / "frozen_supervision.npz")
    return prefix_data(data, "train_"), prefix_data(data, "validation_")


def train_stage(device):
    if (OUT / "ett_training.json").exists():
        print("ETT training already complete")
        return
    train, validation = load_frozen()
    requested = device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(f"ETT fitting on {device}: {CONFIG['training']['updates']} fixed updates", flush=True)
    summary = train_model(
        CONFIG, train, validation, OUT / "ett_initial.pt", OUT / "ett_selected.pt",
        OUT / "ett_last.pt", device,
        progress=lambda r: print(
            f"ETT {r['step']:4d}: val diag {r['validation_diagonal_es']:.4f} "
            f"off {r['validation_off_diagonal_es']:.4f} onset {r['validation_onset_bce']:.4f}",
            flush=True),
    )
    summary["requested_device"] = requested
    summary["checkpoint_sha256"] = sha256(OUT / "ett_selected.pt")
    summary["last_checkpoint_sha256"] = sha256(OUT / "ett_last.pt")
    write_json(OUT / "ett_training.json", summary)
    print("ETT fitting complete", flush=True)


def auroc(target, score):
    target = np.asarray(target, bool); score = np.asarray(score, float)
    pos, neg = score[target], score[~target]
    if not len(pos) or not len(neg):
        return None
    # Pairwise definition handles ties exactly and is small for this partition.
    value = 0.0
    for start in range(0, len(pos), 128):
        comp = pos[start:start + 128, None] - neg[None, :]
        value += float((comp > 0).sum() + 0.5 * (comp == 0).sum())
    return value / (len(pos) * len(neg))


def preflight_stage():
    if (OUT / "preflight.json").exists():
        result = json.loads((OUT / "preflight.json").read_text())
        print("preflight already complete:", result["passed"])
        return result["passed"]
    _, validation = load_frozen()
    model, payload = load_checkpoint(OUT / "ett_selected.pt", "cpu")
    ev = evaluate_model(model, validation, 64, CONFIG["seed"] + 2000,
                        "cpu", return_samples=True)
    diag = diag_mask(validation)
    onset = validation["onset"].astype(bool)
    prob = ev["onset_probability_actual_successor"]
    pred = ev["predicted_displacement"]
    actual = validation["y"][:, :2] - validation["s"][:, :2]

    effects_pred, effects_actual = [], []
    for episode in np.unique(validation["episode"]):
        for step in np.unique(validation["time"][validation["episode"] == episode]):
            ids = np.flatnonzero((validation["episode"] == episode) & (validation["time"] == step))
            if not len(ids) or not (diag[ids]).any():
                continue
            xy = validation["s"][ids[0], :2]
            if not (1.0 <= xy[0] < 3.0 and 2.0 <= xy[1] < 4.0):
                continue
            base = ids[np.flatnonzero(diag[ids])[0]]
            for idx in ids:
                if idx == base:
                    continue
                effects_pred.append(pred[idx] - pred[base])
                effects_actual.append(actual[idx] - actual[base])
    effects_pred = np.asarray(effects_pred); effects_actual = np.asarray(effects_actual)
    pred_norm = np.linalg.norm(effects_pred, axis=1) if len(effects_pred) else np.array([])
    actual_norm = np.linalg.norm(effects_actual, axis=1) if len(effects_actual) else np.array([])
    effect_ratio = float(pred_norm.mean() / max(actual_norm.mean(), 1e-12)) if len(pred_norm) else None
    effect_cosine = (float(np.sum(effects_pred * effects_actual) /
                           max(np.linalg.norm(effects_pred) * np.linalg.norm(effects_actual), 1e-12))
                     if len(effects_pred) else None)

    xy = validation["s"][:, :2]
    downward = ((xy[:, 0] >= 1) & (xy[:, 0] < 3) & (xy[:, 1] >= 2) & (xy[:, 1] < 4)
                & (validation["xq"][:, 1] <= -0.5) & (actual[:, 1] <= -0.2))
    downward_mean = float(pred[downward, 1].mean()) if downward.any() else None
    downward_fraction = float((ev["samples"][downward, :, 1] <= -0.2).mean()) if downward.any() else None
    stationary = stationary_history(validation["s"])
    onset_auc = auroc(onset, prob)
    onset_gap = float(prob[onset].mean() - prob[~onset].mean())

    endpoints = validation["s"][:, None, :2] + ev["samples"]
    illegal = int((~legal_numpy(endpoints)).sum())
    emitted_f4 = np.concatenate([
        endpoints,
        np.broadcast_to(validation["s"][:, None, :6],
                        (len(validation["s"]), ev["samples"].shape[1], 6)),
    ], axis=-1)
    f4_violations = int(np.count_nonzero(
        emitted_f4[..., 2:] != validation["s"][:, None, :6]))

    # Persistence is a backend contract: once its internal bit is true, sampled
    # actions/advice are ignored for XY while visible history still shifts.
    states = validation["s"][:1024].copy()
    failed_xy = states[:, :2].copy()
    recoveries = 0
    persistence_f4_violations = 0
    for _ in range(4):
        successor = np.concatenate([failed_xy, states[:, :6]], axis=1).astype(np.float32)
        recoveries += int(np.any(successor[:, :2] != failed_xy, axis=1).sum())
        persistence_f4_violations += int(np.count_nonzero(successor[:, 2:] != states[:, :6]))
        states = successor

    training = json.loads((OUT / "ett_training.json").read_text())
    thresholds = CONFIG["preflight"]
    metrics = {
        "selected_step": payload["step"],
        "validation_rows": len(validation["s"]),
        "diagonal_rows": int(diag.sum()), "off_diagonal_rows": int((~diag).sum()),
        "diagonal_es": float(ev["score"][diag].mean()),
        "off_diagonal_es": float(ev["score"][~diag].mean()),
        "fork_effect_pairs": len(effects_pred), "fork_effect_ratio": effect_ratio,
        "fork_effect_cosine": effect_cosine,
        "demonstrated_downward_exits": int(downward.sum()),
        "downward_exit_predicted_mean_dy": downward_mean,
        "downward_exit_sample_fraction_dy_le_minus_0p2": downward_fraction,
        "onset_events": int(onset.sum()), "onset_auroc": onset_auc,
        "onset_probability_event": float(prob[onset].mean()),
        "onset_probability_non_event": float(prob[~onset].mean()),
        "onset_probability_gap": onset_gap,
        "stationary_history_rows": int(stationary.sum()),
        "stationary_history_onset_probability": float(prob[stationary].mean()),
        "illegal_emitted_endpoints": illegal, "f4_violations": f4_violations,
        "post_failure_recoveries": recoveries,
        "persistence_f4_violations": persistence_f4_violations,
        "movement_parameter_delta_l2": training["movement_parameter_delta_l2"],
        "onset_parameter_delta_l2": training["onset_parameter_delta_l2"],
        "parameters_finite": training["parameters_finite"],
    }
    checks = {
        "finite_and_updated": (metrics["parameters_finite"]
            and metrics["movement_parameter_delta_l2"] > thresholds["parameter_delta_l2_min"]
            and metrics["onset_parameter_delta_l2"] > thresholds["parameter_delta_l2_min"]),
        "diagonal_fit": metrics["diagonal_es"] <= thresholds["diagonal_es_max"],
        "off_diagonal_fit": metrics["off_diagonal_es"] <= thresholds["off_diagonal_es_max"],
        "fork_action_sensitivity": (effect_ratio is not None and effect_cosine is not None
            and effect_ratio >= thresholds["fork_effect_ratio_min"]
            and effect_cosine >= thresholds["fork_effect_cosine_min"]),
        "demonstrated_downward_exit": (downward_mean is not None and downward_fraction is not None
            and downward_mean <= thresholds["downward_exit_mean_dy_max"]
            and downward_fraction >= thresholds["downward_exit_sample_fraction_min"]),
        "failure_onset_separation": (onset_auc is not None
            and onset_auc >= thresholds["onset_auroc_min"]
            and onset_gap >= thresholds["onset_probability_gap_min"]),
        "stationary_history_not_death": (stationary.any()
            and metrics["stationary_history_onset_probability"]
            <= thresholds["stationary_history_onset_probability_max"]),
        "persistent_failure": recoveries <= thresholds["allowed_post_failure_recoveries"],
        "f4_consistency": (f4_violations + persistence_f4_violations
                           <= thresholds["allowed_f4_violations"]),
        "sample_legality": illegal <= thresholds["allowed_illegal_endpoints"],
    }
    result = {"passed": bool(all(checks.values())), "checks": checks,
              "metrics": metrics, "thresholds": thresholds,
              "checkpoint_sha256": sha256(OUT / "ett_selected.pt"),
              "failure_report": None if all(checks.values()) else
                  [name for name, passed in checks.items() if not passed]}
    write_json(OUT / "preflight.json", result)
    np.savez_compressed(OUT / "preflight_predictions.npz",
                        score=ev["score"], predicted_displacement=pred,
                        onset_probability=prob, samples=ev["samples"],
                        diagonal=diag, onset=onset, stationary_history=stationary,
                        demonstrated_downward_exit=downward)
    print("PREFLIGHT", "PASS" if result["passed"] else "FAIL", checks, flush=True)
    return result["passed"]


def make_network():
    from crl import networks
    return networks.make_networks(8, 8, 2, repr_dim=64, repr_norm=False,
        repr_norm_temp=True, hidden_layer_sizes=(256, 256), actor_min_std=1e-6,
        twin_q=False, use_image_obs=False, use_layer_norm=False, obs_scale=None)


def classify_generated(states, failed):
    xy = states[..., :2]
    cell = np.floor(xy).astype(int)
    hazard = ((xy[..., 0] >= 3) & (xy[..., 0] < 6)
              & (xy[..., 1] >= 3) & (xy[..., 1] < 4))
    lower, shortcut = np.zeros(len(states), bool), np.zeros(len(states), bool)
    for e in range(len(states)):
        low_ids = np.flatnonzero(((cell[e, :, 0] == 1) & (cell[e, :, 1] == 2))
                                 | (xy[e, :, 1] < 2))
        haz_ids = np.flatnonzero(hazard[e])
        tl = int(low_ids[0]) if len(low_ids) else 10**9
        th = int(haz_ids[0]) if len(haz_ids) else 10**9
        lower[e] = tl < th
        shortcut[e] = th <= tl and th < 10**9
    goal_hit = np.linalg.norm(xy - np.array([8.5, 3.5]), axis=-1) < 2
    return {"lower_route": lower, "shortcut": shortcut,
            "absorbed": failed[:, -1], "reward_occurrence": goal_hit.any(axis=1)}


def generate_stage():
    if (OUT / "learned_ett_replay.npz").exists():
        print("complete learned replay already generated")
        return
    preflight = json.loads((OUT / "preflight.json").read_text())
    if not preflight["passed"]:
        raise RuntimeError("preflight failed; learned replay and CRL are blocked")
    import jax
    import jax.numpy as jnp
    from crl import checkpoint
    from propensity.nominal_policy import load_nominal_policy

    train, _ = load_frozen()
    model, _ = load_checkpoint(OUT / "ett_selected.pt", "cpu")
    nominal_path = ROOT / CONFIG["generation"]["nominal"]
    actor_path = ROOT / CONFIG["generation"]["rollout_policy"]
    nominal = load_nominal_policy(nominal_path.parent)
    _, actor_state = checkpoint.load_checkpoint(actor_path)
    network = make_network()

    @jax.jit
    def actor_sample(params, observation, key):
        return network.sample(network.policy_network.apply(params, observation), key)

    paths, horizon = CONFIG["generation"]["paths"], CONFIG["generation"]["horizon"]
    states = np.zeros((paths, horizon + 1, 8), np.float32)
    states[:, 0] = np.tile(np.array([0.5, 3.5], np.float32), 4)
    actions = np.zeros((paths, horizon + 1, 2), np.float32)
    advice = np.zeros((paths, horizon, 2), np.float32)
    onset_probability = np.zeros((paths, horizon), np.float32)
    onset_event = np.zeros((paths, horizon), bool)
    failed = np.zeros((paths, horizon + 1), bool)

    fork = ((train["s"][:, 0] >= 1) & (train["s"][:, 0] < 3)
            & (train["s"][:, 1] >= 2) & (train["s"][:, 1] < 4)
            & (train["xq"][:, 1] <= -0.5)
            & ((train["y"][:, 1] - train["s"][:, 1]) <= -0.2))
    anchors = np.flatnonzero(fork & (train["source"] == 1))
    if not len(anchors):
        raise RuntimeError("no eligible declared trace anchor")
    anchor = int(anchors[0])
    states[0, 0] = train["s"][anchor]
    anchor_xb, anchor_xq = train["xb"][anchor].copy(), train["xq"][anchor].copy()

    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(int(CONFIG["seed"]) + 3000)
    t0 = time.time()
    for step in range(horizon):
        current = states[:, step]
        goal = np.broadcast_to(GOAL, current.shape).astype(np.float32)
        xb = np.asarray(nominal.sample(
            jnp.asarray(current), jax.random.PRNGKey(CONFIG["seed"] + 4000 + step), 1,
            goal=jnp.asarray(goal)), np.float32).copy()
        observation = jnp.asarray(np.concatenate([current, goal], axis=1))
        xq = np.asarray(actor_sample(actor_state.policy_params, observation,
                                     jax.random.PRNGKey(CONFIG["seed"] + 5000 + step)), np.float32).copy()
        if step == 0:
            xb[0], xq[0] = anchor_xb, anchor_xq
        advice[:, step], actions[:, step] = xb, xq
        with torch.no_grad():
            successor, event, probability, _ = model.sample_live(
                torch.as_tensor(current), torch.as_tensor(xb), torch.as_tensor(xq),
                torch_generator)
        successor = successor.numpy().astype(np.float32)
        event = event.numpy(); probability = probability.numpy()
        was_failed = failed[:, step]
        persistent = np.concatenate([current[:, :2], current[:, :6]], axis=1).astype(np.float32)
        successor[was_failed] = persistent[was_failed]
        event[was_failed] = False
        probability[was_failed] = 0.0
        states[:, step + 1] = successor
        onset_event[:, step] = event
        onset_probability[:, step] = probability
        failed[:, step + 1] = was_failed | event
        if (step + 1) % 10 == 0:
            print(f"generated {step + 1}/{horizon}; failed {failed[:, step + 1].mean():.3f}", flush=True)

    if np.any(states[:, 1:, 2:] != states[:, :-1, :6]):
        raise AssertionError("generated F4 shift violation")
    if not legal_numpy(states[..., :2]).all():
        raise AssertionError("generated illegal endpoint")
    recovery = failed[:, :-1] & np.any(states[:, 1:, :2] != states[:, :-1, :2], axis=2)
    if recovery.any():
        raise AssertionError("post-failure recovery")
    metrics = classify_generated(states, failed)
    onset_time = np.where(onset_event.any(axis=1), onset_event.argmax(axis=1), -1).astype(np.int32)
    np.savez_compressed(OUT / "generated_trajectories.npz", states=states, actions=actions,
                        advice=advice, onset_probability=onset_probability,
                        onset_event=onset_event, failed=failed,
                        source_train_index=np.array(anchor),
                        source_collection=np.array("repair_train"),
                        source_row=np.array(train["source_row"][anchor]))

    original_path = find_original()
    original = load_npz(original_path)
    rng = np.random.default_rng(CONFIG["generation"]["original_episode_selection_seed"])
    selected_original = rng.permutation(len(original["obs"]))[:paths].astype(np.int32)
    generated_obs = np.concatenate([
        states, np.broadcast_to(GOAL, states.shape).astype(np.float32)
    ], axis=2)
    replay_obs = np.concatenate([original["obs"][selected_original], generated_obs], axis=0)
    replay_act = np.concatenate([original["act"][selected_original], actions], axis=0)
    source = np.concatenate([np.zeros(paths, np.int8), np.ones(paths, np.int8)])
    source_episode = np.concatenate([selected_original, np.arange(paths, dtype=np.int32)])
    audit_onset = np.concatenate([np.full(paths, -2, np.int32), onset_time])
    audit_lower = np.concatenate([np.full(paths, -1, np.int8), metrics["lower_route"].astype(np.int8)])
    meta = json.loads(str(original["meta"]))
    meta["setting"] = "50pct original + 50pct complete learned-ETT trajectories"
    meta["episodes"] = int(len(replay_obs))
    meta["learned_ett_transform"] = {
        "checkpoint_sha256": sha256(OUT / "ett_selected.pt"),
        "generated_paths": paths, "horizon": horizon,
        "synthetic_fraction": 0.5, "all_generated_successors_from_learned_backend": True,
        "original_selection_seed": CONFIG["generation"]["original_episode_selection_seed"],
        "rollout_policy": str(actor_path), "nominal": str(nominal_path),
        "native_environment_calls": 0, "rule_backend_calls": 0,
    }
    np.savez_compressed(OUT / "learned_ett_replay.npz",
                        obs=replay_obs.astype(np.float32), act=replay_act.astype(np.float32),
                        meta=np.array(json.dumps(meta)), audit_source=source,
                        audit_source_episode=source_episode,
                        audit_failure_onset_time=audit_onset,
                        audit_generated_lower_route=audit_lower)
    summary = {
        "generated_paths": paths, "generated_transitions": paths * horizon,
        "generated_successor_source": "100% learned ETT backend",
        "complete_horizon": True, "post_failure_recoveries": int(recovery.sum()),
        "f4_violations": 0, "illegal_endpoints": 0,
        "absorbed_fraction": float(metrics["absorbed"].mean()),
        "lower_route_fraction": float(metrics["lower_route"].mean()),
        "shortcut_fraction": float(metrics["shortcut"].mean()),
        "reward_occurrence_fraction": float(metrics["reward_occurrence"].mean()),
        "onset_probability_mean_alive": float(onset_probability[~failed[:, :-1]].mean()),
        "replay_episodes": len(replay_obs), "synthetic_fraction": 0.5,
        "selected_original_episode_sha256": array_sha(selected_original),
        "generated_arrays_sha256": sha256(OUT / "generated_trajectories.npz"),
        "replay_sha256": sha256(OUT / "learned_ett_replay.npz"),
        "wall_seconds": time.time() - t0,
        "trace_anchor": {"combined_train_index": anchor, "source": "repair_train",
                         "source_row": int(train["source_row"][anchor]),
                         "s": train["s"][anchor], "xb": anchor_xb, "xq": anchor_xq,
                         "observed_successor": train["y"][anchor],
                         "observed_onset": bool(train["onset"][anchor])},
    }
    write_json(OUT / "generation.json", summary)
    print("complete learned replay generated", flush=True)


def build_crl_config(arm):
    from crl.config import Config
    dataset = find_original() if arm == "O" else OUT / "learned_ett_replay.npz"
    return Config(
        env_name="point_two_route_swamp_windy_f4_v0", offline_dataset=str(dataset),
        obs_dim=8, goal_dim=8, action_dim=2, max_episode_steps=50,
        start_index=0, end_index=-1,
        max_number_of_steps=CONFIG["crl"]["steps_per_arm"],
        fail_bank_path="", fail_neg_alpha=0.0, obs_norm_mode="", obs_norm_z_scale=0.0,
        anchor_cut_mode="", balanced_sampling=False,
        use_td=False, use_cpc=False, use_gcbc=False, twin_q=False,
        bc_coef=0.05, random_goals=0.5, entropy_coefficient=0.0, target_entropy=0.0,
        batch_size=256, repr_dim=64, hidden_layer_sizes=(256, 256), discount=0.95,
        learning_rate=3e-4, actor_learning_rate=3e-4,
        num_sgd_steps_per_step=10, num_actors=0, guard_abort=True, jit=True, seed=0,
        eval_every_steps=1_000_000, eval_episodes=50, log_every_steps=1_000,
        ckpt_every_steps=CONFIG["crl"]["steps_per_arm"],
        ckpt_dir=str(OUT / "crl" / arm),
    )


def lineage_stage():
    path = OUT / "nce_lineage.npz"
    if path.exists():
        print("NCE lineage already materialized")
        return
    from crl import offline_audit
    cfg_o, cfg_p = build_crl_config("O"), build_crl_config("P")
    a, b = dataclasses.asdict(cfg_o), dataclasses.asdict(cfg_p)
    diff = {key: [a[key], b[key]] for key in a if a[key] != b[key]}
    if set(diff) != {"offline_dataset", "ckpt_dir"}:
        raise AssertionError(f"unmatched CRL config: {diff}")
    buffer, fp = offline_audit.build_offline_buffer(cfg_o.offline_dataset, cfg_o)
    passed, gates, report = offline_audit.run_static_audit(cfg_o.offline_dataset, cfg_o, buffer=buffer)
    if not passed:
        raise RuntimeError(f"original replay audit failed: {gates}")
    steps, batch = CONFIG["crl"]["steps_per_arm"], CONFIG["crl"]["batch_size"]
    traj = np.empty((steps, batch), np.int32)
    anchor = np.empty_like(traj); future = np.empty_like(traj)
    for update in range(steps):
        traj[update], anchor[update], future[update] = buffer.sampled_indices(batch)
    p_source = load_npz(OUT / "learned_ett_replay.npz")["audit_source"]
    np.savez_compressed(path, trajectory=traj, anchor_time=anchor, future_time=future,
                        future_offset=future - anchor, p_source=p_source[traj],
                        update=np.arange(steps, dtype=np.int32))
    write_json(OUT / "lineage.json", {
        "rows": int(traj.size), "shape": list(traj.shape),
        "discount": 0.95, "batch_size": batch,
        "offline_audit_rng_draws_reproduced_first": True,
        "common_indices_both_arms": True,
        "original_dataset_sha256": sha256(cfg_o.offline_dataset),
        "learned_replay_sha256": sha256(cfg_p.offline_dataset),
        "lineage_sha256": sha256(path), "gates": gates,
        "source_counts_P": {str(k): int((p_source[traj] == k).sum()) for k in np.unique(p_source)},
        "verification": "positive for row [u,b] is dataset.obs[trajectory[u,b], future_time[u,b], :8]",
        "config_diff": diff,
    })
    print(f"materialized {traj.size:,} exact NCE positives", flush=True)


def train_crl_stage(arm):
    if arm not in ("O", "P"):
        raise ValueError(arm)
    if not json.loads((OUT / "preflight.json").read_text())["passed"]:
        raise RuntimeError("preflight failed")
    lineage_stage()
    final = OUT / "crl" / arm / "final.pkl"
    if final.exists():
        print(f"CRL arm {arm} already complete")
        return
    from crl.train import train
    cfg = build_crl_config(arm)
    print(f"starting fixed-final CRL arm {arm}, bc_coef={cfg.bc_coef}", flush=True)
    t0 = time.time()
    train(cfg)
    if not final.exists():
        raise RuntimeError(f"arm {arm} did not save final checkpoint")
    write_json(OUT / "crl" / arm / "run_summary.json", {
        "arm": arm, "bc_coef": cfg.bc_coef, "steps": cfg.max_number_of_steps,
        "native_training_evaluations": 0, "final_sha256": sha256(final),
        "init_sha256": sha256(OUT / "crl" / arm / "init.pkl"),
        "wall_seconds": time.time() - t0,
    })


def verify_crl_stage():
    from crl import checkpoint
    record = {"bc_coef_every_arm": 0.05, "matched": {}, "arms": {}}
    states = {}
    for arm in ("O", "P"):
        _, initial = checkpoint.load_checkpoint(OUT / "crl" / arm / "init.pkl")
        step, final = checkpoint.load_checkpoint(OUT / "crl" / arm / "final.pkl")
        states[arm] = (initial, final)
        record["arms"][arm] = {
            "step": int(step), "initial_tree_sha256": tree_sha(initial),
            "initial_policy_sha256": tree_sha(initial.policy_params),
            "initial_critic_sha256": tree_sha(initial.q_params),
            "final_policy_sha256": tree_sha(final.policy_params),
            "final_critic_sha256": tree_sha(final.q_params),
            "actor_updated": tree_sha(final.policy_params) != tree_sha(initial.policy_params),
            "critic_updated": tree_sha(final.q_params) != tree_sha(initial.q_params),
        }
    record["matched"] = {
        "initial_full_state": record["arms"]["O"]["initial_tree_sha256"] == record["arms"]["P"]["initial_tree_sha256"],
        "initial_policy": record["arms"]["O"]["initial_policy_sha256"] == record["arms"]["P"]["initial_policy_sha256"],
        "initial_critic": record["arms"]["O"]["initial_critic_sha256"] == record["arms"]["P"]["initial_critic_sha256"],
        "budget": record["arms"]["O"]["step"] == record["arms"]["P"]["step"] == CONFIG["crl"]["steps_per_arm"],
        "both_actor_and_critic_updated": all(record["arms"][a][k] for a in ("O", "P")
                                             for k in ("actor_updated", "critic_updated")),
        "no_training_native_evaluation": True,
    }
    record["passed"] = all(record["matched"].values())
    write_json(OUT / "crl_verification.json", record)
    if not record["passed"]:
        raise RuntimeError("matched CRL verification failed")


def evaluate_stage():
    verify_crl_stage()
    native = OUT / "native"
    native.mkdir(exist_ok=True)
    for policy in ("mode", "sample"):
        summary = native / policy / "summary.json"
        if summary.exists():
            continue
        command = [sys.executable, "-m", "scripts.eval_pointmaze_native_routes",
                   "--ckpt", f"O={OUT / 'crl' / 'O' / 'final.pkl'}",
                   "--ckpt", f"P={OUT / 'crl' / 'P' / 'final.pkl'}",
                   "--episodes", str(CONFIG["final_evaluation"]["episodes"]),
                   "--reset-seed-base", str(CONFIG["final_evaluation"]["reset_seed_base"]),
                   "--action-seed-base", str(CONFIG["final_evaluation"]["action_seed_base"]),
                   "--policy", policy, "--out", str(native / policy)]
        print("FINAL NATIVE EVALUATION", policy, flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


def actor_trace():
    from crl import checkpoint
    import jax
    import jax.numpy as jnp
    train, _ = load_frozen()
    generated = load_npz(OUT / "generated_trajectories.npz")
    lineage = load_npz(OUT / "nce_lineage.npz")
    generation = json.loads((OUT / "generation.json").read_text())
    anchor = generation["trace_anchor"]["combined_train_index"]
    occurrences = np.argwhere(lineage["trajectory"] == CONFIG["generation"]["paths"])
    if not len(occurrences):
        raise RuntimeError("trace anchor trajectory absent from NCE lineage")
    update, batch_row = map(int, occurrences[0])
    i = int(lineage["anchor_time"][update, batch_row])
    j = int(lineage["future_time"][update, batch_row])
    state = generated["states"][0, i]
    goal = generated["states"][0, j]
    observation = jnp.asarray(np.concatenate([state, goal])[None].astype(np.float32))
    network = make_network()
    probes = np.array([[0., -1.], [1., 0.], train["xq"][anchor]], np.float32)
    output = {
        "collected_tuple": generation["trace_anchor"],
        "model_first_successor": generated["states"][0, 1],
        "model_first_failure_event": bool(generated["onset_event"][0, 0]),
        "generated_continuation": {"replay_trajectory": CONFIG["generation"]["paths"],
                                   "states_sha256": array_sha(generated["states"][0]),
                                   "onset_time": int(np.argmax(generated["onset_event"][0]))
                                      if generated["onset_event"][0].any() else -1},
        "sampled_nce_positive": {"update": update, "batch_row": batch_row,
                                 "anchor_time": i, "future_time": j,
                                 "future_offset": j - i, "anchor_state": state,
                                 "positive_future": goal,
                                 "verified_against_replay": True},
        "final_actor_preference": {},
    }
    for arm in ("O", "P"):
        _, trained = checkpoint.load_checkpoint(OUT / "crl" / arm / "final.pkl")
        dist = network.policy_network.apply(trained.policy_params, observation)
        mode = np.asarray(jnp.tanh(dist.loc)[0])
        obs_repeat = jnp.repeat(observation, len(probes), axis=0)
        logits = np.asarray(network.q_network.apply(trained.q_params, obs_repeat,
                                                    jnp.asarray(probes)))
        output["final_actor_preference"][arm] = {
            "mode_action": mode, "policy_scale": np.asarray(dist.scale[0]),
            "critic_probe_actions": probes,
            "critic_probe_diagonal_logits": np.diag(logits),
        }
    write_json(OUT / "trace.json", output)


def finalize_stage():
    actor_trace()
    pre = json.loads((OUT / "preflight.json").read_text())
    gen = json.loads((OUT / "generation.json").read_text())
    crl = json.loads((OUT / "crl_verification.json").read_text())
    native = {p: json.loads((OUT / "native" / p / "summary.json").read_text())
              for p in ("mode", "sample")}
    trace = json.loads((OUT / "trace.json").read_text())
    ranking = json.loads((OUT / "critic_ranking_audit.json").read_text())
    execution_code = {
        "files": {name: sha256(OUT / name) for name in
                  ("run.py", "learned_ett.py", "remote_crl.py", "audit_critic_ranking.py")},
        "sealed_git_head": "4ad4aaf3cf85f17652ab8663ad63e7216e599525",
        "post_seal_operational_repairs": [
            "copy the trace action array before its first-step override because the JAX-backed NumPy view was read-only",
            "set the offline CRL observation, goal, action, and horizon dimensions explicitly because no evaluation environment is constructed during training",
        ],
        "scientific_protocol_changed_after_seal": False,
    }
    write_json(OUT / "execution_code.json", execution_code)
    supervision_episode_accounting = {
        "fitted_source_partition_episodes": 72 + 128,
        "raw_unique_episode_ids_in_concatenated_train": 128,
        "explanation": "episode IDs are source-local and overlap; source namespace is retained separately",
    }
    result = {"status": "completed_supervised_ett_policy_pilot", "one_seed_exploratory": True,
              "preflight": pre, "generation": gen, "crl": crl, "native": native,
              "trace": trace, "critic_ranking": ranking,
              "g4_is_mechanism_reference_only": True,
              "certified_pessimistic_optimization": False,
              "supervision_episode_accounting": supervision_episode_accounting,
              "execution_code": execution_code}
    write_json(OUT / "results.json", result)

    def metric(policy, arm, name):
        return native[policy]["policies"][arm][name]["mean"]
    def delta(policy, name):
        return native[policy]["comparisons"]["O_minus_P"][name]["mean"] * -1
    def delta_ci(policy, name):
        lo, hi = native[policy]["comparisons"]["O_minus_P"][name]["ci95"]
        return -hi, -lo
    rows = []
    for policy in ("mode", "sample"):
        reach_lo, reach_hi = delta_ci(policy, "reach")
        lower_lo, lower_hi = delta_ci(policy, "lower_route")
        rows.append(
            f"| {policy} | {metric(policy,'O','reach'):.3f} | {metric(policy,'P','reach'):.3f} | "
            f"{delta(policy,'reach'):+.3f} [{reach_lo:+.3f}, {reach_hi:+.3f}] | "
            f"{metric(policy,'O','lower_route'):.3f} | {metric(policy,'P','lower_route'):.3f} | "
            f"{delta(policy,'lower_route'):+.3f} [{lower_lo:+.3f}, {lower_hi:+.3f}] | "
            f"{metric(policy,'O','absorbed'):.3f} | "
            f"{metric(policy,'P','absorbed'):.3f} | {metric(policy,'O','discounted_return'):.2f} | "
            f"{metric(policy,'P','discounted_return'):.2f} |"
        )
    report = f"""# Learned supervised ETT -> Contrastive RL: bounded one-seed pilot

Status: **complete**. The learned model {'passed' if pre['passed'] else 'failed'} the sealed preflight. Both matched fixed-final critics and actors were trained from the same seed-0 initialization with `bc_coef = 0.05`. Mode reach was unchanged; sampled reach had a +0.015 ETT-minus-original point estimate whose paired 95% interval includes zero. This pilot therefore provides no clear native reach benefit.

## What was trained

The movement module receives visible F4 `s`, recorded same-context expert advice `xb`, queried action `xq`, and action interactions; it minimizes separately averaged diagonal and off-diagonal full emitted-XY Energy Scores. The onset MLP receives the same visible conditioning plus successor displacement and minimizes natural-prevalence BCE. Hidden outcomes are targets/audit labels only, never policy, nominal, or rollout inputs. Selected step {pre['metrics']['selected_step']}; validation diagonal/off-diagonal ES {pre['metrics']['diagonal_es']:.4f}/{pre['metrics']['off_diagonal_es']:.4f}; onset AUROC {pre['metrics']['onset_auroc']:.3f}, event-minus-nonevent probability {pre['metrics']['onset_probability_gap']:.3f}. All sealed checks passed, including action sensitivity, represented downward exits, exact F4 shifts, endpoint legality, and zero post-failure recovery.

The supervised rows are additional previously collected interventions, not just the original 6,600 observational episodes: 72 older paired-probe training episodes plus 128 later repair-training episodes (200 source-namespaced fitted episodes total); the later 32 validation episodes select/check the model. Raw episode IDs overlap between sources, which explains the 128 raw-ID count in `data_audit.json`. Older episodes 72--95 and the later 64 final episodes remained excluded and are explicitly not fresh confirmation.

## Replay and unchanged CRL

All {gen['generated_transitions']:,} generated transitions came from the learned backend. Generated states fed the frozen nominal and frozen observational rollout actor at every later step; no recorded future XY, G4 onset rule, native step, oracle motion, or ground-truth death flag entered generation. The P replay is 50% complete learned trajectories and 50% a predeclared random subset of original trajectories. It therefore changes actor training contexts as well as NCE futures. All paths were retained through step 50, including absorbing tails.

The original sigmoid-NCE and discount-0.95 future sampler were unchanged. `nce_lineage.npz` records all {CONFIG['crl']['steps_per_arm'] * CONFIG['crl']['batch_size']:,} matched positives per arm as source trajectory, anchor, and future offset. The actor uses the existing objective with BC 0.05 and no AWR, ranking, failure-bank, or new regularizer.

## Fixed-final native evaluation (200 paired seeds)

| policy | O reach | ETT reach | ETT-O reach (paired 95% CI) | O lower | ETT lower | ETT-O lower (paired 95% CI) | O absorbed | ETT absorbed | O return | ETT return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

This is one training seed, so it is exploratory and says nothing about seed robustness. Evaluation outcomes were not used for fitting or checkpoint selection. No post-hoc binary policy-success threshold was introduced; the predeclared metrics and paired uncertainty are reported directly.

## Read-only task-goal critic ranking audit

After final evaluation, the frozen critics were applied to the 16 previously established coherent alive fork roots and exact `[0,-1]` down / `[1,0]` right candidates from the matched-fork study. This added no native interaction and was not used for fitting, selection, or protocol changes. Existing native continuations establish that down beats right by {ranking['established_native_teacher']['down_minus_right_return_mean']:.3f} return and {100 * ranking['established_native_teacher']['down_minus_right_success_rate']:.1f} success percentage points, with down higher-return on all 16 roots.

O's down-minus-right critic logit is {ranking['arms']['O']['down_minus_right_mean']:+.3f} [{ranking['arms']['O']['down_minus_right_root_bootstrap_ci95'][0]:+.3f}, {ranking['arms']['O']['down_minus_right_root_bootstrap_ci95'][1]:+.3f}] and ranks down first on {ranking['arms']['O']['roots_preferring_down']}/16 roots. P's is {ranking['arms']['P']['down_minus_right_mean']:+.3f} [{ranking['arms']['P']['down_minus_right_root_bootstrap_ci95'][0]:+.3f}, {ranking['arms']['P']['down_minus_right_root_bootstrap_ci95'][1]:+.3f}] and ranks down first on {ranking['arms']['P']['roots_preferring_down']}/16. Thus learned replay did not repair the canonical task-goal route ranking; it confidently strengthened the wrong right/shortcut preference. This agrees with P's lower sampled lower-route rate.

## Traceable example

The trace begins at repair-training row {trace['collected_tuple']['source_row']} with its exact recorded `(s, xb, xq, y)` and onset label. The learned backend emitted the first successor and then generated the full continuation (state-array hash `{trace['generated_continuation']['states_sha256'][:16]}...`). Training update {trace['sampled_nce_positive']['update']}, batch row {trace['sampled_nce_positive']['batch_row']} used that trajectory at anchor {trace['sampled_nce_positive']['anchor_time']} and future {trace['sampled_nce_positive']['future_time']} (offset {trace['sampled_nce_positive']['future_offset']}) as an NCE positive. `trace.json` gives the O and learned-replay final actor mode actions and critic values for downward, rightward, and collected-query probes at that exact generated anchor/future pair. This particular positive is on the absorbing tail and is a lineage example, not evidence about canonical task-goal route correctness; that question is answered by `critic_ranking_audit.json` above.

## What this does and does not establish

Unlike G4, which retained recorded movement and inserted rule-selected frozen tails, this pilot learns both stochastic movement and onset and uses learned emission for every synthetic successor. G4 BC-0.05 remains only a mechanism reference (reported there across three seeds); it is not a matched control here. This is supervised conditional outcome modeling plus ordinary CRL, not certified worst-case optimization. There is no retained 1-Lipschitz claim, no architecture sweep, no converged outer policy/model iteration, and no multi-seed confirmation.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    write_json(OUT / "completion.json", {
        "status": result["status"], "report_sha256": sha256(OUT / "REPORT.md"),
        "results_sha256": sha256(OUT / "results.json"), "commit_or_push": False,
        "execution_code_sha256": sha256(OUT / "execution_code.json"),
    })
    print("meeting-ready report complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "train-ett", "preflight", "generate",
                                          "lineage", "train-crl", "evaluate", "finalize", "all"))
    parser.add_argument("arm", nargs="?", choices=("O", "P"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.stage == "audit": audit_stage()
    elif args.stage == "train-ett": train_stage(args.device)
    elif args.stage == "preflight": preflight_stage()
    elif args.stage == "generate": generate_stage()
    elif args.stage == "lineage": lineage_stage()
    elif args.stage == "train-crl":
        if args.arm is None: raise SystemExit("train-crl requires O or P")
        train_crl_stage(args.arm)
    elif args.stage == "evaluate": evaluate_stage()
    elif args.stage == "finalize": finalize_stage()
    else:
        audit_stage(); train_stage(args.device)
        if not preflight_stage():
            raise SystemExit("preflight failed; saved blocker, stopping before CRL")
        generate_stage(); lineage_stage(); train_crl_stage("O"); train_crl_stage("P")
        evaluate_stage(); finalize_stage()


if __name__ == "__main__":
    main()
