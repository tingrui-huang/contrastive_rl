"""G1: the sealed 09-15 O/P pipeline with the absorbing-freeze ETT as P transition.

Reuses ``prepare``, ``train`` and ``evaluate`` of
``ett.pointmaze_offline_causal_integration`` unchanged.  Only the phase that
generates the continuation caches is replaced: no auxiliary critic, no ETT
offset updates, the freeze tables come from the training partition alone.
Protocol: ``notes/pointmaze_absorbing_integration.md``.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from crl import checkpoint
from ett import absorbing_ett as ab
from ett import pointmaze_offline_causal_integration as oci
from ett import pointmaze_region_pilot as region
from ett.finite_crl import sha

DEFAULT_OUT = Path("outputs/pointmaze_absorbing_integration_20260915_v1")
PROTOCOL_SOURCE = Path("notes/pointmaze_absorbing_integration.md")
G1_CONFIG = {
    "experiment": "sealed_offline_pipeline_with_absorbing_freeze_P_transition",
    "base_run": "outputs/pointmaze_offline_causal_integration_20260915_v1",
    "transition_backend": "manski_absorbing",
    "ett_offset_updates": 0,
    "auxiliary_critic_updates": 0,
    "freeze_tables_partition": "train",
    "support_min_onsets": ab.SUPPORT_MIN_ONSETS,
    "support_min_absorbing_fraction": ab.SUPPORT_MIN_ABSORBING,
    "fields_read": list(ab.FIELDS_READ),
    "diagonal_identity_rows": 512,
    "diagonal_identity_draws": 32,
}
FORBIDDEN = ("gym", "gymnasium", "dm_control", "native_alive_motion", "true_transition",
             "oracle", "crl.envs")


def file_sha(path):
    return oci.file_sha(Path(path))


def read_backend(out: Path):
    return oci.read_json(out / "g1_config.json")["transition_backend"]


def support_from_cells(cells):
    support = np.zeros(ab.GRID, bool)
    for i, j in cells:
        support[int(i), int(j)] = True
    return support


# --------------------------------------------------------------------------- #
# prepare                                                                       #
# --------------------------------------------------------------------------- #
def prepare(out: Path, backend: str):
    if backend not in ab.MODES:
        raise ValueError(backend)
    oci.prepare(out)
    obs, act, _, train, heldout, plan = oci.load_prepared(out)
    tables = ab.freeze_tables(obs, act, train)
    eligible_train = np.intersect1d(np.arange(1200, 6000), train)
    manski = {"train_all_behaviours": ab.manski_tables(obs, act, train, tables["support"]),
              "train_eligible_teacher_population": ab.manski_tables(obs, act, eligible_train, tables["support"])}
    assert not np.intersect1d(train, heldout).size
    assert not np.isin(plan["fork_context_episode"], train).any()
    freeze = {k: v for k, v in tables.items() if k != "support"}
    freeze["manski_descriptive"] = manski
    freeze["estimated_on"] = {"partition": "train", "episodes": int(len(train)),
                              "heldout_episodes_excluded": int(len(heldout)),
                              "fork_context_episodes_in_train": 0}
    oci.write_json(out / "freeze_tables.json", freeze)
    config = dict(G1_CONFIG, transition_backend=backend)
    oci.write_json(out / "g1_config.json", config)
    (out / "PROTOCOL_G1.md").write_bytes(PROTOCOL_SOURCE.read_bytes())
    provenance = {
        "base_driver": "ett/pointmaze_offline_causal_integration.py",
        "base_driver_sha256": file_sha("ett/pointmaze_offline_causal_integration.py"),
        "g1_source_sha256": {str(p).replace("\\", "/"): file_sha(p) for p in
                             [Path("ett/absorbing_ett.py"), Path("ett/pointmaze_absorbing_integration.py"),
                              PROTOCOL_SOURCE]},
        "dataset_fields_read": list(ab.FIELDS_READ),
        "hidden_fields_read": [],
        "failure_mechanism": {
            "explicit_freeze_latent": backend in ("diagonal_motion_absorbing", "manski_absorbing"),
            "learned_from": "visible exact-stationarity runs in obs of the training partition",
            "off_diagonal_rule": ("per-step Manski upper bound with landing-cell bins and the "
                                  "absorbing support" if backend == "manski_absorbing" else "none"),
            "stationary_rows_relabelled_as_death": False},
        "git": {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()},
        "runtime": {"python": platform.python_version(), "jax": jax.__version__,
                    "devices": [str(d) for d in jax.devices()]},
    }
    oci.write_json(out / "g1_provenance.json", provenance)
    sealed = ["PROTOCOL_G1.md", "g1_config.json", "freeze_tables.json", "g1_provenance.json"]
    oci.write_json(out / "seal_g1.json", {
        "sealed_before_model_generation": True,
        "sha256": {name: file_sha(out / name) for name in sealed},
        "base_seal_sha256": file_sha(out / "seal.json")})
    print("G1 sealed; backend", backend, "support", tables["support_cells"], flush=True)


# --------------------------------------------------------------------------- #
# fit: continuation caches from the backend                                     #
# --------------------------------------------------------------------------- #
EXTRA_KEYS = ("frozen", "onset_manski", "onset_atom", "agree", "hazard_alive")


def generate_cache(engine, theta, obs, act, episode, time_index, seed, ledger, purpose):
    """Mirror of ``oci.generate_cache`` (same grouping, keys, format, ledger charges)
    that also stores the backend's freeze arrays."""
    n = len(episode)
    lengths = (50 - time_index).astype(np.int32)
    states = np.full((n, 51, 8), np.nan, np.float32)
    actions = np.full((n, 50, 2), np.nan, np.float32)
    x_prime = np.full((n, 50, 2), np.nan, np.float32)
    reward = np.full((n, 50), np.nan, np.float32)
    atom = np.zeros((n, 50), bool)
    projected = np.zeros((n, 50), bool)
    valid = np.zeros((n, 50), bool)
    extras = {k: np.zeros((n, 51 if k == "frozen" else 50), bool) for k in EXTRA_KEYS}
    for h in sorted(np.unique(lengths)):
        ix = np.flatnonzero(lengths == h)
        ledger.add(len(ix) * int(h), purpose + f"/h{h}")
        record = engine.rollout(theta, obs[episode[ix], time_index[ix], :8],
                                jax.random.PRNGKey(seed + int(h)), int(h),
                                act[episode[ix], time_index[ix]])
        states[ix, :h + 1] = record["states"]
        actions[ix, :h] = record["action"]
        x_prime[ix, :h] = record["x_prime"]
        reward[ix, :h] = record["reward"]
        atom[ix, :h] = record["atom"]
        projected[ix, :h] = record["projected"]
        valid[ix, :h] = True
        for k in EXTRA_KEYS:
            extras[k][ix, :(h + 1 if k == "frozen" else h)] = record[k]
        np.testing.assert_array_equal(record["action"][:, 0], act[episode[ix], time_index[ix]])
        np.testing.assert_array_equal(record["states"][:, 1:, 2:], record["states"][:, :-1, :6])
        print(purpose, "remaining horizon", int(h), "anchors", len(ix), flush=True)
    state_valid = np.arange(51)[None, :] <= lengths[:, None]
    transition_valid = np.arange(50)[None, :] < lengths[:, None]
    assert np.isfinite(states[state_valid]).all()
    assert np.isnan(states[~state_valid]).all()
    assert np.array_equal(valid, transition_valid)
    assert np.isfinite(actions[transition_valid]).all()
    assert np.isnan(actions[~transition_valid]).all()
    return {"states": states, "action": actions, "x_prime": x_prime,
            "reward": reward, "atom": atom, "projected": projected,
            "valid": valid, "length": lengths, "episode": episode,
            "anchor_time": time_index,
            "anchor_action": act[episode, time_index].astype(np.float32),
            "theta_sha256": np.array(oci.array_sha(theta)),
            "base_diagonal_file_sha256": np.array(oci.EXPECTED[str(oci.DIAGONAL)]),
            "nominal_file_sha256": np.array(oci.EXPECTED[str(oci.NOMINAL)]),
            "fixed_actor_file_sha256": np.array(oci.EXPECTED[str(oci.INITIAL)]),
            **extras}


def cache_freeze_audit(record, source):
    result = {}
    for name, paths in (("all", np.ones(len(source), bool)), ("ordinary_nonfork", source == 0),
                        ("down", source == 1), ("right", source == 2), ("fork", source != 0)):
        rows = np.flatnonzero(paths)
        reach, absorbed, entered, manski, atom, agree = [], [], [], [], [], []
        for i in rows:
            h = int(record["length"][i])
            reach.append(bool(np.any(record["reward"][i, :h] > 0)))
            absorbed.append(bool(record["frozen"][i, h]))
            entered.append(bool(np.any(record["hazard_alive"][i, :h])))
            manski.append(bool(np.any(record["onset_manski"][i, :h])))
            atom.append(bool(np.any(record["onset_atom"][i, :h])))
            agree.append(float(np.mean(record["agree"][i, :h])))
        result[name] = {
            "paths": int(len(rows)),
            "task_region_reached_path_fraction": float(np.mean(reach)),
            "absorbed_by_end_fraction": float(np.mean(absorbed)),
            "entered_absorbing_support_alive_fraction": float(np.mean(entered)),
            "manski_onset_fraction": float(np.mean(manski)),
            "atom_onset_fraction": float(np.mean(atom)),
            "mean_bin_agreement_rate": float(np.mean(agree))}
    return result


def recorded_continuation_audit(obs, episode, time_index, source):
    """Reach and visible freeze of the matching recorded (O) continuations."""
    result = {}
    xy = obs[:, :, :2].astype(np.float64)
    for name, paths in (("all", np.ones(len(source), bool)), ("ordinary_nonfork", source == 0),
                        ("down", source == 1), ("right", source == 2), ("fork", source != 0)):
        rows = np.flatnonzero(paths)
        reach, frozen_end = [], []
        for i in rows:
            e, t = int(episode[i]), int(time_index[i])
            future = xy[e, t + 1:51]
            reach.append(bool(np.any(np.linalg.norm(future - ab.GOAL[:2], axis=1) < 2.)))
            # visible absorbing freeze: the continuation ends in an exactly stationary run
            frozen_end.append(bool(np.linalg.norm(xy[e, 50] - xy[e, 49]) <= ab.STATIONARY_TOLERANCE
                                   and t + 1 < 50))
        result[name] = {"paths": int(len(rows)),
                        "task_region_reached_path_fraction": float(np.mean(reach)),
                        "ends_exactly_stationary_fraction": float(np.mean(frozen_end))}
    return result


def fit(out: Path):
    if (out / "ett_training.json").exists():
        raise ValueError("cache phase already complete")
    backend = read_backend(out)
    if backend == "eligible_response":
        oci.fit_ett(out)
        return
    obs, act, initial, train, _, plan = oci.load_prepared(out)
    freeze = oci.read_json(out / "freeze_tables.json")
    recomputed = ab.freeze_tables(obs, act, train)
    assert recomputed["support_cells"] == freeze["support_cells"]
    support = support_from_cells(freeze["support_cells"])
    engine = region.Kernel()
    assert oci.tree_sha(initial.policy_params) == oci.tree_sha(
        checkpoint.load_checkpoint(oci.INITIAL)[1].policy_params)
    immutable = oci.tree_sha((engine.base.params, engine.nominal.params))
    backend_engine = ab.AbsorbingRollout(engine, support, backend)
    context = oci.diagonal_context(obs, act, plan)
    train_states = obs[train, :50, :8].reshape(-1, 8)
    context["mean"] = train_states.mean(0)
    context["std"] = np.maximum(train_states.std(0), .1)
    ledger = oci.Ledger(out)
    theta = np.zeros(48, np.float32)
    seed = oci.CONFIG["seed"]

    # Diagonal identity: same rows, same key, eligible Kernel versus backend.
    rows = np.arange(G1_CONFIG["diagonal_identity_rows"])
    draws = G1_CONFIG["diagonal_identity_draws"]
    validation_initial = region.diagonal(engine, theta, context, "validation", rows, draws,
                                         seed + 200, ledger, "ett/validation_diagonal_initial")
    validation_backend = region.diagonal(backend_engine, theta, context, "validation", rows, draws,
                                         seed + 200, ledger, "freeze/validation_diagonal_backend")
    s, a = context["validation_state"][rows], context["validation_action"][rows]
    y_kernel, _ = engine.sample(jnp.asarray(theta), s, a, a, jax.random.PRNGKey(seed + 200), draws)
    y_backend, _ = backend_engine.sample(theta, s, a, a, jax.random.PRNGKey(seed + 200), draws)
    diagonal_identical = bool(np.array_equal(np.asarray(y_kernel), np.asarray(y_backend)))
    assert diagonal_identical and np.array_equal(validation_initial, validation_backend)

    train_cache = generate_cache(backend_engine, theta, obs, act, plan["train_pool_episode"],
                                 plan["train_pool_time"], seed + 300, ledger,
                                 "production/train_final_ett_cache")
    validation_cache = generate_cache(backend_engine, theta, obs, act, plan["validation_pool_episode"],
                                      plan["validation_pool_time"], seed + 400, ledger,
                                      "production/validation_final_ett_cache")
    initial_validation_cache = oci.generate_cache(engine, theta, obs, act,
                                                  plan["validation_pool_episode"],
                                                  plan["validation_pool_time"], seed + 400, ledger,
                                                  "audit/validation_initial_ett_cache")
    oci.save_cache(out / "train_trajectory_pool.npz", train_cache)
    oci.save_cache(out / "validation_trajectory_pool.npz", validation_cache)
    oci.save_cache(out / "initial_validation_trajectory_pool.npz", initial_validation_cache)
    np.savez_compressed(out / "checkpoints" / "ett_final.npz", theta=theta)
    interface = {
        "transition_backend": backend,
        "initial_validation": oci.cache_audit(initial_validation_cache),
        "final_validation": oci.cache_audit(validation_cache),
        "paired_initial_to_final_validation": oci.cache_pair_audit(
            initial_validation_cache, validation_cache, plan["validation_pool_source"]),
        "final_training": oci.cache_audit(train_cache),
        "freeze_validation": cache_freeze_audit(validation_cache, plan["validation_pool_source"]),
        "freeze_training": cache_freeze_audit(train_cache, plan["train_pool_source"]),
        "recorded_training_continuations": recorded_continuation_audit(
            obs, plan["train_pool_episode"], plan["train_pool_time"], plan["train_pool_source"]),
        "recorded_validation_continuations": recorded_continuation_audit(
            obs, plan["validation_pool_episode"], plan["validation_pool_time"],
            plan["validation_pool_source"]),
        "diagonal_validation_energy_initial": float(np.mean(validation_initial)),
        "diagonal_validation_energy_final": float(np.mean(validation_backend)),
        "diagonal_validation_energy_change": float(np.mean(validation_backend - validation_initial)),
        "diagonal_one_step_bit_identical_to_eligible_kernel": diagonal_identical,
        "explicit_failure_supported": backend in ("diagonal_motion_absorbing", "manski_absorbing"),
        "failure_assumption": ("absorbing freeze learned from visible stationary runs; off-diagonal "
                               "entry by the per-step Manski upper bound" if backend == "manski_absorbing"
                               else "absorbing freeze from visible stationary runs; observational entry only"),
        "support_cells": freeze["support_cells"],
        "stationary_atom_is_failure": False,
    }
    oci.write_json(out / "sampling_interface_pretraining.json", interface)
    assert immutable == oci.tree_sha((engine.base.params, engine.nominal.params))
    training = {"status": "complete", "transition_backend": backend,
                "theta_initial_sha256": oci.array_sha(theta), "theta_final_sha256": oci.array_sha(theta),
                "theta_l2": 0., "accepted_updates": 0, "history": [], "auxiliary_prefit": None,
                "auxiliary_updates": 0, "model_outputs": ledger.data["charged"],
                "model_output_cap": ledger.data["cap"],
                "immutable_diagonal_and_nominal_sha256": immutable,
                "fixed_actor_sha256": oci.tree_sha(initial.policy_params),
                "failure_model": interface["failure_assumption"]}
    oci.write_json(out / "ett_training.json", training)
    print("absorbing-freeze cache phase complete; model outputs", ledger.data["charged"], flush=True)


# --------------------------------------------------------------------------- #
# verify                                                                        #
# --------------------------------------------------------------------------- #
def verify(out: Path):
    obs, act, initial, train, heldout, plan = oci.load_prepared(out)
    provenance = oci.read_json(out / "provenance.json")
    seal, seal_g1 = oci.read_json(out / "seal.json"), oci.read_json(out / "seal_g1.json")
    backend = read_backend(out)
    if backend == "eligible_response":
        oci.verify(out)
        return
    checks = []

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    check(oci.read_json(out / "config.json") == oci.CONFIG, "sealed base config equals implementation config")
    check(oci.read_json(out / "g1_config.json") == dict(G1_CONFIG, transition_backend=backend),
          "sealed G1 config equals implementation config")
    for name, digest in seal["sha256"].items():
        check(file_sha(out / name) == digest, f"sealed hash unchanged: {name}")
    for name, digest in seal_g1["sha256"].items():
        check(file_sha(out / name) == digest, f"sealed G1 hash unchanged: {name}")
    check(seal_g1["base_seal_sha256"] == file_sha(out / "seal.json"), "base seal unchanged")
    for path, digest in oci.EXPECTED.items():
        check(file_sha(path) == digest, f"eligible input hash unchanged: {path}")
    for path, digest in provenance["source_sha256"].items():
        check(file_sha(path) == digest, f"sealed source hash unchanged: {path}")
    for path, digest in provenance["historical_sha256_before"].items():
        check(file_sha(path) == digest, f"historical artifact unchanged: {path}")
    g1_prov = oci.read_json(out / "g1_provenance.json")
    for path, digest in g1_prov["g1_source_sha256"].items():
        check(file_sha(path) == digest, f"G1 source hash unchanged: {path}")
    for path in ("ett/absorbing_ett.py", "ett/pointmaze_absorbing_integration.py"):
        modules = oci.imported_modules(Path(path))
        check(not any(any(tok in m.lower() for tok in FORBIDDEN) for m in modules),
              f"{path} imports no environment, native, true-transition or oracle module")
    check(list(ab.FIELDS_READ) == ["obs", "act"] and g1_prov["hidden_fields_read"] == [],
          "backend reads only obs and act")
    check(np.array_equal(train, oci.split_ids()[0]) and np.array_equal(heldout, oci.split_ids()[1]),
          "established episode split preserved")
    oci.validate_plan(plan, obs, act, train, heldout)
    checks.append("sampler plan strata, split, and strictly-later offsets valid")

    freeze = oci.read_json(out / "freeze_tables.json")
    recomputed = ab.freeze_tables(obs, act, train)
    check(recomputed["support_cells"] == freeze["support_cells"] and
          recomputed["onset_table"] == freeze["onset_table"],
          "freeze tables reproduce from the training partition alone")
    check(not np.isin(plan["fork_context_episode"], train).any(),
          "evaluation fork roots are outside the freeze-table partition")
    support = support_from_cells(freeze["support_cells"])

    final_theta = np.load(out / "checkpoints" / "ett_final.npz", allow_pickle=False)["theta"]
    check(not np.any(final_theta), "no ETT offsets were fitted")
    cache_specs = (("train_trajectory_pool.npz", "train_pool", True),
                   ("validation_trajectory_pool.npz", "validation_pool", True),
                   ("initial_validation_trajectory_pool.npz", "validation_pool", False))
    loaded = {}
    for filename, prefix, is_backend in cache_specs:
        cache = oci.load_cache(out / filename); loaded[filename] = cache
        episode, anchor_time = plan[prefix + "_episode"], plan[prefix + "_time"]
        check(np.array_equal(cache["episode"], episode) and np.array_equal(cache["anchor_time"], anchor_time),
              f"{filename}: anchor identities exact")
        check(np.array_equal(cache["states"][:, 0], obs[episode, anchor_time, :8]),
              f"{filename}: recorded anchor states exact")
        check(np.array_equal(cache["action"][:, 0], act[episode, anchor_time]) and
              np.array_equal(cache["anchor_action"], act[episode, anchor_time]),
              f"{filename}: recorded first actions exact")
        check(str(cache["theta_sha256"]) == oci.array_sha(np.zeros(48, np.float32)),
              f"{filename}: ETT version identifier exact")
        check(str(cache["base_diagonal_file_sha256"]) == oci.EXPECTED[str(oci.DIAGONAL)] and
              str(cache["nominal_file_sha256"]) == oci.EXPECTED[str(oci.NOMINAL)] and
              str(cache["fixed_actor_file_sha256"]) == oci.EXPECTED[str(oci.INITIAL)],
              f"{filename}: component identifiers exact")
        state_valid = np.arange(51)[None, :] <= cache["length"][:, None]
        transition_valid = np.arange(50)[None, :] < cache["length"][:, None]
        check(np.array_equal(cache["valid"], transition_valid), f"{filename}: validity mask exact")
        check(np.isfinite(cache["states"][state_valid]).all() and np.isnan(cache["states"][~state_valid]).all(),
              f"{filename}: state padding cannot be sampled")
        check(np.isfinite(cache["action"][transition_valid]).all() and
              np.isnan(cache["action"][~transition_valid]).all(), f"{filename}: action padding explicit")
        history_error = max(float(np.max(np.abs(cache["states"][i, 1:h + 1, 2:] - cache["states"][i, :h, :6])))
                            for i, h in enumerate(cache["length"]))
        check(history_error == 0., f"{filename}: full-F4 history shifts exact")
        if is_backend:
            check(all(k in cache for k in EXTRA_KEYS), f"{filename}: freeze arrays present")
            for i, h in enumerate(cache["length"]):
                h = int(h)
                fr = cache["frozen"][i, :h]
                xy = cache["states"][i, :h + 1, :2]
                moved = np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0
                if np.any(moved & fr) or np.any(cache["reward"][i, :h] * fr):
                    raise AssertionError(f"{filename}: frozen path moved or was rewarded")
                onset = cache["onset_manski"][i, :h] | cache["onset_atom"][i, :h]
                if np.any(cache["onset_manski"][i, :h] & cache["agree"][i, :h]):
                    raise AssertionError(f"{filename}: Manski onset with bin agreement")
                land = ab.landing_cell(xy[1:h + 1])
                if np.any(onset & ~support[land[:, 0], land[:, 1]]):
                    raise AssertionError(f"{filename}: onset outside the absorbing support")
                if np.any(np.diff(fr.astype(int)) < 0):
                    raise AssertionError(f"{filename}: frozen flag reverted")
            checks.append(f"{filename}: frozen paths stationary, unrewarded, onsets only in support and only on bin disagreement")

    lineage = np.load(out / "nce_row_lineage.npz", allow_pickle=False)
    ids, offsets = lineage["pool_id"], lineage["future_offset"]
    check(np.array_equal(ids, plan["critic_pool_id"]) and np.array_equal(offsets, plan["critic_offset"]) and
          np.array_equal(lineage["source"], plan["critic_source"]),
          "O/P production row identity, strata, and future offsets matched")
    episode, anchor_time = plan["train_pool_episode"][ids], plan["train_pool_time"][ids]
    check(np.array_equal(lineage["observational_goal"], obs[episode, anchor_time + offsets, :8]),
          "every O positive is the matching recorded future")
    expected_p = loaded["train_trajectory_pool.npz"]["states"][ids, offsets]
    check(np.array_equal(lineage["pessimistic_goal"], expected_p), "every P positive is the matching backend future")
    check(np.isfinite(expected_p).all(), "no padded P future entered NCE")
    frozen_goal = loaded["train_trajectory_pool.npz"]["frozen"][ids, offsets]
    checks.append(f"P positives frozen fraction {float(frozen_goal.mean()):.4f}")

    training = oci.read_json(out / "production_training.json")
    check(training["integrity"]["P_positive_source"].startswith("100%") and
          training["integrity"]["O_positive_source"].startswith("100%"),
          "production positive-source fractions are 100% O and 100% P")
    for arm in ("O", "P"):
        cstep, critic_state = checkpoint.load_checkpoint(out / "checkpoints" / f"critic_{arm}_final.pkl")
        astep, actor_state = checkpoint.load_checkpoint(out / "checkpoints" / f"actor_{arm}_final.pkl")
        check(cstep == 150400 and astep == 151400, f"{arm}: fixed checkpoint update counts exact")
        check(oci.tree_sha(critic_state.policy_params) == oci.tree_sha(initial.policy_params) and
              oci.tree_sha(critic_state.policy_optimizer_state) == oci.tree_sha(initial.policy_optimizer_state),
              f"{arm}: actor and actor optimizer frozen during critic stage")
        check(oci.tree_sha(actor_state.q_params) == oci.tree_sha(critic_state.q_params) and
              oci.tree_sha(actor_state.q_optimizer_state) == oci.tree_sha(critic_state.q_optimizer_state),
              f"{arm}: critic and critic optimizer frozen during actor stage")
        check(oci.tree_delta_norm(actor_state.policy_params, initial.policy_params) > 0.,
              f"{arm}: actor optimizer applied nonzero changes")
    curves = np.load(out / "learning_curves.npz", allow_pickle=False)
    check(curves["critic_O"].shape == (400, 8) and curves["critic_P"].shape == (400, 8),
          "exactly 400 production critic updates per arm")
    check(curves["actor_O"].shape == (1000, 9) and curves["actor_P"].shape == (1000, 9),
          "exactly 1000 actor updates per arm")
    ett = oci.read_json(out / "ett_training.json")
    ledger = oci.read_json(out / "model_ledger.json")
    check(ett["auxiliary_updates"] == 0 and ett["history"] == [] and ett["accepted_updates"] == 0,
          "no auxiliary critic or ETT offset updates")
    check(ledger["charged"] == ett["model_outputs"] and ledger["charged"] <= ledger["cap"],
          "model-generation ledger within sealed cap")
    interface = oci.read_json(out / "sampling_interface_pretraining.json")
    check(interface["diagonal_one_step_bit_identical_to_eligible_kernel"] and
          interface["diagonal_validation_energy_change"] == 0.,
          "diagonal one-step law identical to the eligible kernel")
    check(oci.read_json(out / "results.json")["sampling"]["matched_future_change"]["P_positive_fraction_from_ETT"] == 1.,
          "evaluation confirms all P positives use the backend futures")
    payload = {"status": "passed", "checks_passed": len(checks), "checks": checks,
               "transition_backend": backend, "environment_interactions": 0, "native_steps": 0,
               "historical_artifacts_modified": False,
               "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    oci.write_json(out / "verification.json", payload)
    print("G1 verification passed", len(checks), "checks", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "fit", "train", "evaluate", "verify", "all"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--backend", default="manski_absorbing", choices=ab.MODES)
    args = parser.parse_args()
    if args.phase in ("prepare", "all"): prepare(args.out, args.backend)
    if args.phase in ("fit", "all"): fit(args.out)
    if args.phase in ("train", "all"): oci.train(args.out)
    if args.phase in ("evaluate", "all"): oci.evaluate(args.out)
    if args.phase in ("verify", "all"): verify(args.out)


if __name__ == "__main__":
    main()
