"""Deterministic (mode-action) native evaluation of the six G3 policies on the
same 200 seeds.

The repo's training-time evaluation (crl/train.py::evaluate, "greedy
rollouts") and the earlier deployment audits act with the policy MODE,
tanh(loc), whereas G3 and the baseline run sampled tanh(loc + scale * eps)
(the 09-14 pilots' "frozen stochastic actor" harness).  This script repeats
the evaluation with the mode so both protocols sit side by side.  Same reset
seeds, same environment, region-fork route classification, no training.
"""
import csv
import importlib.util
import sys
import time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crl import checkpoint  # noqa: E402
from crl.envs import TwoRouteSwampWindyF4Env  # noqa: E402
from ett import pointmaze_offline_causal_integration as oci  # noqa: E402
from ett.rollout_return import START  # noqa: E402

OUT = Path(__file__).resolve().parent
G2 = ROOT / "outputs/pointmaze_absorbing_actor_sweep_20260915_v1/checkpoints"
G3 = ROOT / "outputs/pointmaze_absorbing_native_g3_20260915_v1"
rf = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("g3_reclassify", G3 / "reclassify.py")); spec.loader.exec_module(rf)
EPISODES, HORIZON, RESET_SEED_BASE = 200, 50, 9_300_000
ORDER = ["vanilla_bc0.5", "vanilla_critic_bc0.2", "O_bc0.5", "O_bc0.2", "P_bc0.5", "P_bc0.2"]


def run_mode(params, network):
    mode = jax.jit(lambda p, o: jnp.tanh(network.policy_network.apply(p, o).loc))
    envs = [TwoRouteSwampWindyF4Env(seed=RESET_SEED_BASE + i, active_prob=.3) for i in range(EPISODES)]
    obs = np.stack([e._get_obs().copy() for e in envs]).astype(np.float32)
    np.testing.assert_array_equal(obs[:, :8], np.broadcast_to(np.asarray(START, np.float32), (EPISODES, 8)))
    physical = [np.stack([e.state.copy() for e in envs])]
    actions, rewards, dead = [], [], []
    for t in range(HORIZON):
        a = np.asarray(mode(params, jnp.asarray(obs)), np.float32)
        step_obs, step_r = [], []
        for e, ai in zip(envs, a):
            o, r, done, _ = e.step(ai); assert not done
            step_obs.append(o); step_r.append(r)
        obs = np.stack(step_obs).astype(np.float32)
        physical.append(np.stack([e.state.copy() for e in envs]))
        actions.append(a); rewards.append(np.asarray(step_r, np.float32)); dead.append(np.asarray([e.dead for e in envs]))
    return {"physical": np.stack(physical, 1), "reward": np.stack(rewards, 1), "dead": np.stack(dead, 1), "action": np.stack(actions, 1)}


def main():
    t0 = time.time()
    network, _, _ = oci.make_network_and_optimizers()
    initial = checkpoint.load_checkpoint(oci.INITIAL)[1]
    params = {"vanilla_bc0.5": initial.policy_params,
              "vanilla_critic_bc0.2": checkpoint.load_checkpoint(OUT / "checkpoints/actor_vanilla_critic_bc0p2.pkl")[1].policy_params,
              "O_bc0.5": checkpoint.load_checkpoint(G2 / "actor_O_bc0p5.pkl")[1].policy_params,
              "O_bc0.2": checkpoint.load_checkpoint(G2 / "actor_O_bc0p2.pkl")[1].policy_params,
              "P_bc0.5": checkpoint.load_checkpoint(G2 / "actor_P_bc0p5.pkl")[1].policy_params,
              "P_bc0.2": checkpoint.load_checkpoint(G2 / "actor_P_bc0p2.pkl")[1].policy_params}
    per = {}
    for name in ORDER:
        rec = run_mode(params[name], network)
        np.savez_compressed(OUT / f"native_mode_{name}.npz", **rec, reset_seed=np.arange(EPISODES) + RESET_SEED_BASE)
        per[name] = rf.classify(rec)
    fields = ["episode", "policy"] + rf.METRICS + ["route_class", "first_departure_cell", "time_to_goal", "death_time"]
    with open(OUT / "per_episode_mode_regionfork.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name in ORDER:
            for i, r in enumerate(per[name]):
                w.writerow({"episode": i, "policy": name, **{k: r[k] for k in fields[2:]}})
    summary = {"protocol": "deterministic mode action tanh(loc); same seeds as G3", "policies": {}, "comparisons": {}}
    for name in ORDER:
        rows = per[name]
        s = {k: oci.bootstrap_mean(np.asarray([r[k] for r in rows], np.float64), 0) for k in rf.METRICS}
        s["route_classes"] = {c: v / EPISODES for c, v in Counter(r["route_class"] for r in rows).items()}
        s["first_departure_cells"] = {c: v / EPISODES for c, v in Counter(r["first_departure_cell"] for r in rows).most_common()}
        ttg = [r["time_to_goal"] for r in rows if r["time_to_goal"] is not None]
        s["time_to_goal_given_reach"] = float(np.mean(ttg)) if ttg else None
        summary["policies"][name] = s
    for a, b in [("P_bc0.2", "O_bc0.2"), ("P_bc0.2", "vanilla_critic_bc0.2"), ("P_bc0.5", "O_bc0.5"), ("P_bc0.2", "P_bc0.5")]:
        summary["comparisons"][f"{a}_minus_{b}"] = {k: rf.paired_bootstrap([r[k] for r in per[a]], [r[k] for r in per[b]], 100 + i) for i, k in enumerate(rf.METRICS)}
    summary["native_steps"] = len(ORDER) * EPISODES * HORIZON
    summary["elapsed_seconds"] = time.time() - t0
    oci.write_json(OUT / "results_mode.json", summary)
    for name in ORDER:
        s = summary["policies"][name]
        print(f"{name:22s} reach {s['reach']['mean']:.3f} absorbed {s['absorbed']['mean']:.3f} lower {s['lower_route']['mean']:.3f} "
              f"y<2 {s['reached_y_below_2']['mean']:.3f} shortcut {s['shortcut']['mean']:.3f} stuck {s['stuck_other']['mean']:.3f} "
              f"disc {s['discounted_return']['mean']:.2f} dep {dict(list(s['first_departure_cells'].items())[:3])}")
    for k, v in summary["comparisons"].items():
        print(k, {m: (round(v[m]["mean"], 3), [round(x, 3) for x in v[m]["ci95"]]) for m in ["reach", "absorbed", "lower_route", "reached_y_below_2", "shortcut"]})


if __name__ == "__main__":
    main()
