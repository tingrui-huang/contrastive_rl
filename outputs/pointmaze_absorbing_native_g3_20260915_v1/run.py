"""G3: pre-registered native evaluation (200 paired episodes) of the faithful
paper actor with bc = 0.2 (P vs O) and the bc = 0.5 baseline pair.

No training. The native environment is the source of truth. See PROTOCOL.md.
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crl import checkpoint  # noqa: E402
from crl.envs import TwoRouteSwampWindyF4Env  # noqa: E402
from ett import pointmaze_offline_causal_integration as oci  # noqa: E402
from ett.diagonal_transition import POINTMAZE_WALLS  # noqa: E402
from ett.rollout_return import GOAL, START  # noqa: E402

OUT = Path(__file__).resolve().parent
G1 = ROOT / "outputs/pointmaze_absorbing_integration_20260915_v1"
G2 = ROOT / "outputs/pointmaze_absorbing_actor_sweep_20260915_v1"
POLICIES = {"P_bc0.2": "actor_P_bc0p2.pkl", "O_bc0.2": "actor_O_bc0p2.pkl",
            "P_bc0.5": "actor_P_bc0p5.pkl", "O_bc0.5": "actor_O_bc0p5.pkl"}
COMPARISONS = [("P_bc0.2", "O_bc0.2"), ("P_bc0.5", "O_bc0.5"), ("P_bc0.2", "P_bc0.5"), ("O_bc0.2", "O_bc0.5")]
EPISODES, HORIZON, DISCOUNT = 200, 50, .95
RESET_SEED_BASE, ACTION_SEED_BASE = 9_300_000, 9_400_000
BOOT_REPLICATES, BOOT_SEED = 2000, 0
WALLS = np.asarray(POINTMAZE_WALLS)
GOAL2 = np.asarray(GOAL[:2], np.float64)
METRICS = ["reach", "strict_success", "discounted_return", "undiscounted_return", "absorbed",
           "lower_route", "entered_1_2", "reached_y_below_2", "shortcut", "stuck_other",
           "first_fork_landing_1_2", "first_fork_landing_2_3", "hazard_landings"]


def cell(xy):
    return np.clip(np.floor(np.asarray(xy)).astype(int), [0, 0], [8, 4])


def hazardous(xy):
    return (xy[..., 0] >= 3) & (xy[..., 0] < 6) & (xy[..., 1] >= 3) & (xy[..., 1] < 4)


def run_policy(name, params, network):
    sample = jax.jit(lambda p, o, key: jnp.tanh((lambda d: d.loc + d.scale * jax.random.normal(key, d.loc.shape))(
        network.policy_network.apply(p, o))))
    envs = [TwoRouteSwampWindyF4Env(seed=RESET_SEED_BASE + i, active_prob=.3) for i in range(EPISODES)]
    obs = np.stack([e._get_obs().copy() for e in envs]).astype(np.float32)
    np.testing.assert_array_equal(obs[:, :8], np.broadcast_to(np.asarray(START, np.float32), (EPISODES, 8)))
    physical = [np.stack([e.state.copy() for e in envs])]
    actions, rewards, dead = [], [], []
    for t in range(HORIZON):
        keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(ACTION_SEED_BASE + t), i))(jnp.arange(EPISODES))
        a = np.asarray(jax.vmap(lambda o, k: sample(params, o[None], k)[0])(jnp.asarray(obs), keys), np.float32)
        assert np.isfinite(a).all() and np.abs(a).max() <= 1.
        step_obs, step_r = [], []
        for e, ai in zip(envs, a):
            o, r, done, _ = e.step(ai)
            assert not done
            step_obs.append(o); step_r.append(r)
        new_obs = np.stack(step_obs).astype(np.float32)
        np.testing.assert_array_equal(new_obs[:, 2:8], obs[:, :6])
        obs = new_obs
        physical.append(np.stack([e.state.copy() for e in envs]))
        actions.append(a); rewards.append(np.asarray(step_r, np.float32)); dead.append(np.asarray([e.dead for e in envs]))
    physical = np.stack(physical, 1)                     # [E, 51, 2]
    rewards = np.stack(rewards, 1); dead = np.stack(dead, 1); actions = np.stack(actions, 1)
    assert not np.any(rewards[:, 1:][dead[:, :-1]] != 0)
    return {"physical": physical, "reward": rewards, "dead": dead, "action": actions}


def classify(rec):
    phys, reward, dead = rec["physical"], rec["reward"], rec["dead"]
    E = len(phys)
    rows = []
    for i in range(E):
        c = cell(phys[i])                                # [51, 2]
        fork = np.flatnonzero((c[:, 0] == 1) & (c[:, 1] == 3))
        haz = hazardous(phys[i])
        t_fork = int(fork[0]) if len(fork) else -1
        landing = c[t_fork + 1] if 0 <= t_fork < HORIZON else np.array([-1, -1])
        after = slice(t_fork + 1, None) if t_fork >= 0 else slice(HORIZON + 1, None)
        in12 = (c[after, 0] == 1) & (c[after, 1] == 2)
        below = phys[i, after, 1] < 2.
        hz = haz[after]
        t12 = np.flatnonzero(in12 | below)
        thz = np.flatnonzero(hz)
        t_low = int(t12[0]) if len(t12) else None
        t_hz = int(thz[0]) if len(thz) else None
        if t_fork < 0:
            route = "stuck_other"
        elif t_low is not None and (t_hz is None or t_low < t_hz):
            route = "lower"
        elif t_hz is not None:
            route = "shortcut"
        else:
            route = "stuck_other"
        reach = bool((reward[i] > 0).any())
        ttg = int(np.argmax(reward[i] > 0) + 1) if reach else None
        rows.append({
            "reach": float(reach),
            "strict_success": float(np.linalg.norm(phys[i] - GOAL2, axis=1).min() < .5),
            "discounted_return": float(reward[i] @ DISCOUNT ** np.arange(HORIZON)),
            "undiscounted_return": float(reward[i].sum()),
            "absorbed": float(dead[i, -1]),
            "lower_route": float(route == "lower"),
            "entered_1_2": float(in12.any()),
            "reached_y_below_2": float(below.any()),
            "shortcut": float(route == "shortcut"),
            "stuck_other": float(route == "stuck_other"),
            "first_fork_landing_1_2": float(tuple(landing) == (1, 2)),
            "first_fork_landing_2_3": float(tuple(landing) == (2, 3)),
            "hazard_landings": float((haz[1:] & ~np.concatenate([[False], dead[i, :-1]])).sum()),
            "route_class": route, "fork_time": t_fork,
            "first_fork_landing_cell": f"({int(landing[0])},{int(landing[1])})" if t_fork >= 0 else "none",
            "time_to_goal": ttg,
            "death_time": int(np.argmax(dead[i]) + 1) if dead[i].any() else None,
        })
    return rows


def paired_bootstrap(a, b, seed):
    rng = np.random.default_rng(seed)
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    idx = rng.integers(0, len(d), (BOOT_REPLICATES, len(d)))
    draws = d[idx].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def main():
    t0 = time.time()
    network, _, _ = oci.make_network_and_optimizers()
    manifest = {}
    params = {}
    g1_actor_sha = {arm: oci.file_sha(G1 / "checkpoints" / f"actor_{arm}_final.pkl") for arm in "OP"}
    for name, fname in POLICIES.items():
        path = G2 / "checkpoints" / fname
        step, state = checkpoint.load_checkpoint(path)
        params[name] = state.policy_params
        manifest[name] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": oci.file_sha(path),
                          "checkpoint_step": int(step), "policy_tree_sha256": oci.tree_sha(state.policy_params),
                          "critic_tree_sha256": oci.tree_sha(state.q_params),
                          "trained_against": f"G1 stratified critic_{name[0]}_final.pkl",
                          "critic_file_sha256": oci.file_sha(G1 / "checkpoints" / f"critic_{name[0]}_final.pkl"),
                          "bc_coef": float(name.split("bc")[1]), "objective": "(1-bc) E_pi[Q] + bc log pi(a_data|s,g); uniform actor rows"}
    for arm in "OP":
        manifest[f"{arm}_bc0.5"]["byte_identical_to_G1_actor"] = (manifest[f"{arm}_bc0.5"]["sha256"] == g1_actor_sha[arm])
    manifest["environment"] = {"name": "point_two_route_swamp_windy_f4_v0", "class": "TwoRouteSwampWindyF4Env",
                               "active_prob": .3, "action_noise": .01, "horizon": HORIZON, "reward": "1[dist<2]",
                               "source_sha256": oci.file_sha(ROOT / "crl/envs.py")}
    manifest["pairing"] = {"episodes": EPISODES, "reset_seeds": [RESET_SEED_BASE + i for i in range(EPISODES)],
                           "action_innovation": "fold_in(PRNGKey(9400000 + t), episode)", "protocol_sha256": oci.file_sha(OUT / "PROTOCOL.md"),
                           "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    oci.write_json(OUT / "checkpoint_manifest.json", manifest)

    records, per_episode = {}, {}
    for name in POLICIES:
        rec = run_policy(name, params[name], network)
        np.savez_compressed(OUT / f"native_{name}.npz", **rec, reset_seed=np.arange(EPISODES) + RESET_SEED_BASE)
        rows = classify(rec)
        records[name] = rec; per_episode[name] = rows
        m = {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
        print(f"{name}: reach {m['reach']:.3f} strict {m['strict_success']:.3f} return {m['discounted_return']:.3f} absorbed {m['absorbed']:.3f} "
              f"lower {m['lower_route']:.3f} shortcut {m['shortcut']:.3f} stuck {m['stuck_other']:.3f} fork->(1,2) {m['first_fork_landing_1_2']:.3f} | {time.time() - t0:.0f}s", flush=True)

    # per-episode table
    fields = ["episode", "reset_seed", "policy"] + METRICS + ["route_class", "fork_time", "first_fork_landing_cell", "time_to_goal", "death_time"]
    with open(OUT / "per_episode.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name in POLICIES:
            for i, r in enumerate(per_episode[name]):
                w.writerow({"episode": i, "reset_seed": RESET_SEED_BASE + i, "policy": name, **{k: r[k] for k in fields[3:]}})

    # summaries and paired bootstraps
    summary = {"policies": {}, "comparisons": {}}
    for name in POLICIES:
        rows = per_episode[name]
        s = {k: oci.bootstrap_mean(np.asarray([r[k] for r in rows], np.float64), BOOT_SEED) for k in METRICS}
        ttg = [r["time_to_goal"] for r in rows if r["time_to_goal"] is not None]
        s["time_to_goal_given_reach"] = {"mean": float(np.mean(ttg)) if ttg else None, "n": len(ttg)}
        dt = [r["death_time"] for r in rows if r["death_time"] is not None]
        s["death_time_given_absorbed"] = {"mean": float(np.mean(dt)) if dt else None, "n": len(dt)}
        s["first_fork_landing_cells"] = {c: v / EPISODES for c, v in sorted(
            __import__("collections").Counter(r["first_fork_landing_cell"] for r in rows).items(), key=lambda kv: -kv[1])}
        s["route_classes"] = {c: v / EPISODES for c, v in __import__("collections").Counter(r["route_class"] for r in rows).items()}
        s["reach_given_lower"] = float(np.mean([r["reach"] for r in rows if r["route_class"] == "lower"])) if any(r["route_class"] == "lower" for r in rows) else None
        s["reach_given_shortcut"] = float(np.mean([r["reach"] for r in rows if r["route_class"] == "shortcut"])) if any(r["route_class"] == "shortcut" for r in rows) else None
        s["absorbed_given_shortcut"] = float(np.mean([r["absorbed"] for r in rows if r["route_class"] == "shortcut"])) if any(r["route_class"] == "shortcut" for r in rows) else None
        summary["policies"][name] = s
    for a, b in COMPARISONS:
        comp = {}
        for i, k in enumerate(METRICS):
            comp[k] = paired_bootstrap([r[k] for r in per_episode[a]], [r[k] for r in per_episode[b]], BOOT_SEED + 100 + i)
        both = [(ra["time_to_goal"], rb["time_to_goal"]) for ra, rb in zip(per_episode[a], per_episode[b])
                if ra["time_to_goal"] is not None and rb["time_to_goal"] is not None]
        comp["time_to_goal_both_reached"] = {"n": len(both), "mean_difference": float(np.mean([x - y for x, y in both])) if both else None}
        summary["comparisons"][f"{a}_minus_{b}"] = comp
    oci.write_json(OUT / "bootstrap_summary.json", summary)

    prim = summary["comparisons"]["P_bc0.2_minus_O_bc0.2"]
    P = summary["policies"]["P_bc0.2"]
    strong = (prim["reach"]["ci95"][0] > 0 and prim["lower_route"]["mean"] >= .15 and prim["lower_route"]["ci95"][0] > 0
              and P["stuck_other"]["mean"] <= .10)
    provisional = (prim["lower_route"]["ci95"][0] > 0 and (prim["absorbed"]["mean"] <= 0 or prim["absorbed"]["ci95"][0] <= 0 <= prim["absorbed"]["ci95"][1])
                   and prim["reach"]["mean"] > 0)
    decision = {"strong_pass": bool(strong), "provisional_pass": bool(provisional and not strong),
                "primary_reach_diff": prim["reach"], "primary_lower_route_diff": prim["lower_route"],
                "primary_absorbed_diff": prim["absorbed"], "primary_return_diff": prim["discounted_return"],
                "P_bc0.2_stuck_other": P["stuck_other"]["mean"]}
    if not (strong or provisional):
        lr, rc = prim["lower_route"]["mean"], prim["reach"]["mean"]
        same = abs(summary["comparisons"]["P_bc0.2_minus_P_bc0.5"]["lower_route"]["mean"]) < .05
        decision["failure_case"] = ("(2) no lower-route shift" if prim["lower_route"]["ci95"][0] <= 0 else
                                    "(3) lower-route shift with stuck behaviour" if P["stuck_other"]["mean"] > .10 else
                                    "(1) lower-route shift without reach gain")
        if same:
            decision["failure_case"] += "; (4) P_bc0.2 ~ P_bc0.5"
    results = {"decision": decision, "summary": summary, "native_steps": 4 * EPISODES * HORIZON, "training_updates": 0,
               "elapsed_seconds": time.time() - t0, "manifest": manifest}
    oci.write_json(OUT / "results.json", results)
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
