"""Vanilla CRL baselines next to the G3 policies on the same 200 native seeds.

See PROTOCOL.md.  Trains one actor (the original critic, bc = 0.2, the sealed
1,000 updates), evaluates it and the untouched original checkpoint natively,
and reuses the G3 trajectories for the other policies.
"""
import csv
import importlib.util
import json
import subprocess
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
from ett import pointmaze_offline_causal_integration as oci  # noqa: E402

OUT = Path(__file__).resolve().parent
G1 = ROOT / "outputs/pointmaze_absorbing_integration_20260915_v1"
G3 = ROOT / "outputs/pointmaze_absorbing_native_g3_20260915_v1"
BC = 0.2


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


g3 = load_module(G3 / "run.py", "g3_run")
rf = load_module(G3 / "reclassify.py", "g3_reclassify")


def build_actor_step(network, optimizer, bc_coef):
    import optax

    def objective(policy_params, q_params, batch, key):
        bc, critic, auxiliary = oci.actor_parts(network, policy_params, q_params, batch, key)
        return bc_coef * bc + (1. - bc_coef) * critic, (bc, critic, auxiliary)
    grad_fn = jax.value_and_grad(objective, has_aux=True)

    @jax.jit
    def step(policy_params, optimizer_state, q_params, batch, key):
        (loss, (bc, critic, auxiliary)), gradient = grad_fn(policy_params, q_params, batch, key)
        update, optimizer_state = optimizer.update(gradient, optimizer_state)
        return optax.apply_updates(policy_params, update), optimizer_state, loss
    return step


def main():
    t0 = time.time()
    obs, act, initial, train, heldout, plan = oci.load_prepared(G1)
    network, _, policy_optimizer = oci.make_network_and_optimizers()
    # 2. original critic, bc = 0.2 actor continuation (the G2 actor stage with the untouched critic)
    step = build_actor_step(network, policy_optimizer, BC)
    state = initial
    for u in range(oci.CONFIG["actor_updates"]):
        batch = oci.ordinary_batch(obs, act, plan, u)
        pp, po, loss = step(state.policy_params, state.policy_optimizer_state, state.q_params, batch, jnp.asarray(plan["actor_keys"][u]))
        assert np.isfinite(float(loss))
        state = state._replace(policy_params=pp, policy_optimizer_state=po)
    assert oci.tree_sha(state.q_params) == oci.tree_sha(initial.q_params)
    checkpoint.save_named(OUT / "checkpoints", "actor_vanilla_critic_bc0p2", 151000, state)
    params = {"vanilla_bc0.5": initial.policy_params, "vanilla_critic_bc0.2": state.policy_params}
    manifest = {
        "vanilla_bc0.5": {"path": str(oci.INITIAL), "sha256": oci.EXPECTED[str(oci.INITIAL)], "steps": 150000, "bc_coef": .5,
                          "policy_tree_sha256": oci.tree_sha(initial.policy_params), "note": "original offline CRL checkpoint, evaluated as trained"},
        "vanilla_critic_bc0.2": {"path": "checkpoints/actor_vanilla_critic_bc0p2.pkl", "critic": "original checkpoint critic, untouched",
                                 "actor_updates": oci.CONFIG["actor_updates"], "bc_coef": BC,
                                 "policy_tree_sha256": oci.tree_sha(state.policy_params),
                                 "policy_delta_l2_vs_initial": oci.tree_delta_norm(state.policy_params, initial.policy_params)},
        "reused_from_G3": {p: oci.file_sha(G3 / f"native_{p}.npz") for p in ["P_bc0.2", "O_bc0.2", "P_bc0.5", "O_bc0.5"]},
        "pairing": {"reset_seeds": f"{g3.RESET_SEED_BASE} + i, i < {g3.EPISODES}", "innovation": f"fold_in(PRNGKey({g3.ACTION_SEED_BASE} + t), i)"},
        "protocol_sha256": oci.file_sha(OUT / "PROTOCOL.md"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    oci.write_json(OUT / "checkpoint_manifest.json", manifest)

    recs = {}
    for name in ("vanilla_bc0.5", "vanilla_critic_bc0.2"):
        recs[name] = g3.run_policy(name, params[name], network)
        np.savez_compressed(OUT / f"native_{name}.npz", **recs[name], reset_seed=np.arange(g3.EPISODES) + g3.RESET_SEED_BASE)
    for name in ("P_bc0.2", "O_bc0.2", "P_bc0.5", "O_bc0.5"):
        recs[name] = dict(np.load(G3 / f"native_{name}.npz"))
    order = ["vanilla_bc0.5", "vanilla_critic_bc0.2", "O_bc0.5", "O_bc0.2", "P_bc0.5", "P_bc0.2"]
    per = {name: rf.classify(recs[name]) for name in order}
    fields = ["episode", "policy"] + rf.METRICS + ["route_class", "first_departure_cell", "time_to_goal", "death_time"]
    with open(OUT / "per_episode_regionfork.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name in order:
            for i, r in enumerate(per[name]):
                w.writerow({"episode": i, "policy": name, **{k: r[k] for k in fields[2:]}})
    summary = {"policies": {}, "comparisons": {}}
    for name in order:
        rows = per[name]
        s = {k: oci.bootstrap_mean(np.asarray([r[k] for r in rows], np.float64), 0) for k in rf.METRICS}
        ttg = [r["time_to_goal"] for r in rows if r["time_to_goal"] is not None]
        s["time_to_goal_given_reach"] = float(np.mean(ttg)) if ttg else None
        s["route_classes"] = {c: v / g3.EPISODES for c, v in Counter(r["route_class"] for r in rows).items()}
        s["first_departure_cells"] = {c: v / g3.EPISODES for c, v in Counter(r["first_departure_cell"] for r in rows).most_common()}
        for rc in ("lower", "shortcut"):
            sel = [r for r in rows if r["route_class"] == rc]
            s[f"reach_given_{rc}"] = float(np.mean([r["reach"] for r in sel])) if sel else None
        summary["policies"][name] = s
    pairs = [(name, "P_bc0.2") for name in order if name != "P_bc0.2"] + [("vanilla_critic_bc0.2", "vanilla_bc0.5"), ("O_bc0.2", "vanilla_critic_bc0.2")]
    for a, b in pairs:
        summary["comparisons"][f"{a}_minus_{b}"] = {k: rf.paired_bootstrap([r[k] for r in per[a]], [r[k] for r in per[b]], 100 + i) for i, k in enumerate(rf.METRICS)}
    summary["native_steps_new"] = 2 * g3.EPISODES * g3.HORIZON
    summary["training_updates"] = {"actor_vanilla_critic_bc0.2": oci.CONFIG["actor_updates"], "critic": 0}
    summary["elapsed_seconds"] = time.time() - t0
    oci.write_json(OUT / "results.json", {"manifest": manifest, "summary": summary})
    for name in order:
        s = summary["policies"][name]
        print(f"{name:22s} reach {s['reach']['mean']:.3f} strict {s['strict_success']['mean']:.3f} absorbed {s['absorbed']['mean']:.3f} "
              f"lower {s['lower_route']['mean']:.3f} shortcut {s['shortcut']['mean']:.3f} stuck {s['stuck_other']['mean']:.3f} "
              f"disc {s['discounted_return']['mean']:.2f} undisc {s['undiscounted_return']['mean']:.2f} ttg {s['time_to_goal_given_reach']}")
    for k, v in summary["comparisons"].items():
        print(k, {m: (round(v[m]["mean"], 3), [round(x, 3) for x in v[m]["ci95"]]) for m in ["reach", "absorbed", "lower_route", "shortcut", "discounted_return"]})


if __name__ == "__main__":
    main()
