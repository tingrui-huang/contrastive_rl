"""Post-run addendum: route classification with the fork defined as the start/fork
region {(0,3), (1,3)} instead of a visit to cell (1,3).

The sealed rule required the agent to end a step inside (1,3) before the route
could be classified; a down-right diagonal from the start cell passes through
(1,3) and ends in (1,2) within one step, which the sealed rule then filed under
"stuck/other" although the agent took the lower route. This script recomputes
every route metric from the saved native trajectories with the region
definition; nothing is re-run. Both classifications are reported.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ett import pointmaze_offline_causal_integration as oci  # noqa: E402
from ett.rollout_return import GOAL  # noqa: E402

OUT = Path(__file__).resolve().parent
POLICIES = ["P_bc0.2", "O_bc0.2", "P_bc0.5", "O_bc0.5"]
COMPARISONS = [("P_bc0.2", "O_bc0.2"), ("P_bc0.5", "O_bc0.5"), ("P_bc0.2", "P_bc0.5"), ("O_bc0.2", "O_bc0.5")]
HORIZON, DISCOUNT, EPISODES = 50, .95, 200
GOAL2 = np.asarray(GOAL[:2], np.float64)
METRICS = ["reach", "strict_success", "discounted_return", "undiscounted_return", "absorbed",
           "lower_route", "entered_1_2", "reached_y_below_2", "shortcut", "stuck_other",
           "first_departure_1_2", "first_departure_2_3", "hazard_landings"]


def cell(xy):
    return np.clip(np.floor(np.asarray(xy)).astype(int), [0, 0], [8, 4])


def hazardous(xy):
    return (xy[..., 0] >= 3) & (xy[..., 0] < 6) & (xy[..., 1] >= 3) & (xy[..., 1] < 4)


def classify(rec):
    phys, reward, dead = rec["physical"], rec["reward"], rec["dead"]
    rows = []
    for i in range(len(phys)):
        c = cell(phys[i]); haz = hazardous(phys[i])
        region = ((c[:, 0] == 0) | (c[:, 0] == 1)) & (c[:, 1] == 3)      # start/fork region
        outside = np.flatnonzero(~region)
        t_dep = int(outside[0]) if len(outside) else None                  # first state outside the region
        departure = c[t_dep] if t_dep is not None else np.array([-1, -1])
        in12 = (c[:, 0] == 1) & (c[:, 1] == 2)
        below = phys[i, :, 1] < 2.
        t12 = np.flatnonzero(in12 | below); thz = np.flatnonzero(haz)
        t_low = int(t12[0]) if len(t12) else None
        t_hz = int(thz[0]) if len(thz) else None
        if t_low is not None and (t_hz is None or t_low < t_hz):
            route = "lower"
        elif t_hz is not None:
            route = "shortcut"
        else:
            route = "stuck_other"
        reach = bool((reward[i] > 0).any())
        rows.append({
            "reach": float(reach), "strict_success": float(np.linalg.norm(phys[i] - GOAL2, axis=1).min() < .5),
            "discounted_return": float(reward[i] @ DISCOUNT ** np.arange(HORIZON)),
            "undiscounted_return": float(reward[i].sum()), "absorbed": float(dead[i, -1]),
            "lower_route": float(route == "lower"), "entered_1_2": float(in12.any()),
            "reached_y_below_2": float(below.any()), "shortcut": float(route == "shortcut"),
            "stuck_other": float(route == "stuck_other"),
            "first_departure_1_2": float(tuple(departure) == (1, 2)),
            "first_departure_2_3": float(tuple(departure) == (2, 3)),
            "hazard_landings": float((haz[1:] & ~np.concatenate([[False], dead[i, :-1]])).sum()),
            "route_class": route, "departure_time": t_dep,
            "first_departure_cell": f"({int(departure[0])},{int(departure[1])})" if t_dep is not None else "none",
            "time_to_goal": int(np.argmax(reward[i] > 0) + 1) if reach else None,
            "death_time": int(np.argmax(dead[i]) + 1) if dead[i].any() else None})
    return rows


def paired_bootstrap(a, b, seed, replicates=2000):
    rng = np.random.default_rng(seed)
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    draws = d[rng.integers(0, len(d), (replicates, len(d)))].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def main():
    per = {p: classify(dict(np.load(OUT / f"native_{p}.npz"))) for p in POLICIES}
    fields = ["episode", "policy"] + METRICS + ["route_class", "departure_time", "first_departure_cell", "time_to_goal", "death_time"]
    with open(OUT / "per_episode_regionfork.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for p in POLICIES:
            for i, r in enumerate(per[p]):
                w.writerow({"episode": i, "policy": p, **{k: r[k] for k in fields[2:]}})
    summary = {"policies": {}, "comparisons": {}}
    for p in POLICIES:
        rows = per[p]
        s = {k: oci.bootstrap_mean(np.asarray([r[k] for r in rows], np.float64), 0) for k in METRICS}
        ttg = [r["time_to_goal"] for r in rows if r["time_to_goal"] is not None]
        s["time_to_goal_given_reach"] = {"mean": float(np.mean(ttg)) if ttg else None, "n": len(ttg)}
        s["first_departure_cells"] = {c: v / EPISODES for c, v in Counter(r["first_departure_cell"] for r in rows).most_common()}
        s["route_classes"] = {c: v / EPISODES for c, v in Counter(r["route_class"] for r in rows).items()}
        for rc in ("lower", "shortcut"):
            sel = [r for r in rows if r["route_class"] == rc]
            s[f"reach_given_{rc}"] = float(np.mean([r["reach"] for r in sel])) if sel else None
            s[f"absorbed_given_{rc}"] = float(np.mean([r["absorbed"] for r in sel])) if sel else None
            s[f"time_to_goal_given_{rc}_reach"] = float(np.mean([r["time_to_goal"] for r in sel if r["time_to_goal"] is not None])) if any(r["time_to_goal"] is not None for r in sel) else None
        summary["policies"][p] = s
    for a, b in COMPARISONS:
        summary["comparisons"][f"{a}_minus_{b}"] = {k: paired_bootstrap([r[k] for r in per[a]], [r[k] for r in per[b]], 100 + i)
                                                    for i, k in enumerate(METRICS)}
    prim = summary["comparisons"]["P_bc0.2_minus_O_bc0.2"]; P = summary["policies"]["P_bc0.2"]
    strong = (prim["reach"]["ci95"][0] > 0 and prim["lower_route"]["mean"] >= .15 and prim["lower_route"]["ci95"][0] > 0 and P["stuck_other"]["mean"] <= .10)
    provisional = (prim["lower_route"]["ci95"][0] > 0 and (prim["absorbed"]["mean"] <= 0 or prim["absorbed"]["ci95"][0] <= 0 <= prim["absorbed"]["ci95"][1]) and prim["reach"]["mean"] > 0)
    summary["decision_regionfork"] = {"strong_pass": bool(strong), "provisional_pass": bool(provisional and not strong),
                                      "P_bc0.2_stuck_other": P["stuck_other"]["mean"],
                                      "note": "fork = start/fork region {(0,3),(1,3)}; post-run classification addendum, not the sealed rule"}
    # how the sealed 'stuck/other' episodes reclassify
    sealed = list(csv.DictReader(open(OUT / "per_episode.csv")))
    recl = {}
    for p in POLICIES:
        idx = [int(r["episode"]) for r in sealed if r["policy"] == p and r["route_class"] == "stuck_other"]
        recl[p] = {"sealed_stuck_other": len(idx), "reclassified": dict(Counter(per[p][i]["route_class"] for i in idx))}
    summary["sealed_stuck_other_reclassified"] = recl
    oci.write_json(OUT / "bootstrap_summary_regionfork.json", summary)
    print(json.dumps({"decision_regionfork": summary["decision_regionfork"], "reclassified": recl}, indent=1))
    for p in POLICIES:
        s = summary["policies"][p]
        print(f"{p}: reach {s['reach']['mean']:.3f} absorbed {s['absorbed']['mean']:.3f} lower {s['lower_route']['mean']:.3f} shortcut {s['shortcut']['mean']:.3f} "
              f"stuck {s['stuck_other']['mean']:.3f} dep(1,2) {s['first_departure_1_2']['mean']:.3f} dep(2,3) {s['first_departure_2_3']['mean']:.3f} "
              f"reach|lower {s['reach_given_lower']} reach|short {s['reach_given_shortcut']} ttg|lower {s['time_to_goal_given_lower_reach']} ttg|short {s['time_to_goal_given_shortcut_reach']}")
    for c, v in summary["comparisons"].items():
        print(c, {k: (round(v[k]["mean"], 3), [round(x, 3) for x in v[k]["ci95"]]) for k in ["reach", "discounted_return", "absorbed", "lower_route", "shortcut", "stuck_other", "first_departure_1_2"]})


if __name__ == "__main__":
    main()
