"""Saved-rollout contrasts for the persistent-failure experiment; no model calls."""
import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))


def load(name):
    with np.load(OUT / name, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


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


def paired(left, right, weights):
    difference = np.asarray(left, np.float64) - np.asarray(right, np.float64)
    bootstrap = weights @ difference / weights.sum(axis=1)
    return {
        "direction": "left_minus_right",
        "difference": float(difference.mean()),
        "ci95": np.quantile(bootstrap, [0.025, 0.975]),
    }


def components(record):
    rewarded = record["return"] > 0
    failed = record["d"][:, :, -1]
    return {
        "probability_any_reward": rewarded.mean(axis=1),
        "probability_failure": failed.mean(axis=1),
        "probability_survive_without_reward": ((~rewarded) & (~failed)).mean(axis=1),
        "mean_discounted_return": record["return"].mean(axis=1),
    }


# Reuse the predeclared complete-trajectory bootstrap stream from run.py.
rng = np.random.default_rng(CONFIG["bootstrap_seed"] + 1)
weights = rng.multinomial(48, np.full(48, 1 / 48), CONFIG["bootstrap_replicates"])
records = {
    name: load(f"{name}_rollouts.npz")
    for name in [
        "baseline_p0", "baseline_p1", "joint_p0_h0", "joint_p0_h1",
        "joint_p1_h0", "joint_p1_h1",
    ]
}
values = {name: components(record) for name, record in records.items()}

result = {
    "bootstrap_seed": CONFIG["bootstrap_seed"] + 1,
    "replicates": CONFIG["bootstrap_replicates"],
    "joint_minus_matching_baseline": {},
    "head1_minus_head0_within_position": {},
    "position1_minus_position0_within_head": {},
}
for position in (0, 1):
    baseline = f"baseline_p{position}"
    for head in (0, 1):
        joint = f"joint_p{position}_h{head}"
        result["joint_minus_matching_baseline"][joint] = {
            metric: paired(values[joint][metric], values[baseline][metric], weights)
            for metric in values[joint]
        }
    left, right = f"joint_p{position}_h1", f"joint_p{position}_h0"
    result["head1_minus_head0_within_position"][f"p{position}"] = {
        metric: paired(values[left][metric], values[right][metric], weights)
        for metric in values[left]
    }
for head in (0, 1):
    left, right = f"joint_p1_h{head}", f"joint_p0_h{head}"
    result["position1_minus_position0_within_head"][f"h{head}"] = {
        metric: paired(values[left][metric], values[right][metric], weights)
        for metric in values[left]
    }

(OUT / "mechanism_contrasts.json").write_text(
    json.dumps(plain(result), indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
print(json.dumps({"status": "complete", "model_calls": 0, "native_steps": 0}))
