"""Shared frozen-model, accounting, metric, and bootstrap utilities."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
MATCHED = ROOT / "outputs" / "pointmaze_matched_failure_audit_20260914_v1"
PERSISTENT = ROOT / "outputs" / "pointmaze_persistent_failure_20260914_v1"
RESPONSE = ROOT / "outputs" / "pointmaze_response_gate_20260914_v1"
SUPERVISED = ROOT / "outputs" / "supervised_stochastic_ett_20260914_v1"
REVIEW = Path(r"C:\Users\trhua\Documents\Codex\2026-09-08\f\work\matched-audit-88ec819")
OLD_TEACHER = Path(
    r"C:\Users\trhua\Documents\Codex\2026-09-08\f\work\branch-source-f85a5f4"
    r"\scripts\collect_swamp_windy.py"
)
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(ROOT))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PARENT = load_module("repair_persistent_parent", PERSISTENT / "run.py")
RESPONSE_MODULE = PARENT.RESPONSE_MODULE


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


def write_json(path, value):
    Path(path).write_text(
        json.dumps(plain(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hazardous(xy):
    xy = np.asarray(xy)
    return (
        (xy[..., 0] >= 3)
        & (xy[..., 0] < 6)
        & (xy[..., 1] >= 3)
        & (xy[..., 1] < 4)
    )


def input_groups(state):
    xy = np.asarray(state)[:, :2]
    approach = (xy[:, 0] >= 2) & (xy[:, 0] < 3) & (xy[:, 1] >= 3) & (xy[:, 1] < 4)
    inside = hazardous(xy)
    right = (xy[:, 0] >= 6) & (xy[:, 0] < 7) & (xy[:, 1] >= 3) & (xy[:, 1] < 4)
    return OrderedDict([
        ("all", np.ones(len(xy), bool)),
        ("approach_left", approach),
        ("inside_hazard", inside),
        ("right_exit", right),
        ("elsewhere", ~(approach | inside | right)),
    ])


def exact_diagonal(data):
    return np.all(data["xq"] == data["xb"], axis=1)


def full_energy_score(samples, target):
    samples = np.asarray(samples, np.float64)
    target = np.asarray(target, np.float64)
    result = np.empty(len(samples), np.float64)
    count = samples.shape[1]
    for start in range(0, len(samples), 128):
        stop = min(start + 128, len(samples))
        value = samples[start:stop]
        first = np.linalg.norm(value - target[start:stop, None], axis=-1).mean(axis=1)
        pairs = np.linalg.norm(value[:, :, None] - value[:, None, :], axis=-1)
        result[start:stop] = first - pairs.sum(axis=(1, 2)) / (2 * count * (count - 1))
    return result


def load_head_scaling():
    scaling = load_npz(PERSISTENT / "head_feature_scaling.npz")
    return scaling["state_mean"], scaling["state_std"]


def old_head(seed):
    return load_npz(PERSISTENT / f"head_s{seed}.npz")["theta"]


def old_position(seed):
    return load_npz(RESPONSE / f"candidate_s{seed}.npz")["theta"]


def supported_head_probability(theta, state, xb, xq, next_xy, mean, std):
    features = PARENT.head_features(state, xb, xq, next_xy, mean, std)
    raw = PARENT.predict_numpy(theta, features)
    return raw * hazardous(next_xy)


class Ledger:
    def __init__(self):
        path = OUT / "ledger.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.native_counts = OrderedDict(saved["native_steps"]["counts"])
            self.position_counts = OrderedDict(saved["position_successors"]["counts"])
            self.head_counts = OrderedDict(saved["head_rows"]["counts"])
            self.update_counts = OrderedDict(saved["training_updates"]["counts"])
            self.path_counts = OrderedDict(saved["complete_model_paths"]["counts"])
        else:
            self.native_counts = OrderedDict()
            self.position_counts = OrderedDict()
            self.head_counts = OrderedDict()
            self.update_counts = OrderedDict()
            self.path_counts = OrderedDict()
            self.save()

    @staticmethod
    def total(counts):
        return int(sum(counts.values()))

    def _charge(self, counts, number, purpose, cap, label):
        number = int(number)
        if purpose in counts:
            return False
        if self.total(counts) + number > cap:
            raise RuntimeError(f"{label} cap exhausted")
        counts[purpose] = number
        self.save()
        return True

    def native(self, number, purpose):
        return self._charge(self.native_counts, number, purpose, CONFIG["native_step_cap"], "native step")

    def position(self, number, purpose):
        return self._charge(self.position_counts, number, purpose, CONFIG["position_successor_cap"], "position successor")

    def heads(self, number, purpose):
        return self._charge(self.head_counts, number, purpose, 10**12, "head row")

    def updates(self, number, purpose):
        return self._charge(self.update_counts, number, purpose, CONFIG["training_update_cap"], "training update")

    def paths(self, number, purpose):
        return self._charge(self.path_counts, number, purpose, CONFIG["complete_model_path_cap"], "complete path")

    def save(self):
        write_json(OUT / "ledger.json", {
            "native_steps": {"counts": self.native_counts, "total": self.total(self.native_counts), "cap": CONFIG["native_step_cap"]},
            "position_successors": {"counts": self.position_counts, "total": self.total(self.position_counts), "cap": CONFIG["position_successor_cap"]},
            "head_rows": {"counts": self.head_counts, "total": self.total(self.head_counts)},
            "training_updates": {"counts": self.update_counts, "total": self.total(self.update_counts), "cap": CONFIG["training_update_cap"]},
            "complete_model_paths": {"counts": self.path_counts, "total": self.total(self.path_counts), "cap": CONFIG["complete_model_path_cap"]},
        })


def bootstrap_weights():
    path = OUT / "bootstrap_weights.npz"
    if path.exists():
        return load_npz(path)["weights"]
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    episodes = CONFIG["splits"]["final"]["episodes"]
    weights = rng.multinomial(episodes, np.full(episodes, 1 / episodes), CONFIG["bootstrap_replicates"])
    np.savez_compressed(path, weights=weights)
    return weights


def row_components(values, episodes, mask):
    values = np.asarray(values, np.float64)
    episodes = np.asarray(episodes, int)
    mask = np.asarray(mask, bool)
    count = CONFIG["splits"]["final"]["episodes"]
    sums = np.zeros(count, np.float64)
    denominators = np.zeros(count, np.float64)
    np.add.at(sums, episodes[mask], values[mask])
    np.add.at(denominators, episodes[mask], 1)
    return sums, denominators


def ratio_summary(numerator, denominator, weights):
    numerator = np.asarray(numerator, np.float64)
    denominator = np.asarray(denominator, np.float64)
    if denominator.sum() == 0:
        return {"estimate": None, "ci95": [None, None]}
    den = weights @ denominator
    valid = den > 0
    replicates = (weights[valid] @ numerator) / den[valid]
    return {"estimate": float(numerator.sum() / denominator.sum()), "ci95": np.quantile(replicates, [0.025, 0.975])}


def row_mean_summary(values, episodes, mask, weights):
    return ratio_summary(*row_components(values, episodes, mask), weights)


def paired_row_difference(left, right, episodes, mask, weights):
    return row_mean_summary(np.asarray(left) - np.asarray(right), episodes, mask, weights)


def binary_summary(probability, target, episodes, mask, weights):
    probability = np.asarray(probability, np.float64)
    target = np.asarray(target, np.float64)
    mask = np.asarray(mask, bool)
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    return {
        "rows": int(mask.sum()),
        "episodes": int(len(np.unique(np.asarray(episodes)[mask]))),
        "events": int(target[mask].sum()),
        "observed_rate": row_mean_summary(target, episodes, mask, weights),
        "predicted_rate": row_mean_summary(probability, episodes, mask, weights),
        "calibration_bias": row_mean_summary(probability - target, episodes, mask, weights),
        "brier": row_mean_summary((probability - target) ** 2, episodes, mask, weights),
        "log_loss": row_mean_summary(-(target * np.log(clipped) + (1 - target) * np.log1p(-clipped)), episodes, mask, weights),
    }


def scalar_summary(per_episode, weights):
    value = np.asarray(per_episode, np.float64)
    replicates = weights @ value / weights.sum(axis=1)
    return {"estimate": float(value.mean()), "ci95": np.quantile(replicates, [0.025, 0.975])}


def paired_scalar_difference(left, right, weights):
    return scalar_summary(np.asarray(left) - np.asarray(right), weights)


def paired_ratio_difference(left_num, left_den, right_num, right_den, weights):
    left_num = np.asarray(left_num, np.float64)
    left_den = np.asarray(left_den, np.float64)
    right_num = np.asarray(right_num, np.float64)
    right_den = np.asarray(right_den, np.float64)
    if left_den.sum() == 0 or right_den.sum() == 0:
        return {"estimate": None, "ci95": [None, None]}
    ld, rd = weights @ left_den, weights @ right_den
    valid = (ld > 0) & (rd > 0)
    replicates = (weights[valid] @ left_num) / ld[valid] - (weights[valid] @ right_num) / rd[valid]
    return {
        "estimate": float(left_num.sum() / left_den.sum() - right_num.sum() / right_den.sum()),
        "ci95": np.quantile(replicates, [0.025, 0.975]),
    }
