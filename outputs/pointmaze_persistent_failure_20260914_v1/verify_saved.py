"""Independent saved-artifact verification; makes no model or environment calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


OUT = Path(__file__).resolve().parent
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
GOAL_XY = np.array([8.5, 3.5], np.float32)
WALLS = np.array(
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
    np.int8,
)


def read_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def sha256(path):
    with Path(path).open("rb") as stream:
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


def write_json(name, value):
    (OUT / name).write_text(
        json.dumps(plain(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def assert_close(actual, expected, atol=1e-12):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def hazard(xy):
    xy = np.asarray(xy)
    return (
        (xy[..., 0] >= 3)
        & (xy[..., 0] < 6)
        & (xy[..., 1] >= 3)
        & (xy[..., 1] < 4)
    )


def legal(xy):
    xy = np.asarray(xy)
    floor = np.floor(xy).astype(np.int64)
    bounded = (
        (xy[..., 0] >= 0)
        & (xy[..., 0] <= 9)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] <= 5)
    )
    cell = WALLS[
        np.clip(floor[..., 0], 0, 8), np.clip(floor[..., 1], 0, 4)
    ]
    return bounded & (cell == 0)


# Hash, freeze, and budget contract.
provenance = read_json("provenance.json")
started = read_json("execution_started.json")
completion = read_json("completion.json")
ledger = read_json("ledger.json")
frozen = read_json("frozen_settings.json")
assert completion["status"] == "complete"
assert started["protocol_sha256"] == sha256(OUT / "PROTOCOL.md")
assert started["config_sha256"] == sha256(OUT / "config.json")
assert started["collector_sha256"] == sha256(OUT / "collect_evaluation.py")
assert started["driver_sha256"] == sha256(OUT / "run.py")
for path, expected in provenance["dependency_sha256"].items():
    assert sha256(path) == expected, path
assert frozen["configuration_sha256"] == sha256(OUT / "config.json")
assert frozen["protocol_sha256"] == sha256(OUT / "PROTOCOL.md")
for seed in (0, 1):
    assert frozen["head_checkpoint_sha256"][f"h{seed}"] == sha256(OUT / f"head_s{seed}.npz")
assert ledger["position_model"]["total"] == completion["position_model_successors"] == 2151424
assert ledger["position_model"]["total"] < CONFIG["position_successor_cap"]
assert ledger["native"]["total"] == completion["native_steps"] == 14400
assert ledger["native"]["total"] < CONFIG["native_step_cap"]
assert ledger["head_only"]["total"] == 1024000
assert completion["position_model_updates"] == 0
assert completion["actor_updates"] == completion["critic_updates"] == completion["nominal_updates"] == 0
assert completion["explicit_failure_label_supervision"]
assert not completion["evaluation_tuning"]


# Independently reconstruct both MLP training runs.
HEAD_DIM = 14
HIDDEN = 32
W1 = slice(0, HEAD_DIM * HIDDEN)
B1 = slice(W1.stop, W1.stop + HIDDEN)
W2 = slice(B1.stop, B1.stop + HIDDEN * HIDDEN)
B2 = slice(W2.stop, W2.stop + HIDDEN)
W3 = slice(B2.stop, B2.stop + HIDDEN)
B3 = W3.stop
PARAM_DIM = B3 + 1


def logits(theta, features):
    first = jnp.tanh(features @ theta[W1].reshape(HEAD_DIM, HIDDEN) + theta[B1])
    second = jnp.tanh(first @ theta[W2].reshape(HIDDEN, HIDDEN) + theta[B2])
    return second @ theta[W3] + theta[B3]


def loss_function(theta, features, target):
    value = logits(theta, features)
    return jnp.mean(jax.nn.softplus(value) - target * value)


value_gradient = jax.jit(jax.value_and_grad(loss_function))
predict = jax.jit(lambda theta, features: jax.nn.sigmoid(logits(theta, features)))


def initialize(seed):
    rng = np.random.default_rng(CONFIG["head_initialization_seed_base"] + seed)
    theta = np.zeros(PARAM_DIM, np.float32)
    theta[W1] = rng.uniform(
        -np.sqrt(6 / (HEAD_DIM + HIDDEN)), np.sqrt(6 / (HEAD_DIM + HIDDEN)), W1.stop
    )
    theta[W2] = rng.uniform(
        -np.sqrt(6 / (2 * HIDDEN)), np.sqrt(6 / (2 * HIDDEN)), W2.stop - W2.start
    )
    theta[W3] = rng.uniform(
        -np.sqrt(6 / (HIDDEN + 1)), np.sqrt(6 / (HIDDEN + 1)), W3.stop - W3.start
    )
    return theta


def features(state, xb, xq, next_xy, mean, std):
    next_xy = np.asarray(next_xy, np.float32)
    if next_xy.ndim == 3:
        state = np.broadcast_to(state[:, None], next_xy.shape[:2] + (8,))
        xb = np.broadcast_to(xb[:, None], next_xy.shape[:2] + (2,))
        xq = np.broadcast_to(xq[:, None], next_xy.shape[:2] + (2,))
    return np.concatenate(
        [(state - mean) / std, xb, xq, next_xy - state[..., :2]], axis=-1
    ).astype(np.float32)


def predict_numpy(theta, values, chunk=65536):
    flat = values.reshape(-1, HEAD_DIM)
    output = []
    for start in range(0, len(flat), chunk):
        output.append(np.asarray(predict(jnp.asarray(theta), jnp.asarray(flat[start:start + chunk]))))
    return np.concatenate(output).reshape(values.shape[:-1])


train_path = next(
    Path(path)
    for path in provenance["dependency_sha256"]
    if path.endswith("paired_transitions.npz")
)
train = load_npz(train_path)
train_rows = train["episode"] <= 71
alive_train = train_rows & (~train["dead_before"])
scaling = load_npz(OUT / "head_feature_scaling.npz")
mean = train["s"][train_rows].mean(axis=0).astype(np.float32)
std = np.maximum(
    train["s"][train_rows].std(axis=0).astype(np.float32), CONFIG["state_scale_floor"]
).astype(np.float32)
np.testing.assert_array_equal(mean, scaling["state_mean"])
np.testing.assert_array_equal(std, scaling["state_std"])
train_features = features(
    train["s"][alive_train], train["xb"][alive_train], train["xq"][alive_train],
    train["y"][alive_train, :2], mean, std,
)
train_target = train["dead_after"][alive_train].astype(np.float32)
assert len(train_target) == 9248 and int(train_target.sum()) == 214
head_parameters = {}
training_checks = {}
for seed in (0, 1):
    saved = load_npz(OUT / f"head_s{seed}.npz")
    theta = initialize(seed)
    np.testing.assert_array_equal(theta, saved["initial_theta"])
    adam_m = np.zeros(PARAM_DIM, np.float64)
    adam_v = np.zeros(PARAM_DIM, np.float64)
    rng = np.random.default_rng(CONFIG["head_batch_seed_base"] + seed)
    for step in range(CONFIG["head_updates"]):
        batch = rng.choice(len(train_target), CONFIG["head_batch_size"])
        loss, gradient = value_gradient(
            jnp.asarray(theta), jnp.asarray(train_features[batch]), jnp.asarray(train_target[batch])
        )
        gradient = np.asarray(gradient, np.float64)
        adam_m = CONFIG["adam_beta1"] * adam_m + (1 - CONFIG["adam_beta1"]) * gradient
        adam_v = CONFIG["adam_beta2"] * adam_v + (1 - CONFIG["adam_beta2"]) * gradient**2
        age = step + 1
        update = CONFIG["head_learning_rate"] * (
            adam_m / (1 - CONFIG["adam_beta1"] ** age)
        ) / (
            np.sqrt(adam_v / (1 - CONFIG["adam_beta2"] ** age)) + CONFIG["adam_epsilon"]
        )
        theta = (theta - update).astype(np.float32)
        assert_close(float(loss), saved["loss"][step], atol=1e-7)
        assert_close(np.linalg.norm(gradient), saved["gradient_norm"][step], atol=1e-10)
        assert_close(np.linalg.norm(update), saved["update_norm"][step], atol=1e-10)
        assert_close(train_target[batch].mean(), saved["batch_positive_fraction"][step], atol=1e-12)
        assert_close(np.linalg.norm(theta), saved["theta_norm"][step], atol=1e-6)
    np.testing.assert_array_equal(theta, saved["theta"])
    assert_close(adam_m, saved["adam_m"], atol=1e-12)
    assert_close(adam_v, saved["adam_v"], atol=1e-12)
    probability = predict_numpy(theta, train_features)
    assert probability.std() > 1e-4 and np.ptp(probability) > 1e-3
    summary = read_json(f"head_s{seed}_training_summary.json")
    assert summary["not_constant"]
    assert_close(probability.mean(), summary["prediction_mean"], atol=1e-7)
    assert_close(probability.std(), summary["prediction_std"], atol=1e-7)
    head_parameters[seed] = theta
    training_checks[f"h{seed}"] = {
        "updates_reconstructed": 2000,
        "final_parameters_exact": True,
        "optimizer_moments_reconstructed": True,
        "alive_rows_only": True,
        "unweighted_bce": True,
        "prediction_not_constant": True,
    }


# New paired-set alignment and exact onset arrays.
paired = load_npz(OUT / "paired_evaluation.npz")
manifest = read_json("paired_evaluation_collection.json")
assert paired["s"].shape == (9600, 8)
assert manifest["base_seed"] == CONFIG["paired_collection_seed"]
assert manifest["native_steps"] == 12000
assert manifest["output_sha256"] == sha256(OUT / "paired_evaluation.npz")
assert np.all(np.bincount(paired["episode"]) == 200)
for offset in range(0, 9600, 4):
    block = np.arange(offset, offset + 4)
    assert set(paired["query"][block]) == {0, 1, 2, 3}
    assert len(set(zip(paired["episode"][block], paired["time"][block]))) == 1
    np.testing.assert_array_equal(paired["s"][block], np.broadcast_to(paired["s"][block[0]], (4, 8)))
    np.testing.assert_array_equal(paired["xb"][block], np.broadcast_to(paired["xb"][block[0]], (4, 2)))
    q0 = block[paired["query"][block] == 0][0]
    np.testing.assert_array_equal(paired["xq"][q0], paired["xb"][q0])
np.testing.assert_array_equal(paired["y"][:, 2:], paired["s"][:, :6])
alive = ~paired["dead_before"]
diagonal = np.all(paired["xb"] == paired["xq"], axis=1)
landing_hazard = hazard(paired["y"][:, :2])
assert not paired["dead_after"][alive & ~landing_hazard].any()
groups = {
    "all_alive": alive,
    "diagonal": alive & diagonal,
    "off_diagonal": alive & ~diagonal,
    "hazardous_landing": alive & landing_hazard,
    "nonhazardous_landing": alive & ~landing_hazard,
    "diagonal_hazardous": alive & diagonal & landing_hazard,
    "diagonal_nonhazardous": alive & diagonal & ~landing_hazard,
    "off_diagonal_hazardous": alive & ~diagonal & landing_hazard,
    "off_diagonal_nonhazardous": alive & ~diagonal & ~landing_hazard,
}
saved_groups = load_npz(OUT / "onset_groups.npz")
for label, mask in groups.items():
    np.testing.assert_array_equal(mask, saved_groups[label])

actual_feature = features(paired["s"], paired["xb"], paired["xq"], paired["y"][:, :2], mean, std)
onset_array_checks = {}
for head_seed in (0, 1):
    saved = load_npz(OUT / f"head_h{head_seed}_actual_successor_evaluation.npz")
    probability = predict_numpy(head_parameters[head_seed], actual_feature)
    assert_close(probability, saved["probability"], atol=1e-7)
    np.testing.assert_array_equal(saved["target"], paired["dead_after"])
    np.testing.assert_array_equal(saved["alive_before"], alive)
    onset_array_checks[f"actual_h{head_seed}"] = True

for position_seed in (0, 1):
    sampled = load_npz(OUT / f"position_p{position_seed}_one_step_samples.npz")
    assert sampled["samples_xy"].shape == (9600, 64, 2)
    np.testing.assert_array_equal(legal(sampled["samples_xy"]), sampled["legal"])
    assert sampled["legal"].all()
    assert np.all((sampled["h"] >= 0) & (sampled["h"] <= 1))
    sampled_feature = features(
        paired["s"], paired["xb"], paired["xq"], sampled["samples_xy"], mean, std
    )
    for head_seed in (0, 1):
        saved = load_npz(OUT / f"joint_p{position_seed}_h{head_seed}_onset_evaluation.npz")
        probability = predict_numpy(head_parameters[head_seed], sampled_feature)
        assert_close(probability, saved["sample_probability"], atol=1e-7)
        np.testing.assert_array_equal(probability.mean(axis=1), saved["mean_probability"])
        onset_array_checks[f"joint_p{position_seed}_h{head_seed}"] = True


# Reconstruct the reported primary onset metrics and their episode bootstraps.
onset_metrics = read_json("onset_metrics.json")
rng = np.random.default_rng(CONFIG["bootstrap_seed"])
row_weights = rng.multinomial(48, np.full(48, 1 / 48), CONFIG["bootstrap_replicates"])


def row_bootstrap(value, mask):
    count = np.array([np.sum(mask & (paired["episode"] == item)) for item in range(48)])
    total = np.array([value[mask & (paired["episode"] == item)].sum() for item in range(48)])
    denominator = row_weights @ count
    valid = denominator > 0
    return value[mask].mean(), np.quantile((row_weights[valid] @ total) / denominator[valid], [0.025, 0.975])


def verify_onset_summary(probability, reported):
    target = paired["dead_after"].astype(np.float64)
    probability = probability.astype(np.float64)
    brier = (probability - target) ** 2
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    logloss = -(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))
    for label, mask in groups.items():
        assert reported[label]["rows"] == int(mask.sum())
        for key, value in [
            ("observed_rate", target),
            ("predicted_rate", probability),
            ("brier", brier),
            ("log_loss", logloss),
        ]:
            point, interval = row_bootstrap(value, mask)
            assert_close(point, reported[label][key]["estimate"], atol=1e-12)
            assert_close(interval, reported[label][key]["ci95"], atol=1e-12)
        point, interval = row_bootstrap(probability - target, mask)
        assert_close(point, reported[label]["predicted_minus_observed"]["estimate"], atol=1e-12)
        assert_close(interval, reported[label]["predicted_minus_observed"]["ci95"], atol=1e-12)


for head_seed in (0, 1):
    probability = load_npz(OUT / f"head_h{head_seed}_actual_successor_evaluation.npz")["probability"]
    verify_onset_summary(probability, onset_metrics["actual_successor"][f"h{head_seed}"])
for position_seed in (0, 1):
    for head_seed in (0, 1):
        probability = load_npz(OUT / f"joint_p{position_seed}_h{head_seed}_onset_evaluation.npz")["mean_probability"]
        verify_onset_summary(
            probability,
            onset_metrics["sampled_position_joint"][f"p{position_seed}"][f"h{head_seed}"],
        )


# Native evaluation and all six trajectory arrays.
native = load_npz(OUT / "native_actor_episodes.npz")
assert native["states"].shape == (48, 51, 8)
np.testing.assert_array_equal(native["states"][:, 1:, 2:], native["states"][:, :-1, :6])
np.testing.assert_array_equal(native["onset"], (~native["dead_before"]) & native["dead_after"])
np.testing.assert_array_equal(native["dead_before"][:, 1:], native["dead_after"][:, :-1])
assert np.all(native["dead_after"] >= native["dead_before"])
assert not (native["onset"] & ~hazard(native["states"][:, 1:, :2])).any()
native_reward = (
    (np.linalg.norm(native["states"][:, 1:, :2] - GOAL_XY, axis=-1) < 2)
    & (~native["dead_after"])
).astype(np.float64)
np.testing.assert_array_equal(native_reward, native["rewards"])
native_return = np.sum(
    native_reward * CONFIG["return_discount"] ** np.arange(50), axis=1
)
assert_close(native_return, native["returns"], atol=2e-12)
np.testing.assert_array_equal(
    native["environment_seed"], np.arange(48) + CONFIG["return_environment_seed"]
)

arm_names = [
    "baseline_p0", "baseline_p1", "joint_p0_h0", "joint_p0_h1", "joint_p1_h0", "joint_p1_h1"
]
records = {}
trajectory_checks = {}
discount32 = np.power(np.float32(CONFIG["return_discount"]), np.arange(50), dtype=np.float32)
for name in arm_names:
    record = load_npz(OUT / f"{name}_rollouts.npz")
    assert record["states"].shape == (48, 64, 51, 8)
    np.testing.assert_array_equal(
        record["states"][:, :, 0], np.broadcast_to(native["states"][:, None, 0], (48, 64, 8))
    )
    np.testing.assert_array_equal(record["states"][:, :, 1:, 2:], record["states"][:, :, :-1, :6])
    assert legal(record["states"][:, :, 1:, :2]).all()
    np.testing.assert_array_equal(record["d"][:, :, :-1], record["d_before"])
    np.testing.assert_array_equal(record["d"][:, :, 1:], record["d_after"])
    np.testing.assert_array_equal(record["onset"], (~record["d_before"]) & record["d_after"])
    assert np.all(record["d_after"] >= record["d_before"])
    np.testing.assert_array_equal(record["hazard_landing"], hazard(record["position_proposal_xy"]))
    failed = record["d_before"]
    np.testing.assert_array_equal(
        record["states"][:, :, 1:, :2][failed], record["states"][:, :, :-1, :2][failed]
    )
    onset = record["onset"]
    np.testing.assert_array_equal(
        record["states"][:, :, 1:, :2][onset], record["position_proposal_xy"][onset]
    )
    reward = (
        (np.linalg.norm(record["states"][:, :, 1:, :2] - GOAL_XY, axis=-1) < 2)
        & (~record["d_after"])
    ).astype(np.float32)
    np.testing.assert_array_equal(reward, record["reward"])
    recomputed_return = np.sum(reward * discount32, axis=-1, dtype=np.float32)
    assert_close(recomputed_return, record["return"], atol=1e-6)
    assert np.all((record["failure_probability"] >= 0) & (record["failure_probability"] <= 1))
    assert not record["failure_probability"][failed].any()
    if name.startswith("baseline"):
        assert not record["d"].any()
        assert not record["onset"].any()
        assert not record["failure_probability"].any()
    records[name] = record
    trajectory_checks[name] = {
        "common_native_roots": True,
        "f4_shift": True,
        "position_geometry_legal": True,
        "mode_monotone": True,
        "absorbing_xy": True,
        "fatal_onset_landing_preserved": True,
        "joint_reward_reconstructed": True,
        "discounted_return_reconstructed": True,
    }

for position_seed in (0, 1):
    baseline = records[f"baseline_p{position_seed}"]
    for head_seed in (0, 1):
        joint = records[f"joint_p{position_seed}_h{head_seed}"]
        before_failure = ~joint["d_before"]
        for field in ["action", "xb", "position_proposal_xy"]:
            np.testing.assert_array_equal(joint[field][before_failure], baseline[field][before_failure])


# Reconstruct primary trajectory point estimates and paired bootstrap intervals.
reported_trajectory = read_json("trajectory_metrics.json")
rng = np.random.default_rng(CONFIG["bootstrap_seed"] + 1)
weights = rng.multinomial(48, np.full(48, 1 / 48), CONFIG["bootstrap_replicates"])


def bootstrap_scalar(value):
    value = np.asarray(value, np.float64)
    return value.mean(), np.quantile(weights @ value / weights.sum(axis=1), [0.025, 0.975])


native_any = native_return > 0
native_failed = native["dead_after"].any(axis=1)
native_survive = (~native_failed) & (~native_any)
native_values = {
    "probability_any_reward": native_any.astype(float),
    "probability_failure": native_failed.astype(float),
    "probability_survive_without_reward": native_survive.astype(float),
    "mean_discounted_return": native_return,
}
for key, value in native_values.items():
    point, interval = bootstrap_scalar(value)
    assert_close(point, reported_trajectory["native"][key]["estimate"], atol=1e-12)
    assert_close(interval, reported_trajectory["native"][key]["ci95"], atol=1e-12)

for name, record in records.items():
    any_reward = record["return"] > 0
    failed = record["d"][:, :, -1]
    survive = (~failed) & (~any_reward)
    model_values = {
        "probability_any_reward": any_reward.mean(axis=1),
        "probability_failure": failed.mean(axis=1),
        "probability_survive_without_reward": survive.mean(axis=1),
        "mean_discounted_return": record["return"].mean(axis=1).astype(np.float64),
    }
    for key, value in model_values.items():
        point, interval = bootstrap_scalar(value)
        assert_close(point, reported_trajectory["models"][name][key]["estimate"], atol=1e-12)
        assert_close(interval, reported_trajectory["models"][name][key]["ci95"], atol=1e-12)
        gap_point, gap_interval = bootstrap_scalar(value - native_values[key])
        assert_close(gap_point, reported_trajectory["gaps_from_native"][name][key]["estimate"], atol=1e-12)
        assert_close(gap_interval, reported_trajectory["gaps_from_native"][name][key]["ci95"], atol=1e-12)

checks = read_json("contract_checks.json")
assert checks["contract"]["internal_mode_not_given_to_actor_or_nominal"]
assert checks["contract"]["native_labels_not_referenced_by_rollout"]
assert not checks["contract"]["joint_action_lipschitz_claim_made"]
assert all(checks[f"joint_p{p}_h{h}"]["failure_support_mask_applied"] is False for p in (0, 1) for h in (0, 1))

result = {
    "status": "pass",
    "scope": "saved arrays and hashed inputs only; zero position-model, actor, nominal, or environment calls",
    "hashes": {
        "all_recorded_dependencies_unchanged": True,
        "dependency_count": len(provenance["dependency_sha256"]),
        "settings_frozen_before_new_collection": True,
    },
    "budget": {
        "position_model_successors": ledger["position_model"]["total"],
        "position_model_cap": CONFIG["position_successor_cap"],
        "native_steps": ledger["native"]["total"],
        "native_cap": CONFIG["native_step_cap"],
        "extra_calls_from_verification": 0,
    },
    "training": training_checks,
    "paired_evaluation": {
        "episodes": 48,
        "rows": 9600,
        "row_alignment_and_f4_shift": True,
        "same_snapshot_four_query_blocks": True,
        "actual_onsets_only_on_hazardous_landings": True,
    },
    "onset_probability_arrays_reconstructed": onset_array_checks,
    "onset_metrics_and_episode_bootstraps_reconstructed": True,
    "native_records_reconstructed": True,
    "trajectory_arrays": trajectory_checks,
    "trajectory_primary_metrics_and_episode_bootstraps_reconstructed": True,
    "contract": {
        "disabled_baselines_have_no_failure": True,
        "pre_onset_actor_nominal_and_position_proposals_match_baseline": True,
        "fatal_onset_movement_preserved": True,
        "failed_mode_absorbing": True,
        "no_hazard_support_mask": True,
        "no_joint_action_lipschitz_claim": True,
    },
}
write_json("verification.json", result)
print(json.dumps({"status": "pass", "position_successors": 2151424, "native_steps": 14400}))
