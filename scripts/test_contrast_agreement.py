"""Independent saved-array checks for the contrast-agreement headline values."""

from pathlib import Path
import json

import numpy as np
from scipy.stats import spearmanr


CONTINUATION = Path("artifacts/pointmaze_region_pilot/continuation_precision_s01_v1")
RESPONSE = Path("artifacts/pointmaze_region_pilot/response_search_s01_v1")
RESULTS = Path("artifacts/pointmaze_region_pilot/contrast_agreement_v1/results.json")
PAIRS = ("s0_k2", "s1_k0")


def _weighted_midrank(values, weights):
    order = np.argsort(values, kind="stable")
    answer = np.empty(len(values), np.float64)
    cumulative = 0.0
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        group = float(weights[order[start:end]].sum())
        answer[order[start:end]] = cumulative + group / 2
        cumulative += group
        start = end
    return answer


def _weighted_spearman(x, y, weights):
    rx = _weighted_midrank(x, weights)
    ry = _weighted_midrank(y, weights)
    mx = np.average(rx, weights=weights)
    my = np.average(ry, weights=weights)
    numerator = np.sum(weights * (rx - mx) * (ry - my))
    denominator = np.sqrt(
        np.sum(weights * (rx - mx) ** 2) * np.sum(weights * (ry - my) ** 2)
    )
    return float(numerator / denominator)


def _weighted_kendall_brute(x, y, weights):
    numerator = 0.0
    comparable_x = 0.0
    comparable_y = 0.0
    for index in range(len(x) - 1):
        pair_weight = weights[index] * weights[index + 1 :]
        sx = np.sign(x[index] - x[index + 1 :])
        sy = np.sign(y[index] - y[index + 1 :])
        numerator += float(np.sum(pair_weight * sx * sy))
        comparable_x += float(np.sum(pair_weight[sx != 0]))
        comparable_y += float(np.sum(pair_weight[sy != 0]))
    return float(numerator / np.sqrt(comparable_x * comparable_y))


def _continuation_queries(name):
    with np.load(CONTINUATION / name / "queries.npz") as archive:
        query = {key: archive[key] for key in archive.files}
    critic = []
    mc = []
    changed = []
    for batch_number in range(9):
        with np.load(CONTINUATION / name / f"batch{batch_number}.npz") as archive:
            critic.append(
                (archive["candidate_critic"] - archive["reference_critic"]).mean(
                    axis=(1, 2)
                )
            )
            mc.append(
                (archive["candidate_mc"] - archive["reference_mc"]).mean(axis=(1, 2))
            )
            first = archive["candidate_reward"] != archive["reference_reward"]
            later = np.any(
                archive["candidate_continuation_rewards"]
                != archive["reference_continuation_rewards"],
                axis=-1,
            )
            changed.append(np.any(first[:, :, None] | later, axis=(1, 2)))
    return query, np.concatenate(critic), np.concatenate(mc), np.concatenate(changed)


def _load_visitation(path):
    with np.load(path) as archive:
        buckets = sorted({int(key.split("_")[0][1:]) for key in archive.files})
        return [
            {
                key.partition("_")[2]: archive[key]
                for key in archive.files
                if key.startswith(f"b{bucket}_")
            }
            for bucket in buckets
        ]


def _saved_reward_pre_entry(check, matched):
    records = _load_visitation(check / "visitation.npz")
    paths = [(record, i) for record in records for i in range(len(record["root"]))]
    flags = []
    for query_index, path_index in enumerate(matched["query_path"]):
        record, row = paths[int(path_index)]
        time = int(matched["query_t"][query_index])
        np.testing.assert_array_equal(matched["query_state"][query_index], record["states"][row, time])
        flags.append(not bool(record["reward"][row, :time].any()))
    return np.asarray(flags)


def _bootstrap_spearman(x, y, indices):
    values = []
    for draw in indices:
        if np.ptp(x[draw]) == 0 or np.ptp(y[draw]) == 0:
            continue
        values.append(float(spearmanr(x[draw], y[draw]).statistic))
    return np.quantile(values, [0.025, 0.975])


def test_saved_array_headlines():
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    continuation = results["continuation_precision"]["comparisons"]
    for name in PAIRS:
        query, critic, mc, changed = _continuation_queries(name)
        published = continuation[name]["unstratified_reproduction"]["published"]
        np.testing.assert_allclose(np.sum(query["mass"] * critic), published["critic"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(np.sum(query["mass"] * mc), published["mc"], atol=1e-12, rtol=1e-12)
        for label, mask in (("pre_entry", query["pre_entry"]), ("entered", ~query["pre_entry"])):
            saved = continuation[name]["strata"][label]
            x = critic[mask]
            y = mc[mask]
            weight = query["mass"][mask]
            assert saved["n_queries"] == len(x)
            np.testing.assert_allclose(saved["critic_mean"]["estimate"], np.average(x, weights=weight), atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(saved["mc_mean"]["estimate"], np.average(y, weights=weight), atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(saved["sign_agreement"]["estimate"], np.mean(np.sign(x) == np.sign(y)), atol=1e-12, rtol=0)
            np.testing.assert_allclose(saved["weighted_spearman"]["estimate"], _weighted_spearman(x, y, weight), atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(saved["weighted_kendall_tau_b"]["estimate"], _weighted_kendall_brute(x, y, weight), atol=1e-12, rtol=1e-12)
            changed_mask = changed[mask]
            changed_saved = saved["sequence_changed_only"]
            assert changed_saved["n_queries"] == int(changed_mask.sum())
            np.testing.assert_allclose(
                changed_saved["sign_agreement"]["estimate"],
                np.mean(np.sign(x[changed_mask]) == np.sign(y[changed_mask])),
                atol=1e-12,
                rtol=0,
            )

    response_rows = results["response_search"]["comparisons"]
    critic = []
    mc = []
    pre_critic = []
    pre_mc = []
    for saved in response_rows:
        check = RESPONSE / "checks" / saved["name"]
        with np.load(check / "matched_mc.npz") as archive:
            matched = {key: archive[key] for key in archive.files}
        cq = (matched["proposed_critic"] - matched["before_critic"]).mean(axis=(1, 2))
        mq = (matched["proposed_mc"] - matched["before_mc"]).mean(axis=(1, 2))
        np.testing.assert_allclose(saved["critic_delta"]["estimate"], cq.mean(), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(saved["mc_delta"]["estimate"], mq.mean(), atol=1e-12, rtol=1e-12)
        pre = _saved_reward_pre_entry(check, matched)
        assert saved["pre_entry"]["n_queries"] == int(pre.sum())
        assert saved["pre_entry"]["flag_reconstructed"]
        if pre.any():
            assert saved["pre_entry"]["delta_estimable"]
            np.testing.assert_allclose(saved["pre_entry"]["critic_delta"]["estimate"], cq[pre].mean(), atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(saved["pre_entry"]["mc_delta"]["estimate"], mq[pre].mean(), atol=1e-12, rtol=1e-12)
            pre_critic.append(cq[pre].mean())
            pre_mc.append(mq[pre].mean())
        else:
            assert not saved["pre_entry"]["delta_estimable"]
            assert saved["pre_entry"]["critic_delta"] is None
            assert saved["pre_entry"]["mc_delta"] is None
        critic.append(cq.mean())
        mc.append(mq.mean())

    critic = np.asarray(critic)
    mc = np.asarray(mc)
    pre_critic = np.asarray(pre_critic)
    pre_mc = np.asarray(pre_mc)
    aggregate = results["response_search"]["all_comparisons"]
    pre_aggregate = results["response_search"]["pre_entry_estimable_subset"]
    assert aggregate["sign_agreement"]["count"] == int(np.sum(np.sign(critic) == np.sign(mc)))
    assert pre_aggregate["sign_agreement"]["count"] == int(np.sum(np.sign(pre_critic) == np.sign(pre_mc)))
    np.testing.assert_allclose(aggregate["spearman"]["estimate"], spearmanr(critic, mc).statistic, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(pre_aggregate["spearman"]["estimate"], spearmanr(pre_critic, pre_mc).statistic, atol=1e-12, rtol=1e-12)
    generator = np.random.default_rng(190000001)
    draws = generator.integers(18, size=(2000, 18))
    pre_draws = generator.integers(len(pre_critic), size=(2000, len(pre_critic)))
    np.testing.assert_allclose(aggregate["spearman"]["ci95"], _bootstrap_spearman(critic, mc, draws), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(pre_aggregate["spearman"]["ci95"], _bootstrap_spearman(pre_critic, pre_mc, pre_draws), atol=1e-12, rtol=1e-12)
    assert results["audit"]["computed_new_transitions"] == 0


if __name__ == "__main__":
    test_saved_array_headlines()
    print("contrast agreement saved-array headline checks passed")
