"""Read-only recorded-outcome ranking utilities; no model fitting or bank changes."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def reconstruct_death(obs, bits, died, lengths):
    """First fatal landing under the PRE-action mask, verified for all episodes."""
    cells = np.floor(obs[:, 1:, :2]).astype(int)
    hit = np.zeros(cells.shape[:2], bool)
    for j, cell in enumerate([(3, 3), (4, 3), (5, 3)]):
        hit |= np.all(cells == cell, axis=-1) & bits[:, :-1, j].astype(bool)
    hit &= np.arange(hit.shape[1])[None, :] < lengths[:, None]-1
    death = np.where(hit.any(1), hit.argmax(1)+1, -1)
    if not np.array_equal(death >= 1, np.asarray(died, bool)):
        raise ValueError('fatal landing reconstruction disagrees with episode death audit')
    for ep, length in enumerate(lengths):
        state = obs[ep, :length, :8]
        if not np.array_equal(state[1:, 2:], state[:-1, :6]):
            raise ValueError('F4 shift mismatch')
        if not np.array_equal(state[0].reshape(4, 2), np.tile(state[0, :2], (4, 1))):
            raise ValueError('reset stack is not repeated alive XY')
        if death[ep] >= 1:
            d = death[ep]
            if not np.all(state[d:, :2] == state[d, :2]):
                raise ValueError('nonabsorbing post-death trajectory')
            if d+3 < length and not np.all(state[d+3:] == np.tile(state[d, :2], 4)):
                raise ValueError('age 3 does not have established history')
    return death


def outcome_labels(episodes, observation_rows, death_rows):
    death = death_rows[episodes]
    dead = (death >= 1) & (observation_rows >= death)
    return dead, np.where(dead, observation_rows-death, -1)


def nearest_scores(states, references, mean, std):
    """Call the existing JAX scoring implementation, preserving bank order and dtype."""
    import jax
    import jax.numpy as jnp
    from ett.failure_objectives import nearest_reference
    evaluate = jax.jit(nearest_reference)
    parts = []
    for start in range(0, len(states), 2048):
        value = evaluate(jnp.asarray(states[start:start+2048], dtype=jnp.float32),
                         jnp.asarray(references), jnp.asarray(mean), jnp.asarray(std))
        parts.append({k: np.asarray(v) for k, v in value.items()})
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def binary_metrics(labels, distance):
    labels = np.asarray(labels, bool)
    result = dict(rows=len(labels), positives=int(labels.sum()),
                  prevalence=float(labels.mean()) if len(labels) else None,
                  roc_auc=None, average_precision=None)
    if labels.any() and (~labels).any():
        result.update(roc_auc=float(roc_auc_score(labels, -distance)),
                      average_precision=float(average_precision_score(labels, -distance)))
    return result


def describe(values):
    values = np.asarray(values)
    if not len(values):
        return dict(count=0, mean=None, quantiles=None)
    return dict(count=len(values), mean=float(values.mean()),
                quantiles=np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist())


def match_outcomes(failed, alive, episodes, rows, state, previous, source, goal, config):
    """Score-blind, deterministic greedy matching with disjoint episode pairs.

    Each phase supplies one failed row per episode. An episode can serve only
    once, in either role. Pair-block bootstrap therefore preserves whole episode
    dependencies. Controls may die later; they are alive at the matched outcome.
    """
    rng = np.random.default_rng(config['seed'])
    candidates = np.flatnonzero(alive)
    order = rng.permutation(np.flatnonzero(failed))
    used, pairs, eligible = set(), [], 0
    cells = np.floor(state[:, :2]).astype(int)
    for f in order:
        mask = ((episodes[candidates] != episodes[f]) & (source[candidates] == source[f])
                & (np.abs(rows[candidates]-rows[f]) <= config['time_tolerance'])
                & np.all(cells[candidates] == cells[f], axis=1)
                & np.all(goal[candidates] == goal[f], axis=1))
        choices = candidates[mask]
        delta = np.linalg.norm(state[choices, :2]-state[f, :2], axis=1)
        prior_delta = np.linalg.norm(previous[choices, :2]-previous[f, :2], axis=1)
        valid = (delta <= config['xy_tolerance']) & (prior_delta <= config['previous_xy_tolerance'])
        choices, delta, prior_delta = choices[valid], delta[valid], prior_delta[valid]
        eligible += int(len(choices)>0)
        if int(episodes[f]) in used:
            continue
        available = np.array([int(episodes[c]) not in used for c in choices], bool)
        choices, delta, prior_delta = choices[available], delta[available], prior_delta[available]
        if not len(choices):
            continue
        cost = delta/config['xy_tolerance']+prior_delta/config['previous_xy_tolerance']
        cost += np.abs(rows[choices]-rows[f])/config['time_tolerance']
        # np.argmin resolves matching-cost ties by original episode/row order.
        a = int(choices[np.argmin(cost)])
        pairs.append((int(f), a))
        used.update([int(episodes[f]), int(episodes[a])])
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    info = dict(failed_candidates=int(failed.sum()), alive_candidate_rows=int(alive.sum()),
                failed_with_context_support_before_reuse=eligible, matched_pairs=len(pairs),
                unmatched_failed=int(failed.sum())-len(pairs),
                coverage=float(len(pairs)/failed.sum()) if failed.any() else None,
                disjoint_episode_pairs=True)
    if len(pairs):
        assert len(np.unique(episodes[pairs])) == pairs.size
        info.update(xy_gap=describe(np.linalg.norm(state[pairs[:, 0], :2]-state[pairs[:, 1], :2], axis=1)),
                    previous_xy_gap=describe(np.linalg.norm(previous[pairs[:, 0], :2]-previous[pairs[:, 1], :2], axis=1)),
                    time_gap=describe(np.abs(rows[pairs[:, 0]]-rows[pairs[:, 1]])))
    return pairs, info


def pair_ranking(pairs, distances, reps=1000, seed=271):
    if not len(pairs):
        return dict(pairs=0, accuracy_half_ties=None, strict_win=None, ties=None, interval95=None)
    failed, alive = distances[pairs[:, 0]], distances[pairs[:, 1]]
    values = (failed<alive).astype(float)+.5*(failed==alive)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(reps, len(values)))].mean(1)
    return dict(pairs=len(pairs), accuracy_half_ties=float(values.mean()), strict_win=float(np.mean(failed<alive)),
                ties=float(np.mean(failed==alive)), interval95=np.quantile(means, [.025, .975]).tolist(),
                uncertainty='pair-block percentile bootstrap; each block contains two disjoint episodes; fixed matching/bank')
