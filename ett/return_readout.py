"""Small readouts and logged-return metrics; frozen contrastive representations."""
import numpy as np
from scipy.stats import spearmanr


def return_windows(observations, actions, lengths, horizon=20, discount=.95):
    from ett.rollout_return import task_reward, validate_fixed_goal
    validate_fixed_goal(observations[..., 8:])
    rewards = np.asarray(task_reward(observations[..., :8], observations[..., 8:]), np.float64)
    episodes, timesteps, targets = [], [], []
    for e, length in enumerate(lengths):
        # A length-L observation trajectory has L-1 real actions and futures.
        for t in range(max(0, int(length)-horizon)):
            episodes.append(e); timesteps.append(t)
            targets.append(float(rewards[e, t+1:t+horizon+1] @ (discount**np.arange(horizon))))
    ep, t = np.asarray(episodes, int), np.asarray(timesteps, int)
    if not np.isfinite(actions[ep, t]).all():
        raise ValueError('nonfinite recorded action')
    return ep, t, np.asarray(targets), rewards


def episode_split(n, bank_episodes):
    permutation = np.random.default_rng(0).permutation(n)
    test = np.sort(permutation[:round(.1*n)])
    pool = np.sort(permutation[round(.1*n):])
    perm = np.random.default_rng(9201).permutation(pool)
    count = round(.1*len(pool))
    return dict(train=np.sort(perm[count:]), selection=np.setdiff1d(perm[:count], bank_episodes),
                test=np.setdiff1d(test, bank_episodes))


def standardize_fit(x):
    x = np.asarray(x, np.float64)
    mean, std = x.mean(0), x.std(0)
    std = np.where(std<1e-8, 1., std)
    return mean, std


def ridge_fit(x, target, selection_x, selection_y, grid):
    mean, std = standardize_fit(x)
    z = (x-mean)/std; v = (selection_x-mean)/std
    intercept = float(target.mean())
    gram, rhs = z.T@z/len(z), z.T@(target-intercept)/len(z)
    trials, coefficients = [], []
    for penalty in grid:
        coef = np.linalg.solve(gram+penalty*np.eye(z.shape[1]), rhs)
        prediction = v@coef+intercept
        trials.append(dict(penalty=penalty, selection_mse=float(np.mean((prediction-selection_y)**2))))
        coefficients.append(coef)
    selected = int(np.argmin([v['selection_mse'] for v in trials]))
    return dict(mean=mean, std=std, coef=coefficients[selected], intercept=intercept), dict(
        trials=trials, selected_penalty=grid[selected], objective='mean squared error + penalty*||standardized-feature coefficients||^2; unpenalized intercept')


def ridge_predict(model, x):
    return (x-model['mean'])/model['std'] @ model['coef']+model['intercept']


def regression_metrics(target, prediction, calibrated=True):
    result = dict(rows=len(target), mae=None, rmse=None, spearman=None)
    if not len(target):
        return result
    if calibrated:
        result.update(mae=float(np.mean(np.abs(prediction-target))), rmse=float(np.sqrt(np.mean((prediction-target)**2))))
    if np.ptp(target)>0 and np.ptp(prediction)>0:
        result['spearman'] = float(spearmanr(target, prediction).statistic)
    return result


def make_pairs(ep, t, state, goal, source, seed=95121, matched=False):
    """One anchor window per episode; pair distinct episodes without reuse.

    Matched controls may use any window of an unused episode. No returns,
    predictions, hidden fields or recorded actions determine pairing.
    """
    rng = np.random.default_rng(seed)
    episodes = rng.permutation(np.unique(ep))
    anchors = {int(e): int(rng.choice(np.flatnonzero(ep==e))) for e in episodes}
    used, pairs = set(), []
    distance = np.linalg.norm(state[:, :2]-goal[:, :2], axis=1)
    for e in episodes:
        if e in used:
            continue
        a = anchors[int(e)]
        if not matched:
            pool = [int(e2) for e2 in episodes if e2 not in used and e2!=e]
            if not pool:
                break
            b = anchors[pool[0]]
        else:
            eligible = (~np.isin(ep, list(used)+[e])) & (source==source[a]) & (np.abs(t-t[a])<=3)
            eligible &= np.all(np.floor(state[:, :2])==np.floor(state[a, :2]), axis=1)
            eligible &= np.all(goal==goal[a], axis=1) & (np.abs(distance-distance[a])<=.25)
            delta = np.linalg.norm(state[:, :2]-state[a, :2], axis=1)
            eligible &= delta<=.25
            candidates = np.flatnonzero(eligible)
            if not len(candidates):
                continue
            cost = delta[candidates]/.25+np.abs(distance[candidates]-distance[a])/.25+np.abs(t[candidates]-t[a])/3
            b = int(candidates[np.argmin(cost)])
        pairs.append((a,b)); used.update([int(ep[a]), int(ep[b])])
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    assert len(np.unique(ep[pairs]))==pairs.size
    return pairs


def pair_values(target, prediction, pairs):
    a,b = pairs.T
    delta = target[a]-target[b]
    valid = np.abs(delta)>1e-10
    change = prediction[a]-prediction[b]
    # Prediction ties get half credit; equal-return pairs are excluded explicitly.
    values = (delta[valid]*change[valid]>0).astype(float)+.5*(change[valid]==0)
    return values, valid


def pair_metrics(target, prediction, pairs, replicates=1000):
    values, valid = pair_values(target, prediction, pairs)
    ci = None
    if len(values):
        rng = np.random.default_rng(95122)
        means = values[rng.integers(len(values), size=(replicates, len(values)))].mean(1)
        ci = np.quantile(means, [.025,.975]).tolist()
    return dict(pairs=len(pairs), unequal_return_pairs=int(valid.sum()), equal_return_pairs=int((~valid).sum()),
                accuracy=float(values.mean()) if len(values) else None, interval95=ci,
                uncertainty='resample disjoint two-episode blocks conditional on unequal-return subset and fixed pairing')


def error_intervals(target, prediction, ep, replicates=500):
    episodes = np.unique(ep)
    absolute = np.array([np.mean(np.abs(target[ep==e]-prediction[ep==e])) for e in episodes])
    square = np.array([np.mean((target[ep==e]-prediction[ep==e])**2) for e in episodes])
    rng = np.random.default_rng(95123)
    draw = rng.integers(len(episodes), size=(replicates, len(episodes)))
    return dict(mae95=np.quantile(absolute[draw].mean(1), [.025,.975]).tolist(),
                rmse95=np.quantile(np.sqrt(square[draw].mean(1)), [.025,.975]).tolist(),
                weighting='complete episodes; each has 31 eligible windows in this dataset')
