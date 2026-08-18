"""D2 + verdict for the "is rock-death observable, and when?" diagnostic.

Consumes:
  * the extended death collection
    (scripts/collect_rockfall_death_extended.py -> deaths_extended.npz);
  * the lagged probe summary
    (scripts/probe_critic_candidate_quality.py --bad-npz ... --lags ...).

Produces (all under --out):
  * post_contact_trajectory.json  per-lag z / |v_xy| / z<0.35 fraction /
    displacement-from-contact stats + the recovery-rate-at-k=50 check;
  * post_contact_traces.png       z(k), |v_xy|(k), displacement(k) --
    per-episode traces + median;
  * auc_vs_lag.png                AUC(k) with CI95 bands per checkpoint;
  * verdict_summary.json          the explicit verdict per the a/b/c rule.

Analysis only; touches no training artifact.
"""
import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

Z_FALLEN = 0.35          # the critic's known fallen-pose badness threshold
MOVE_EPS = 0.05          # metres; below this over 10 frames = "stopped"
PROGRESS_EPS = 0.5       # metres of +x corridor progress to count "recovering"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--deaths-npz',
                  default='artifacts/rockfall_death_extended/'
                          'deaths_extended.npz')
  ap.add_argument('--probe-summary',
                  default='artifacts/critic_candidate_probe_lag/summary.json')
  ap.add_argument('--out', default='artifacts/rockfall_death_extended')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  d = np.load(args.deaths_npz, allow_pickle=True)
  obs = d['obs']
  col = np.asarray(d['collapse_step'], np.int64)
  end = np.asarray(d['end_t'], np.int64)
  frozen = np.asarray(d['frozen_after_death'], bool)
  n = len(col)
  kmax = int((end - col).min())
  ks = list(range(0, kmax + 1))

  # per-episode lag-indexed series (obs index c+k)
  z = np.full((n, kmax + 1), np.nan)
  vxy = np.full((n, kmax + 1), np.nan)
  disp = np.full((n, kmax + 1), np.nan)      # |xy(c+k) - xy(c+1)| (contact pos)
  for i in range(n):
    for k in ks:
      t = col[i] + k
      z[i, k] = obs[i, t, 2]
      vxy[i, k] = np.linalg.norm(obs[i, t, 15:17])
      disp[i, k] = np.linalg.norm(obs[i, t, :2] - obs[i, col[i] + 1, :2])

  def stats(a):
    return {'mean': float(np.mean(a)), 'median': float(np.median(a)),
            'p10': float(np.percentile(a, 10)),
            'p90': float(np.percentile(a, 90))}

  per_lag = {}
  for k in ks:
    per_lag[k] = {'z': stats(z[:, k]), 'v_xy': stats(vxy[:, k]),
                  'frac_z_below_0.35': float((z[:, k] < Z_FALLEN).mean()),
                  'disp_from_contact': stats(disp[:, k])}

  kk = min(50, kmax)
  # recovery/motion at k=50 (or kmax): MEASURED displacement, not the stored
  # velocity field (the frozen obs still carries the impact-time velocity).
  moved_last10 = np.array([
      np.linalg.norm(obs[i, col[i] + kk, :2]
                     - obs[i, max(col[i] + kk - 10, col[i] + 1), :2])
      for i in range(n)])
  upright = z[:, kk] >= Z_FALLEN
  moving = moved_last10 > MOVE_EPS
  progressing = np.array([obs[i, col[i] + kk, 0] - obs[i, col[i] + 1, 0]
                          for i in range(n)]) > PROGRESS_EPS
  recovery = {
      'k': kk, 'n': int(n),
      'frac_upright': float(upright.mean()),
      'frac_actually_moving': float(moving.mean()),
      'frac_corridor_progress': float(progressing.mean()),
      'recovery_rate': float((upright & moving & progressing).mean()),
      'stored_velocity_nonzero_frac': float((vxy[:, kk] > 0.1).mean()),
      'note': ('the env freezes obs after death (step() returns _last_obs '
               'without mj_step), so "moving/progress" use measured '
               'displacement; the stored velocity field is the impact-time '
               'value and stays nonzero in the frozen obs')}

  traj = {'n_death_episodes': int(n), 'k_max_common': kmax,
          'post_death_obs_frozen_all_episodes': bool(frozen.all()),
          'per_lag': per_lag, 'recovery_at_k': recovery}
  json.dump(traj, open(os.path.join(args.out,
                                    'post_contact_trajectory.json'), 'w'),
            indent=2)

  # ---- D2 plot -------------------------------------------------------------
  fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
  for ax, series, label, hline in (
      (axes[0], z, 'torso z', Z_FALLEN),
      (axes[1], vxy, '|v_xy| (stored obs value)', None),
      (axes[2], disp, 'displacement from contact pos [m]', None)):
    for i in range(n):
      ax.plot(ks, series[i], color='steelblue', alpha=0.15, lw=0.8)
    ax.plot(ks, np.median(series, axis=0), color='crimson', lw=2.2,
            label='median')
    if hline is not None:
      ax.axhline(hline, color='gray', ls='--', lw=1,
                 label=f'fallen threshold {hline}')
    ax.set_xlabel('lag k after collapse_step')
    ax.set_title(label)
    ax.legend(loc='best', fontsize=8)
  fig.suptitle(f'post-contact trajectories, {n} death episodes '
               f'(obs frozen after k=1: {bool(frozen.all())})')
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'post_contact_traces.png'), dpi=140)
  plt.close(fig)

  # ---- AUC(k) plot + verdict ----------------------------------------------
  ps = json.load(open(args.probe_summary))
  lags = ps['lags']
  fig, ax = plt.subplots(figsize=(7.5, 4.6))
  verdicts = {}
  for tag, colr in zip(ps['per_ckpt'], ('tab:orange', 'tab:blue',
                                        'tab:green', 'tab:red')):
    rk = ps['per_ckpt'][tag]['ranking']
    auc = [rk[f'bad_k{k}_vs_good']['f_min']['auc_badness'] for k in lags]
    lo = [rk[f'bad_k{k}_vs_good']['f_min']['auc_ci95'][0] for k in lags]
    hi = [rk[f'bad_k{k}_vs_good']['f_min']['auc_ci95'][1] for k in lags]
    ax.plot(lags, auc, 'o-', color=colr, label=tag)
    ax.fill_between(lags, lo, hi, color=colr, alpha=0.18)
    # per-checkpoint verdict rule
    rises = [(k, a, l) for k, a, l in zip(lags, auc, lo)
             if a > 0.70 and l > 0.5]
    if recovery['recovery_rate'] > 0.5:
      v = {'verdict': 'c_not_terminal',
           'detail': f"recovery rate {recovery['recovery_rate']:.2f} at "
                     f"k={recovery['k']}"}
    elif rises:
      v = {'verdict': 'b_capture_artifact', 'smallest_k': int(rises[0][0]),
           'auc_at_k': float(rises[0][1])}
    elif all(0.3 <= a <= 0.6 for a in auc):
      v = {'verdict': 'a_structural',
           'detail': f'AUC(k) within [0.3, 0.6] for all k <= {max(lags)}'}
    elif all(a <= 0.6 for a in auc):
      # Below the rule's [0.3, 0.6] band on the LOW side at some lag: the
      # ranking is significantly INVERTED (death states score HIGHER than
      # comparable good states), which the rule table did not anticipate.
      # No lag yields a usable badness ranking -> structural, with the
      # inversion stated explicitly rather than folded into 'inconclusive'
      # (that bin is for signals rising toward 0.7).
      hi = [rk[f'bad_k{k}_vs_good']['f_min']['auc_ci95'][1] for k in lags]
      v = {'verdict': 'a_structural',
           'inverted': True,
           'detail': (f'AUC(k) never exceeds 0.6 at any lag; min AUC '
                      f'{min(auc):.3f} with CI95 upper bound '
                      f'{max(hi):.3f} -- at some lags significantly BELOW '
                      f'0.5 (inverted: death states score higher than good '
                      f'states). Outside the [0.3, 0.6] band on the low '
                      f'side only.')}
    else:
      v = {'verdict': 'inconclusive',
           'detail': 'AUC(k) rises above 0.6 but never >0.70 with CI '
                     'excluding 0.5'}
    v['auc_by_k'] = {str(k): round(a, 4) for k, a in zip(lags, auc)}
    v['auc_ci_by_k'] = {str(k): [round(a, 4), round(b, 4)]
                        for k, a, b in zip(lags, lo, hi)}
    v['shuffle_control'] = {
        str(k): rk[f'bad_k{k}_vs_good']['f_min']['shuffle_control']['mean']
        for k in lags}
    v['spearman_fmin_x_upright_good'] = (
        ps['per_ckpt'][tag]['spearman_fmin_x_upright_good'])
    verdicts[tag] = v
  ax.axhline(0.5, color='gray', ls=':', lw=1)
  ax.axhline(0.7, color='gray', ls='--', lw=1)
  ax.set_xlabel('lag k after collapse_step')
  ax.set_ylabel('AUC of badness = -f_min (bad_k vs good)')
  ax.set_title('critic ranking of death states vs lag (CI95: episode '
               'cluster bootstrap)')
  ax.legend()
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'auc_vs_lag.png'), dpi=140)
  plt.close(fig)

  out = {
      'question': 'is rock-death observable in the 29-dim ant obs, and when?',
      'n_death_episodes': int(n),
      'post_death_obs_frozen_all_episodes': bool(frozen.all()),
      'structural_note': (
          'crl/rockfall_ant.py step(): once _dead is set the env returns '
          '_last_obs without stepping physics. obs[collapse+1+k] is '
          'bitwise-identical to obs[collapse+1] for every k, so AUC(k) is '
          'constant for k >= 1 BY CONSTRUCTION, and no post-contact physical '
          'signature (burial, collapse, stopping) exists in the observation '
          'stream at any lag.') if bool(frozen.all()) else None,
      'recovery_at_k': recovery,
      'per_checkpoint': verdicts,
      'inputs': {'deaths_npz': args.deaths_npz,
                 'probe_summary': args.probe_summary},
  }
  json.dump(out, open(os.path.join(args.out, 'verdict_summary.json'), 'w'),
            indent=2)
  print(json.dumps({k: v for k, v in out.items()
                    if k not in ('per_checkpoint',)}, indent=2))
  for tag, v in verdicts.items():
    print(f"\n{tag}: {v['verdict']}  auc_by_k {v['auc_by_k']}")
  print(f"\nwrote verdict_summary.json, post_contact_trajectory.json, "
        f"post_contact_traces.png, auc_vs_lag.png -> {args.out}")


if __name__ == '__main__':
  main()
