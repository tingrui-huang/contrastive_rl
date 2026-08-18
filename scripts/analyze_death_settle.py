"""Settling-horizon calibration + Gate-4 physics analysis + probe-input build
for the rock-death observability patch (Stages C/D/E support).

Consumes settle_traces.npz (scripts/death_settle_sweep.py). Produces, under
the same artifact dir:

  settle_sweep.json         per-horizon physics stats + the chosen N with the
                            data-driven justification
  settle_traces.png         z / up_z / |v_xy| / displacement vs settle substep
  settled_vs_healthy.png    (z, |v_xy|) scatter: legacy frozen death states,
                            settled death states at the chosen N, healthy
                            in-zone states (probe good group)
  settled_deaths_probe.npz  probe-local bad-state input for
                            probe_critic_candidate_quality.py --bad-npz
                            (lag 0 = pre-hit state, lag 1 = settled state)
  patch_summary.json        machine-readable roll-up: gates, chosen N,
                            physics stats, probe results (when present)

Selection rule (physics only, no RL in the loop): the chosen N is the
smallest sweep horizon at which the ant is physically stopped
(median |v_xy| < 0.05 m/s) and the pose is approximately stable
(median |z(2N) - z(N)| < 0.02 m). Purely an environment-physics calibration.
"""
import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SWEEP_NS = (5, 10, 20, 40, 80, 160)
V_STOP = 0.05
DZ_STABLE = 0.02
Z_FALLEN = 0.35


def up_z(state):
  """Torso world-up z from the obs quaternion (state[3:7] = w,x,y,z)."""
  x, y = state[..., 4], state[..., 5]
  return 1.0 - 2.0 * (x * x + y * y)


def stats(a):
  return {'mean': float(np.mean(a)), 'median': float(np.median(a)),
          'p10': float(np.percentile(a, 10)),
          'p90': float(np.percentile(a, 90))}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dir', default='artifacts/rockfall_death_physics_patch')
  ap.add_argument('--good-npz',
                  default='artifacts/critic_candidate_probe_lag/'
                          'per_state_scores.npz',
                  help='probe per-state npz supplying the healthy in-zone '
                       'comparison group')
  args = ap.parse_args()

  d = np.load(os.path.join(args.dir, 'settle_traces.npz'), allow_pickle=True)
  tr = d['traces']                       # [n, N_max, 29]
  n, n_max = tr.shape[0], tr.shape[1]
  ns = [N for N in SWEEP_NS if N <= n_max]
  legacy = d['legacy_frozen_state']      # [n, 29]
  z_t = tr[:, :, 2]
  v_t = np.linalg.norm(tr[:, :, 15:17], axis=2)
  u_t = up_z(tr)
  disp_t = np.linalg.norm(tr[:, :, :2] - tr[:, 0:1, :2], axis=2)

  # ---- per-horizon stats + stability ---------------------------------------
  per_n = {}
  for N in ns:
    k = N - 1
    row = {'z': stats(z_t[:, k]), 'up_z': stats(u_t[:, k]),
           'v_xy': stats(v_t[:, k]), 'disp_from_contact': stats(disp_t[:, k]),
           'frac_z_below_0.35': float((z_t[:, k] < Z_FALLEN).mean()),
           'frac_stopped_v_below_0.05': float((v_t[:, k] < V_STOP).mean())}
    if 2 * N <= n_max:
      dz = np.abs(z_t[:, 2 * N - 1] - z_t[:, k])
      dxy = np.linalg.norm(tr[:, 2 * N - 1, :2] - tr[:, k, :2], axis=1)
      row['stability_to_2N'] = {'abs_dz_median': float(np.median(dz)),
                                'abs_dxy_median': float(np.median(dxy))}
    per_n[N] = row

  chosen = None
  for N in ns:
    r = per_n[N]
    stable = ('stability_to_2N' in r
              and r['stability_to_2N']['abs_dz_median'] < DZ_STABLE)
    if r['v_xy']['median'] < V_STOP and stable:
      chosen = N
      break
  if chosen is None:
    # fall back to the largest horizon if motion has stopped there; else
    # report failure to settle (do NOT manufacture a signature).
    if per_n[ns[-1]]['v_xy']['median'] < V_STOP:
      chosen = ns[-1]
  kc = chosen - 1 if chosen else None

  sweep = {
      'n_death_episodes': int(n), 'n_max': int(n_max), 'sweep_ns': ns,
      'selection_rule': (f'smallest N with median |v_xy| < {V_STOP} AND '
                         f'median |z(2N)-z(N)| < {DZ_STABLE}'),
      'chosen_n': chosen,
      'per_n': per_n,
      'legacy_frozen': {'z': stats(legacy[:, 2]),
                        'v_xy': stats(np.linalg.norm(legacy[:, 15:17],
                                                     axis=1))},
  }
  json.dump(sweep, open(os.path.join(args.dir, 'settle_sweep.json'), 'w'),
            indent=2)
  print(json.dumps({k: sweep[k] for k in
                    ('chosen_n', 'selection_rule', 'legacy_frozen')},
                   indent=2))
  for N in ns:
    r = per_n[N]
    st = r.get('stability_to_2N', {})
    print(f"N={N:4d}: z med {r['z']['median']:.3f}  up_z med "
          f"{r['up_z']['median']:+.3f}  |v| med {r['v_xy']['median']:.3f}  "
          f"disp med {r['disp_from_contact']['median']:.3f}  "
          f"z<0.35 {r['frac_z_below_0.35']:.2f}  "
          f"dz->2N {st.get('abs_dz_median', float('nan')):.4f}")

  # ---- healthy comparison group (Gate 4) -----------------------------------
  g = np.load(args.good_npz, allow_pickle=True)
  good = g['group'] == 'good'
  gz = g['states'][good, 2]
  gv = np.linalg.norm(g['states'][good, 15:17], axis=1)
  gu = up_z(g['states'][good])
  gate4 = None
  if chosen:
    sz, sv, su = z_t[:, kc], v_t[:, kc], u_t[:, kc]
    gate4 = {
        'chosen_n': chosen,
        'settled': {'z': stats(sz), 'v_xy': stats(sv), 'up_z': stats(su),
                    'disp_from_contact': stats(disp_t[:, kc])},
        'healthy_in_zone': {'n': int(good.sum()), 'z': stats(gz),
                            'v_xy': stats(gv), 'up_z': stats(gu)},
        # overlap of the dominant emergent signature (stopped + sunk):
        'frac_healthy_with_v_below_settled_p90':
            float((gv <= np.percentile(sv, 90)).mean()),
        'frac_healthy_with_z_below_settled_p90':
            float((gz <= np.percentile(sz, 90)).mean()),
        'frac_settled_inside_healthy_v_p10':
            float((sv >= np.percentile(gv, 10)).mean()),
    }
    json.dump(gate4, open(os.path.join(args.dir, 'gate4_observable.json'),
                          'w'), indent=2)
    print('\nGate 4:', json.dumps(gate4['settled'], indent=1))

  # ---- plots ---------------------------------------------------------------
  ks = np.arange(1, n_max + 1)
  fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
  for ax, series, label in ((axes[0], z_t, 'torso z'),
                            (axes[1], u_t, 'torso up_z (orientation)'),
                            (axes[2], v_t, '|v_xy| [m/s]'),
                            (axes[3], disp_t, 'displacement since contact '
                                              '[m]')):
    for i in range(n):
      ax.plot(ks, series[i], color='steelblue', alpha=0.12, lw=0.8)
    ax.plot(ks, np.median(series, axis=0), color='crimson', lw=2.2,
            label='median')
    for N in ns:
      ax.axvline(N, color='gray', lw=0.6, ls=':')
    if chosen:
      ax.axvline(chosen, color='green', lw=1.6, ls='--',
                 label=f'chosen N={chosen}')
    ax.set_xscale('log')
    ax.set_xlabel('settle substep (dt=0.02 s)')
    ax.set_title(label)
    ax.legend(fontsize=8)
  fig.suptitle(f'death-settle physics, {n} fatal episodes '
               '(actor ctrl zeroed at fatal contact)')
  fig.tight_layout()
  fig.savefig(os.path.join(args.dir, 'settle_traces.png'), dpi=140)
  plt.close(fig)

  if chosen:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(gv, gz, s=4, alpha=0.15, color='tab:gray',
               label=f'healthy in-zone good states (n={good.sum()})')
    lv = np.linalg.norm(legacy[:, 15:17], axis=1)
    ax.scatter(lv, legacy[:, 2], s=36, color='tab:orange', marker='^',
               label='legacy frozen death obs (old env)')
    ax.scatter(v_t[:, kc], z_t[:, kc], s=36, color='tab:red', marker='x',
               label=f'settled death obs (patched, N={chosen})')
    ax.axhline(Z_FALLEN, color='gray', ls='--', lw=1,
               label='fallen threshold z=0.35')
    ax.set_xlabel('|v_xy| [m/s]')
    ax.set_ylabel('torso z')
    ax.set_title('fatal states before/after the physics patch vs healthy '
                 'states')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.dir, 'settled_vs_healthy.png'), dpi=140)
    plt.close(fig)

    # ---- probe-local bad npz at the chosen N -------------------------------
    col = np.asarray(d['collapse_step'], np.int64)
    L = int(col.max()) + 2
    obs = np.zeros((n, L, 58), np.float32)
    for i in range(n):
      obs[i, col[i], :29] = d['prehit_state'][i]     # lag 0: pre-hit state
      obs[i, col[i] + 1] = d['settled_obs58'][i]     # lag 1: settled state
    p = os.path.join(args.dir, 'settled_deaths_probe.npz')
    with open(p + '.tmp', 'wb') as f:
      np.savez_compressed(
          f, obs=obs, collapse_step=col, end_t=col + 1,
          goal_xy=d['goal_xy'], source=d['source'],
          episode_id=d['episode_id'], env_seed=d['env_seed'],
          meta=json.dumps({'chosen_n': chosen,
                           'note': 'probe-local settled fatal states; NOT '
                                   'the production failure bank'}))
    os.replace(p + '.tmp', p)
    print(f'probe input -> {p}')

  # ---- roll-up summary (includes gates + probe results when present) ------
  summary = {'stage': 'rockfall death-settle physics patch',
             'sweep': {'chosen_n': chosen,
                       'selection_rule': sweep['selection_rule']},
             'gate4': gate4}
  for name, key in (('sweep_gates.json', 'gates_1_5_3'),
                    ('gate2_nondeath.json', 'gate2'),
                    ('probe_settled/summary.json', 'probe_settled')):
    fp = os.path.join(args.dir, name)
    if os.path.exists(fp):
      summary[key] = json.load(open(fp))
  json.dump(summary, open(os.path.join(args.dir, 'patch_summary.json'), 'w'),
            indent=2)
  print(f"patch_summary.json written (chosen N = {chosen})")
  if chosen is None:
    print('WARNING: no sweep horizon met the settling criteria -- the fatal '
          'state does not stabilize within the tested range. Do not proceed; '
          'report this result.')


if __name__ == '__main__':
  main()
