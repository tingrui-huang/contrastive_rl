"""Read-only critic readout for round 2: which first torque does each critic prefer at the logged reset states?

States: the anchors of the reset row (t = 0; 192 logged-shortcut + 4 logged-detour
episodes) and, secondarily, every anchor whose state lies in the start region.
Candidates at each state: the logged torque of that row, the 4 teacher detour
and 2 teacher shortcut reset torques used by the candidates analysis (same draw),
and the mode torque of the start agent, the round-1 CF agent of the lineage and
the round-2 O / CF / CFold agents of the lineage.  Critics: the start agent's,
the round-1 O / CF critics and the round-2 O / CF / CFold critics of the same
lineage.  Goal = the anchor episode's task goal (as in the fork / candidates
analyses).  Reports, per critic, f(candidate) - f(logged torque) with P(> 0)
over the states, the share of states whose critic top-1 is a teacher detour
torque, and the mode-vs-start-mode score gaps.  No rollouts, no training.

  python scripts/diag_v6_round2_critic_readout.py --out <json>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import exp_v6_mainline_pilot as MP  # noqa: E402
import diag_v6_pilot_trajectories as DT  # noqa: E402


def bundle(ckpt):
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  cfg = MP.recipe_config(0, MP.R2 / '_cfg')
  MP.fill_dims(cfg)
  nets = MP.make_nets(cfg)
  _, st = checkpoint.load_checkpoint(ckpt)
  pp, qp = st.policy_params, st.q_params

  @jax.jit
  def mode(o):
    return jnp.tanh(nets.policy_network.apply(pp, o).loc)

  @jax.jit
  def f(o, a):
    phi, psi = nets.representation_network.apply(qp, o, a)
    return jnp.min(jnp.sum(phi * psi, axis=1), axis=1)
  return mode, f


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', required=True)
  args = ap.parse_args()
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  with np.load(MP.SIDECAR, allow_pickle=True) as sc:
    route = sc['route_realized'].astype(str)
  logged_route = route[anchors.episode]
  reg = np.array([DT.region(float(p[0]), float(p[1])) for p in anchors.state[:, :2]])
  teach = DT._teacher_reset_torques()
  state_sets = {'reset rows, logged shortcut': (anchors.t == 0) & (logged_route == 'shortcut'),
                'reset rows, logged detour': (anchors.t == 0) & (logged_route == 'detour'),
                'start region, logged shortcut': (reg == 'start') & (logged_route == 'shortcut'),
                'start region, logged detour': (reg == 'start') & (logged_route == 'detour')}
  ckpts = {'start': MP.START_CKPT}
  for s in MP.SEEDS:
    ckpts[f'O1_s{s}'] = MP.run_dir('O', s) / 'final.pkl'
    ckpts[f'CF1_s{s}'] = MP.run_dir('CF', s) / 'final.pkl'
    for arm in MP.R2_ARMS:
      ckpts[f'{arm}2_s{s}'] = MP.run_dir(arm, s, MP.R2) / 'final.pkl'
  B = {n: bundle(p) for n, p in ckpts.items() if p.exists()}
  print('loaded', sorted(B), flush=True)
  res = {'teacher_torques': {k: v.tolist() for k, v in teach.items()}, 'sets': {}}
  for set_name, m in state_sets.items():
    idx = np.flatnonzero(m)
    if len(idx) == 0:
      continue
    obs = anchors.obs31[idx].astype(np.float32)
    cands = {'logged': anchors.action[idx].astype(np.float32)}
    for k, v in teach.items():
      cands[k] = np.repeat(v[None], len(idx), axis=0)
    for n in ('start',) + tuple(f'CF1_s{s}' for s in MP.SEEDS) + tuple(f'{arm}2_s{s}' for arm in MP.R2_ARMS for s in MP.SEEDS):
      if n in B:
        cands[f'mode_{n}'] = np.asarray(B[n][0](obs))
    names = list(cands)
    out = {'n_states': int(len(idx)), 'weight': float(anchors.weight[idx].sum()), 'critics': {}}
    for cn, (_, f) in B.items():
      F = np.stack([np.asarray(f(obs, cands[c])) for c in names], axis=1)    # [n, n_cand]
      Fl = F[:, names.index('logged')]
      det_cols = [names.index(k) for k in teach if k.startswith('teacher_detour')]
      sc_cols = [names.index(k) for k in teach if k.startswith('teacher_shortcut')]
      best_det = F[:, det_cols].max(axis=1); best_sc = F[:, sc_cols].max(axis=1)
      top1 = F.argmax(axis=1)
      rec = {'candidates': {c: {'mean_f': float(F[:, j].mean()), 'gap_vs_logged_mean': float((F[:, j] - Fl).mean()),
                                'P_above_logged': float((F[:, j] > Fl).mean())} for j, c in enumerate(names)},
             'best_teacher_detour_minus_logged': {'mean': float((best_det - Fl).mean()), 'P_above': float((best_det > Fl).mean())},
             'best_teacher_detour_minus_best_teacher_shortcut': {'mean': float((best_det - best_sc).mean()), 'P_above': float((best_det > best_sc).mean())},
             'top1_family_share': {'teacher_detour': float(np.isin(top1, det_cols).mean()), 'teacher_shortcut': float(np.isin(top1, sc_cols).mean()),
                                   'logged': float((top1 == names.index('logged')).mean()),
                                   'agent_modes': float(np.isin(top1, [names.index(c) for c in names if c.startswith('mode_')]).mean())}}
      for c in names:
        if c.startswith('mode_') and c != 'mode_start':
          j, j0 = names.index(c), names.index('mode_start')
          rec['candidates'][c]['gap_vs_mode_start'] = float((F[:, j] - F[:, j0]).mean())
          rec['candidates'][c]['P_above_mode_start'] = float((F[:, j] > F[:, j0]).mean())
      out['critics'][cn] = rec
    res['sets'][set_name] = out
    print(f'== {set_name}: n {len(idx)}', flush=True)
    for cn, rec in out['critics'].items():
      d = rec['best_teacher_detour_minus_logged']; e = rec['best_teacher_detour_minus_best_teacher_shortcut']; t = rec['top1_family_share']
      print(f'  {cn:10s} best_detour - logged {d["mean"]:+.3f} (P {d["P_above"]:.2f}) | best_detour - best_shortcut {e["mean"]:+.3f} (P {e["P_above"]:.2f}) | '
            f'top1: detour {t["teacher_detour"]:.2f} shortcut {t["teacher_shortcut"]:.2f} logged {t["logged"]:.2f} modes {t["agent_modes"]:.2f}', flush=True)
  MP.write_json(Path(args.out), res)


if __name__ == '__main__':
  main()
