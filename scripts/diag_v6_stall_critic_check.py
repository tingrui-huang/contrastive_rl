"""At the states of the CF continuations that stall where O finishes (the 19
pairs): does the CF critic reward the CF actor's own torque above the O /
start torques at the same state?  Also at the O continuation's states.
Frozen models, no rollouts."""
import sys, json
import numpy as np
sys.path.insert(0, 'scripts')
import diag_v6_pilot_trajectories as DT
import exp_v6_mainline_pilot as MP

P = MP.read_json(DT.OUT / 'pairs.json')
Z = np.load(DT.OUT / 'pairs_traj.npz')
out = {}
for kind in ('O_finishes_CF_not', 'CF_finishes_O_not'):
  rows = [r for r in P['pairs'] if r['kind'] == kind and r['reproduced']]
  agg = {}
  for r in rows:
    s, k = r['seed'], r['episode']
    cf_obs, o_obs = Z[f's{s}_e{k}_CF_s{s}'], Z[f's{s}_e{k}_O_s{s}']
    fail_obs = cf_obs if kind == 'O_finishes_CF_not' else o_obs
    fin_obs = o_obs if kind == 'O_finishes_CF_not' else cf_obs
    st = r['anomaly']['first_step'] if r['anomaly']['first_step'] is not None else len(fail_obs) - 1
    segs = {'fail_before_stall': fail_obs[:max(2, st)], 'fail_after_stall': fail_obs[max(2, st):], 'finisher_path': fin_obs}
    pols = {'CF': f'CF_s{s}', 'O': f'O_s{s}', 'start': 'start', 'CFv': f'CF_s{s}@actor_lr1e-4'}
    B = {n: DT._policy_bundle(p) for n, p in pols.items()}
    for seg, O in segs.items():
      if len(O) < 2:
        continue
      O = O[::5]                                              # every 5th state
      A = {n: B[n]['mode_batch'](O) for n in pols}
      for cname in ('CF', 'O'):
        f = {n: B[cname]['f'](O, A[n]) for n in pols}
        d = agg.setdefault((seg, cname), {'n': 0, 'f_CF_minus_O': [], 'f_CF_minus_start': [], 'f_CF_minus_CFv': [], 'P_CF_gt_O': [], 'P_CF_gt_start': [], 'torque_dist_CF_O': [], 'torque_dist_CF_start': []})
        d['n'] += len(O)
        d['f_CF_minus_O'].append(float((f['CF'] - f['O']).mean())); d['f_CF_minus_start'].append(float((f['CF'] - f['start']).mean())); d['f_CF_minus_CFv'].append(float((f['CF'] - f['CFv']).mean()))
        d['P_CF_gt_O'].append(float((f['CF'] > f['O']).mean())); d['P_CF_gt_start'].append(float((f['CF'] > f['start']).mean()))
        d['torque_dist_CF_O'].append(float(np.linalg.norm(A['CF'] - A['O'], axis=1).mean())); d['torque_dist_CF_start'].append(float(np.linalg.norm(A['CF'] - A['start'], axis=1).mean()))
  out[kind] = {f'{seg}|critic {c}': {k: (round(float(np.mean(v)), 3) if isinstance(v, list) else v) for k, v in d.items()} for (seg, c), d in agg.items()}
  out[kind]['n_pairs'] = len(rows)
print(json.dumps(out, indent=1))
MP.write_json(DT.OUT / 'stall_critic_check.json', out)
