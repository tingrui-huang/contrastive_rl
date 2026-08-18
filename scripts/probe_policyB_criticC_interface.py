"""Policy-B action / Critic-C scoring interface probe (final gate before
transition modeling).

The sealed same-anchor test established f_C(s, a_rec, s'_fatal) <
f_C(s, a_rec, s'_safe) for 39/39 held-out pairs under the RECORDED dataset
action. The deployed pipeline will instead score candidates for actions
produced by the behaviorally preferred policy B. This probe asks, for the
SAME 39 held-out anchor states:

    a_B = pi_B(s, g)               (deterministic eval convention tanh(loc),
                                    g = the episode task goal, deployment
                                    zero-padded encoding already present in
                                    the stored anchor obs)
    fatal world:  (s, a_B) -> s'_fatal,B   (patched env, N=80 settle)
    safe world:   (s, a_B) -> s'_safe,B    (normal one-step transition)
    Delta_B = f_C(s, a_B, s'_safe,B) - f_C(s, a_B, s'_fatal,B)  > 0 ?

Pair validity is RE-ESTABLISHED from scratch under a_B: bitwise anchor
equality between worlds (same replay gates as the pair builder), the same
a_B array applied to both, the factual branch must STILL die on the paired
step (a different action can dodge the rock -- such episodes are excluded
and counted in the fatal-preservation rate, a property of policy B, not of
the critic), and the counterfactual branch must remain nonfatal.

Uses build_same_anchor_pairs machinery (no second simulator). C is used for
critic scoring only (its actor is never called); B is used for the action
only. Nothing is retrained or modified.

Outputs (artifacts/policyB_criticC_interface_probe/):
  pairs_Baction.npz  results.json  per_pair.csv  manifest inside results
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import beta as _beta

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import litter_pilot_common as C_mod        # noqa: E402
import rockfall_pilot as RP                # noqa: E402
from crl import networks as networks_mod   # noqa: E402
from crl import checkpoint as ckpt_mod     # noqa: E402
from crl.replay import obs_to_goal         # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402
from death_settle_sweep import make_patched_env    # noqa: E402
from build_same_anchor_pairs import replay_actions  # noqa: E402
from collect_rockfall_death_extended import PILOT_DIR  # noqa: E402

OUT_DIR = 'artifacts/policyB_criticC_interface_probe'
PAIR_DIR = 'artifacts/same_anchor_candidate_probe'
EXT_DIR = 'artifacts/rockfall_death_extended'
B_CKPT = 'failneg_clean_p30_h800_resetfix_a01_s0_300k/best.pkl'
C_CKPT = ('failneg_settledbank_a01_s0_300k/'
          'failneg_settledbank_p30_h800_resetfix_a01_s0_300k/best.pkl')
SETTLE_N = 80
OBS_DIM = 29


def clopper_pearson(k, n, alpha=0.05):
  lo = 0.0 if k == 0 else float(_beta.ppf(alpha / 2, k, n - k + 1))
  hi = 1.0 if k == n else float(_beta.ppf(1 - alpha / 2, k + 1, n - k))
  return [lo, hi]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--b-ckpt', default=B_CKPT)
  ap.add_argument('--c-ckpt', default=C_CKPT)
  ap.add_argument('--out', default=OUT_DIR)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  pairs0 = np.load(os.path.join(PAIR_DIR, 'pairs_heldout40.npz'),
                   allow_pickle=True)
  accepted_eps = np.asarray(pairs0['episode_id'], np.int64)
  src_idx = np.asarray(pairs0['source_index'], np.int64)
  cols = np.asarray(pairs0['collapse_step'], np.int64)
  cf_masks = np.asarray(pairs0['counterfactual_mask'])
  ep2row = {int(e): j for j, e in enumerate(accepted_eps)}

  ext = np.load(os.path.join(EXT_DIR, 'deaths_extended.npz'),
                allow_pickle=True)
  eman = json.load(open(os.path.join(EXT_DIR, 'manifest.json')))
  env_seed = int(eman['phase_b']['env_seed'])
  n_stream = int(eman['phase_b']['n_episodes'])
  is_fresh = ext['source'] == 'fresh'
  f_obs = ext['obs'][is_fresh]
  f_act = ext['act'][is_fresh]
  f_goal = np.asarray(ext['goal_xy'], np.float32)[is_fresh]

  # ---- networks: B actor (eval mode) + C critic ----------------------------
  cfg = build_offline_cfg()
  nets = networks_mod.make_networks(
      obs_dim=OBS_DIM, goal_dim=OBS_DIM, action_dim=8,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=False,
      use_layer_norm=cfg.use_layer_norm)
  b_step, b_state = ckpt_mod.load_checkpoint(args.b_ckpt)
  c_step, c_state = ckpt_mod.load_checkpoint(args.c_ckpt)

  @jax.jit
  def b_act(o, p=b_state.policy_params):        # deterministic eval action
    return jnp.tanh(nets.policy_network.apply(p, o).loc)

  @jax.jit
  def c_score(og, a, p=c_state.q_params):
    q = nets.q_network.apply(p, og, a)
    return jnp.diagonal(q, axis1=0, axis2=1).T  # [B, 2]

  # ---- env replay config (identical to the pair builder) -------------------
  ecfg, walker, base_act, _, _ = C_mod.load_controllers(RP.WALKER, RP.BASE)
  del walker, base_act
  ecfg.offline_dataset = ''
  ecfg.eval_goal_mode = 'd4rl'
  fatal_env = make_patched_env(ecfg, env_seed, SETTLE_N)
  safe_env = make_patched_env(ecfg, env_seed, SETTLE_N)

  rows, excluded = [], []
  n_fatal_preserved = 0
  for e in range(n_stream):
    o_f = fatal_env.reset()
    if e not in ep2row:
      safe_env.reset()
      if len(rows) + len(excluded) == len(accepted_eps):
        break
      continue
    j = ep2row[e]
    i, c = int(src_idx[j]), int(cols[j])
    o_s = safe_env.reset(mask=tuple(int(b) for b in cf_masks[j]))
    ok_f, div_f, anchor_f, _ = replay_actions(fatal_env, o_f, f_act[i],
                                              f_obs[i], c)
    ok_s, div_s, anchor_s, _ = replay_actions(safe_env, o_s, f_act[i],
                                              f_obs[i], c)
    if not (ok_f and ok_s):
      excluded.append({'episode': e, 'reason':
                       f'replay_prefix_mismatch(f@{div_f},s@{div_s})'})
      continue
    if not np.array_equal(anchor_f, anchor_s):
      excluded.append({'episode': e, 'reason': 'anchor_mismatch'})
      continue
    # sanity: the stored anchor obs carries the deployment goal encoding
    assert np.array_equal(anchor_f[29:31], f_goal[i])
    assert not anchor_f[31:].any(), 'anchor goal half not zero-padded'

    a_rec = f_act[i, c]
    a_B = np.asarray(b_act(jnp.asarray(anchor_f[None]))[0], np.float32)
    o2_f, _, _, info_f = fatal_env.step(a_B)     # same array, both worlds
    o2_s, _, _, info_s = safe_env.step(a_B)
    still_fatal = bool(info_f['dead']) and fatal_env.dead
    if still_fatal:
      n_fatal_preserved += 1
    else:
      excluded.append({'episode': e,
                       'reason': 'factual_no_longer_fatal_under_aB'})
      continue
    if info_s['dead'] or safe_env.dead:
      excluded.append({'episode': e, 'reason': 'safe_branch_fatal_under_aB'})
      continue
    if not np.isfinite(o2_s).all():
      excluded.append({'episode': e, 'reason': 'safe_branch_nonfinite'})
      continue
    rows.append({
        'episode': e, 'source_index': i, 'collapse_step': c,
        'anchor29': anchor_f[:OBS_DIM].astype(np.float32),
        'goal_xy': f_goal[i],
        'a_rec': a_rec.astype(np.float32), 'a_B': a_B,
        'action_shift_l2': float(np.linalg.norm(a_B - a_rec)),
        'fatal_candidate': o2_f[:OBS_DIM].astype(np.float32),
        'safe_candidate': o2_s[:OBS_DIM].astype(np.float32)})
    print(f'  ep {e}: B-action pair OK (|aB-arec| '
          f"{rows[-1]['action_shift_l2']:.3f})", flush=True)
    if len(rows) + len(excluded) == len(accepted_eps):
      break

  n_considered = len(accepted_eps)
  n_valid = len(rows)
  print(f'\nvalid B-action pairs: {n_valid}/{n_considered} '
        f'(fatal preserved {n_fatal_preserved}, excluded {len(excluded)})',
        flush=True)

  # ---- C-critic scoring -----------------------------------------------------
  s29 = np.stack([r['anchor29'] for r in rows])
  aB = np.stack([r['a_B'] for r in rows])
  f = {}
  for name in ('safe', 'fatal'):
    cand = np.stack([r[f'{name}_candidate'] for r in rows])
    g = obs_to_goal(cand, 0, -1, tuple(range(OBS_DIM)))
    og = np.concatenate([s29, g.astype(np.float32)], axis=1)
    f[name] = np.asarray(c_score(jnp.asarray(og), jnp.asarray(aB)))

  conv = {'f1': (f['safe'][:, 0], f['fatal'][:, 0]),
          'f2': (f['safe'][:, 1], f['fatal'][:, 1]),
          'f_min': (f['safe'].min(1), f['fatal'].min(1)),
          'f_mean': (f['safe'].mean(1), f['fatal'].mean(1))}
  shifts = np.array([r['action_shift_l2'] for r in rows])
  results = {
      'b_policy_ckpt': args.b_ckpt, 'b_step': int(b_step),
      'c_critic_ckpt': args.c_ckpt, 'c_step': int(c_step),
      'n_heldout_pairs_considered': n_considered,
      'n_valid_under_aB': n_valid,
      'n_excluded': len(excluded),
      'exclusions': excluded,
      'fatal_preservation_rate': n_fatal_preserved / n_considered,
      'action_shift_l2': {
          'mean': float(shifts.mean()), 'median': float(np.median(shifts)),
          'quantiles': {f'p{q}': float(np.percentile(shifts, q))
                        for q in (10, 25, 50, 75, 90)},
          'max_possible_note': 'actions in [-1,1]^8; max L2 = 5.66'},
      'per_convention': {}, 'per_pair_provenance': [
          {k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in r.items() if k not in
           ('anchor29', 'fatal_candidate', 'safe_candidate')}
          for r in rows],
      'sources': {'pairs_heldout40_sha256': C_mod.sha256_file(
          os.path.join(PAIR_DIR, 'pairs_heldout40.npz'))},
      'env': {'seed': env_seed, 'death_settle_substeps': SETTLE_N},
      'git_commit': C_mod.git_commit()}
  for name, (fs, ff) in conv.items():
    delta = fs - ff
    k = int((delta > 0).sum())
    results['per_convention'][name] = {
        'paired_win_rate': k / n_valid,
        'win_ci95_exact_binomial': clopper_pearson(k, n_valid),
        'mean_delta': float(delta.mean()),
        'median_delta': float(np.median(delta)),
        'min_delta': float(delta.min()),
        'n_ties': int((delta == 0).sum())}

  npz_path = os.path.join(args.out, 'pairs_Baction.npz')
  with open(npz_path + '.tmp', 'wb') as fh:
    np.savez_compressed(
        fh, anchor_obs=s29, anchor_action_B=aB,
        anchor_action_recorded=np.stack([r['a_rec'] for r in rows]),
        safe_candidate=np.stack([r['safe_candidate'] for r in rows]),
        fatal_candidate=np.stack([r['fatal_candidate'] for r in rows]),
        goal_xy=np.stack([r['goal_xy'] for r in rows]),
        episode_id=np.array([r['episode'] for r in rows], np.int64),
        collapse_step=np.array([r['collapse_step'] for r in rows], np.int64),
        meta=json.dumps({'note': 'B-policy-action same-anchor pairs; '
                                 'candidates are env one-step outcomes under '
                                 'a_B (fatal branch N=80 settle)'}))
  os.replace(npz_path + '.tmp', npz_path)
  with open(os.path.join(args.out, 'results.json'), 'w') as fh:
    json.dump(results, fh, indent=2)
  with open(os.path.join(args.out, 'per_pair.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['episode', 'shift_l2', 'f1_safe', 'f1_fatal', 'f2_safe',
                'f2_fatal', 'fmin_safe', 'fmin_fatal', 'delta_fmin'])
    for i, r in enumerate(rows):
      w.writerow([r['episode'], f"{r['action_shift_l2']:.4f}",
                  f"{f['safe'][i, 0]:.4f}", f"{f['fatal'][i, 0]:.4f}",
                  f"{f['safe'][i, 1]:.4f}", f"{f['fatal'][i, 1]:.4f}",
                  f"{f['safe'][i].min():.4f}", f"{f['fatal'][i].min():.4f}",
                  f"{f['safe'][i].min() - f['fatal'][i].min():.4f}"])

  print(f"\n== B-action pairs, C critic (B step {b_step}, C step {c_step}) ==")
  print(f"  fatal preservation: {n_fatal_preserved}/{n_considered}")
  print(f"  action shift L2: median {np.median(shifts):.3f} "
        f"mean {shifts.mean():.3f} p10 {np.percentile(shifts, 10):.3f} "
        f"p90 {np.percentile(shifts, 90):.3f}")
  for name in ('f_min', 'f_mean', 'f1', 'f2'):
    r = results['per_convention'][name]
    print(f"  {name:6s}: win {r['paired_win_rate']:.3f} "
          f"CI95 {[round(v, 3) for v in r['win_ci95_exact_binomial']]}  "
          f"median D {r['median_delta']:+.2f}  min D {r['min_delta']:+.2f}  "
          f"ties {r['n_ties']}")
  print(f"saved -> {args.out}")


if __name__ == '__main__':
  main()
