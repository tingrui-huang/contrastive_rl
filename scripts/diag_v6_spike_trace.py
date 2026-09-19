"""Trace one training collapse update by update (the pilot's own learner, streams and futures; no change of method).

The training logs of every CF and O run (and of the replays) show isolated
spikes at fixed update counts -- e.g. seed 2 at 6,500: the mean logit of the
critic batch goes from -6 to -91 (all B x B logits collapse to one large
negative value; the binary-NCE loss 0.007 -> 0.09 = 91 / 1024), the actor's
q-term 6.6 -> 178, its scale median 0.06 -> 1.1, the saturated share 0.04 ->
0.48, and the detour-entrance walking probe falls from 0.80 to 0.00 by 7,000.
The values are identical across runs that share the critic stream (bc 0 and
bc 0.05, replay and original), so the event is set by the critic batch
sequence, not by the actor.

  trace   From a saved checkpoint (default the replay's upd_6000.pkl of the
          seed) with the streams fast-forwarded to it, apply the pilot's
          update_step one update at a time for --updates updates and record
          per update: the training-batch metrics (critic loss, logits,
          gradient norms, actor terms), the parameter update norms of the
          critic and the actor, the critic on a FIXED batch (drawn once
          before the window: positive / negative logits, |phi| and |psi|),
          the critic at the 192 logged-shortcut reset rows (task goal; f of
          the logged torque and of the current mode), the actor on a FIXED
          actor batch (scale median, |loc|, saturation, E_f of the mode, the
          critic-term action gradient at the mode), and the content of the
          critic batch (max |state|, max |goal|, goal regions, anchor times,
          branch outcomes).  Checkpoints every --ckpt-every updates for the
          walking probe (diag_v6_training_replay.py probe --replay-dir).
          --freeze-critic restores the critic (params, target, Adam state)
          after every update: the same batches, the same actor Adam state,
          the critic held at the window's start.
  report  Locate the first jump and print the surrounding updates and the
          probes of both windows.

  python scripts/diag_v6_spike_trace.py trace --seed 2 --updates 1000
  python scripts/diag_v6_spike_trace.py trace --seed 2 --updates 1000 --freeze-critic
  python scripts/diag_v6_spike_trace.py report --seed 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import exp_v6_mainline_pilot as MP  # noqa: E402
import diag_v6_training_replay as TR  # noqa: E402

SPIKE = MP.OUT / 'diag_replay' / 'spike'
DEFAULT_FROM = {0: 4000, 1: 2000, 2: 6000}      # the logged window that contains each seed's first spike (4,500 / 2,500 / 6,500)
FIXED_SEED = 171_000_000


def goal_region(xy):
  from diag_v6_actor_objective_stream import goal_region_of
  return goal_region_of(np.asarray(xy))


def mode_trace(args):
  import jax
  import jax.numpy as jnp
  import optax
  from crl import checkpoint
  s = args.seed
  src = Path(args.ckpt) if args.ckpt else (TR.REPLAY / f'CF_s{s}' / f'upd_{DEFAULT_FROM[s]}.pkl')
  out = SPIKE / f'CF_s{s}' / ('frozen_critic' if args.freeze_critic else (f'clip{args.critic_clip:g}' if args.critic_clip else 'normal'))
  if (out / 'trace.json').exists() and not args.force:
    print(f'{out} exists', flush=True); return
  out.mkdir(parents=True, exist_ok=True)
  cfg, nets, update_step, critic_stream, actor_stream = TR.build(s, 'CF', MP.OUT / 'branches_cf.npz', out / '_cfg', critic_clip=args.critic_clip)
  step0, state = checkpoint.load_checkpoint(src)
  if args.critic_clip:
    # the saved Adam moments are carried; only the clip's (empty) state is added in front of them
    state = state._replace(q_optimizer_state=TR.wrap_q_opt_state(TR.build.optimizers[1], state.q_params, state.q_optimizer_state, args.critic_clip))
  ff = TR.fast_forward(critic_stream, actor_stream, int(step0))
  print(f'loaded {src} @ step {step0}; fast-forwarded {step0} batches in {ff:.0f} s', flush=True)
  anchors = critic_stream.a
  futures = critic_stream.f
  outcome = getattr(futures, 'outcome', None)
  # fixed batches (a separate RNG draw of the same laws; the training streams are not touched)
  fixed_c = MP.CriticStream(anchors, futures, cfg.batch_size, cfg.discount, FIXED_SEED + s)
  fc, (fk, fm) = fixed_c.sample()
  fa_stream = MP.ActorStream(cfg, FIXED_SEED + 100 + s)
  fa = fa_stream.sample(cfg.batch_size)
  with np.load(MP.SIDECAR, allow_pickle=True) as sc:
    route = sc['route_realized'].astype(str)
  k0 = np.flatnonzero((anchors.t == 0) & (route[anchors.episode] == 'shortcut'))
  O0, A0 = jnp.asarray(anchors.obs31[k0].astype(np.float32)), jnp.asarray(anchors.action[k0].astype(np.float32))
  fc_obs, fc_act = jnp.asarray(fc.observation), jnp.asarray(fc.action)
  fa_obs, fa_act = jnp.asarray(fa.observation), jnp.asarray(fa.action)
  B = cfg.batch_size

  @jax.jit
  def critic_fixed(qp):
    phi, psi = nets.representation_network.apply(qp, fc_obs, fc_act)         # [B, repr, 2]
    logits = jnp.einsum('ikh,jkh->ijh', phi, psi)                              # [B, B, 2]
    lm = jnp.min(logits, axis=-1)
    pos = jnp.diag(lm); neg = (lm.sum() - pos.sum()) / (B * (B - 1))
    loss = jnp.mean(jax.vmap(lambda l: optax.sigmoid_binary_cross_entropy(logits=l, labels=jnp.eye(B)), in_axes=2, out_axes=-1)(logits))
    return {'fixed_logits_pos': pos.mean(), 'fixed_logits_neg': neg, 'fixed_critic_loss': loss,
            'fixed_phi_norm': jnp.linalg.norm(phi, axis=1).mean(), 'fixed_psi_norm': jnp.linalg.norm(psi, axis=1).mean(),
            'fixed_logits_min': lm.min(), 'fixed_logits_max': lm.max()}

  @jax.jit
  def f_min(qp, o, a):
    phi, psi = nets.representation_network.apply(qp, o, a)
    return jnp.min(jnp.sum(phi * psi, axis=1), axis=1)

  @jax.jit
  def reset_rows(qp, pp):
    d = nets.policy_network.apply(pp, O0)
    mode = jnp.tanh(d.loc)
    return {'reset_f_logged': f_min(qp, O0, A0).mean(), 'reset_f_mode': f_min(qp, O0, mode).mean(),
            'reset_scale_median': jnp.median(d.scale), 'reset_loc_abs_mean': jnp.mean(jnp.abs(d.loc))}

  @jax.jit
  def actor_fixed(qp, pp):
    d = nets.policy_network.apply(pp, fa_obs)
    mode = jnp.tanh(d.loc)
    g = jax.grad(lambda a: jnp.sum(f_min(qp, fa_obs, a)))(mode)
    return {'act_scale_median': jnp.median(d.scale), 'act_loc_abs_mean': jnp.mean(jnp.abs(d.loc)), 'act_loc_abs_max': jnp.max(jnp.abs(d.loc)),
            'act_saturation': jnp.mean((jnp.abs(mode) > 0.99).astype(jnp.float32)), 'act_Ef_mode': f_min(qp, fa_obs, mode).mean(),
            'act_bc_nll': -jnp.mean(nets.log_prob(d, fa_act)), 'act_grad_f_a_norm': jnp.linalg.norm(g, axis=1).mean()}

  def pnorm(a, b):
    return float(optax.global_norm(jax.tree_util.tree_map(lambda x, y: x - y, a, b)))

  def pmax(a, b):
    return float(max(jnp.max(jnp.abs(x - y)) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))))
  upd = jax.jit(update_step)
  q0, t0, qo0 = state.q_params, state.target_q_params, state.q_optimizer_state
  checkpoint.save_named(str(out), 'init', int(step0), state)
  rows = []
  t_start = time.time()
  for i in range(args.updates):
    cb, (kk, mm) = critic_stream.sample()
    ab = actor_stream.sample(cfg.batch_size)
    tr = (MP._stack([cb]), MP._stack([ab]))
    tr = tuple(x._replace(**{f: getattr(x, f)[0] for f in x._fields}) for x in tr)     # [B, ...] (no scan axis)
    prev = state
    state, metrics = upd(state, tr)
    if args.freeze_critic:
      state = state._replace(q_params=q0, target_q_params=t0, q_optimizer_state=qo0)
    step = int(step0) + i + 1
    r = {'update': step, **{k: float(v) for k, v in metrics.items()}}
    r['critic_param_update_norm'] = 0.0 if args.freeze_critic else pnorm(state.q_params, prev.q_params)
    r['critic_param_update_max'] = 0.0 if args.freeze_critic else pmax(state.q_params, prev.q_params)
    r['actor_param_update_norm'] = pnorm(state.policy_params, prev.policy_params)
    r['actor_param_update_max'] = pmax(state.policy_params, prev.policy_params)
    r.update({k: float(v) for k, v in critic_fixed(state.q_params).items()})
    r.update({k: float(v) for k, v in reset_rows(state.q_params, state.policy_params).items()})
    r.update({k: float(v) for k, v in actor_fixed(state.q_params, state.policy_params).items()})
    # the content of this update's critic batch
    st = cb.observation[:, :MP.STATE_DIM]; g = cb.observation[:, MP.STATE_DIM:]
    greg = goal_region(g)
    r['batch'] = {'state_abs_max': float(np.abs(st).max()), 'goal_abs_max': float(np.abs(g).max()), 'action_abs_max': float(np.abs(cb.action).max()),
                  'goal_y_max': float(g[:, 1].max()), 'goal_region_share': {k: float((greg == k).mean()) for k in np.unique(greg)},
                  'anchor_t0_share': float((anchors.t[kk] == 0).mean()), 'anchor_t_mean': float(anchors.t[kk].mean()), 'future_m_mean': float(mm.mean()), 'future_m_max': int(mm.max()),
                  'branch_outcome_share': ({k: float((outcome[kk] == k).mean()) for k in ('success', 'death', 'timeout')} if outcome is not None else None),
                  'actor_state_abs_max': float(np.abs(ab.observation[:, :MP.STATE_DIM]).max()), 'actor_goal_abs_max': float(np.abs(ab.observation[:, MP.STATE_DIM:]).max())}
    rows.append(r)
    if (i + 1) % args.ckpt_every == 0 and (i + 1) < args.updates:
      checkpoint.save_named(str(out), f'upd_{step}', step, state)
    if (i + 1) % 50 == 0 or r['critic_loss'] > 0.02 or r['fixed_logits_pos'] < -20:
      print(f'[{step}] critic {r["critic_loss"]:.4f} logits_pos {r["logits_pos"]:+.1f} fixed_pos {r["fixed_logits_pos"]:+.1f} fixed_neg {r["fixed_logits_neg"]:+.1f} '
            f'|phi| {r["fixed_phi_norm"]:.2f} |psi| {r["fixed_psi_norm"]:.2f} cgn {r["critic_grad_norm"]:.2f} dq {r["critic_param_update_norm"]:.3f} '
            f'agn {r["actor_grad_norm"]:.2f} dpi {r["actor_param_update_norm"]:.3f} q_term {r["actor_q_term"]:.1f} scale {r["act_scale_median"]:.3f} sat {r["act_saturation"]:.2f} '
            f'reset f_mode {r["reset_f_mode"]:+.1f} f_logged {r["reset_f_logged"]:+.1f} | batch |s| {r["batch"]["state_abs_max"]:.0f} |g| {r["batch"]["goal_abs_max"]:.0f}', flush=True)
  checkpoint.save_named(str(out), 'final', int(step0) + args.updates, state)
  MP.write_json(out / 'trace.json', {'seed': s, 'source_ckpt': str(src), 'source_step': int(step0), 'updates': args.updates, 'freeze_critic': bool(args.freeze_critic), 'critic_clip': args.critic_clip,
                                     'fixed_batch_seed': FIXED_SEED + s, 'ckpt_every': args.ckpt_every, 'wall_seconds': time.time() - t_start, 'rows': rows})
  print(f'done: {len(rows)} updates traced in {time.time() - t_start:.0f} s -> {out}', flush=True)


def mode_rows(args):
  """Per-row attribution of the critic gradient at the trigger update: reproduce the state just before --at (from the
  nearest saved checkpoint of the normal window), take the critic batch of that update and compute, for every row i,
  the norm of dL/dphi_i (row i as an anchor) and dL/dpsi_i (row i as a goal), the representation norms, and the row's
  identity (anchor episode / t / route, branch outcome, future offset m); the same for the 20 updates before it."""
  import jax
  import jax.numpy as jnp
  import optax
  from crl import checkpoint
  s = args.seed
  win = SPIKE / f'CF_s{s}' / 'normal'
  T = MP.read_json(win / 'trace.json')
  at = args.at
  cks = sorted([(int(p.stem.split('_')[1]), p) for p in win.glob('upd_*.pkl')] + [(T['source_step'], win / 'init.pkl')])
  step0, src = max((st, p) for st, p in cks if st < at)
  cfg, nets, update_step, critic_stream, actor_stream = TR.build(s, 'CF', MP.OUT / 'branches_cf.npz', win / '_cfg')
  _, state = checkpoint.load_checkpoint(src)
  TR.fast_forward(critic_stream, actor_stream, int(step0))
  upd = jax.jit(update_step)
  anchors, futures = critic_stream.a, critic_stream.f
  with np.load(MP.SIDECAR, allow_pickle=True) as sc:
    route = sc['route_realized'].astype(str)
  B = cfg.batch_size

  def loss_from_reprs(phi, psi):
    logits = jnp.einsum('ikh,jkh->ijh', phi, psi)
    return jnp.mean(jax.vmap(lambda l: optax.sigmoid_binary_cross_entropy(logits=l, labels=jnp.eye(B)), in_axes=2, out_axes=-1)(logits))

  @jax.jit
  def row_grads(qp, obs, act):
    phi, psi = nets.representation_network.apply(qp, obs, act)
    gphi, gpsi = jax.grad(loss_from_reprs, argnums=(0, 1))(phi, psi)
    logits = jnp.min(jnp.einsum('ikh,jkh->ijh', phi, psi), axis=-1)
    return (jnp.linalg.norm(gphi.reshape(B, -1), axis=1), jnp.linalg.norm(gpsi.reshape(B, -1), axis=1),
            jnp.linalg.norm(phi.reshape(B, -1), axis=1), jnp.linalg.norm(psi.reshape(B, -1), axis=1), jnp.diag(logits), logits.mean(0), logits.mean(1))
  out = {'seed': s, 'at': at, 'from_ckpt': str(src), 'updates': []}
  for i in range(int(step0) + 1, at + 1):
    cb, (kk, mm) = critic_stream.sample()
    ab = actor_stream.sample(B)
    gphi, gpsi, nphi, npsi, lpos, lcol, lrow = (np.asarray(x) for x in row_grads(state.q_params, jnp.asarray(cb.observation), jnp.asarray(cb.action)))
    rec = {'update': i, 'gphi_norm_total': float(np.linalg.norm(gphi)), 'gpsi_norm_total': float(np.linalg.norm(gpsi)),
           'gphi_max': float(gphi.max()), 'gpsi_max': float(gpsi.max()), 'gphi_top_share': float(np.sort(gphi)[-5:].sum() ** 2 / (gphi ** 2).sum()),
           'gpsi_top_share': float(np.sort(gpsi)[-5:].sum() ** 2 / (gpsi ** 2).sum()), 'phi_norm_max': float(nphi.max()), 'psi_norm_max': float(npsi.max()),
           'phi_norm_mean': float(nphi.mean()), 'psi_norm_mean': float(npsi.mean()), 'logit_pos_min': float(lpos.min()), 'logit_pos_max': float(lpos.max())}
    if i >= at - 3:
      top = np.argsort(-np.maximum(gphi, gpsi))[:8]
      rec['top_rows'] = [{'row': int(r), 'gphi': float(gphi[r]), 'gpsi': float(gpsi[r]), 'phi_norm': float(nphi[r]), 'psi_norm': float(npsi[r]), 'logit_pos': float(lpos[r]),
                          'col_mean_logit (as goal)': float(lcol[r]), 'row_mean_logit (as anchor)': float(lrow[r]),
                          'anchor': {'episode': int(anchors.episode[kk[r]]), 't': int(anchors.t[kk[r]]), 'route': str(route[anchors.episode[kk[r]]]),
                                     'branch_outcome': (str(futures.outcome[kk[r]]) if hasattr(futures, 'outcome') else None), 'm': int(mm[r]), 'weight': float(anchors.weight[kk[r]])},
                          'state_xy': cb.observation[r, :2].tolist(), 'goal_xy': cb.observation[r, MP.STATE_DIM:].tolist(), 'state_abs_max': float(np.abs(cb.observation[r, :MP.STATE_DIM]).max())} for r in top]
    out['updates'].append(rec)
    tr = (MP._stack([cb]), MP._stack([ab]))
    tr = tuple(x._replace(**{f: getattr(x, f)[0] for f in x._fields}) for x in tr)
    state, metrics = upd(state, tr)
    rec['critic_grad_norm'] = float(metrics['critic_grad_norm']); rec['critic_loss'] = float(metrics['critic_loss'])
    print(f'[{i}] cgn {rec["critic_grad_norm"]:.3f} loss {rec["critic_loss"]:.4f} | dL/dphi total {rec["gphi_norm_total"]:.2e} max {rec["gphi_max"]:.2e} top5 share {rec["gphi_top_share"]:.2f} | '
          f'dL/dpsi total {rec["gpsi_norm_total"]:.2e} max {rec["gpsi_max"]:.2e} top5 share {rec["gpsi_top_share"]:.2f} | |phi| max {rec["phi_norm_max"]:.1f} |psi| max {rec["psi_norm_max"]:.1f} | logit_pos min {rec["logit_pos_min"]:.1f}', flush=True)
  MP.write_json(win / f'rows_{at}.json', out)
  for u in out['updates'][-3:]:
    print(f'top rows at {u["update"]}:', flush=True)
    for r in u.get('top_rows', []):
      print('   ', json.dumps(r), flush=True)


def mode_report(args):
  s = args.seed
  L = [f'# Spike trace, seed {s}: one update at a time from the replay checkpoint before the first logged spike', '']
  for arm in sorted(d.name for d in (SPIKE / f'CF_s{s}').iterdir() if (d / 'trace.json').exists()):
    p = SPIKE / f'CF_s{s}' / arm / 'trace.json'
    T = MP.read_json(p); R = T['rows']
    if T.get('critic_clip'):
      share = float(np.mean([r['critic_grad_norm'] > T['critic_clip'] for r in R]))
      L += [f'Clip {T["critic_clip"]}: raw critic gradient above the threshold in {share:.1%} of the updates; max raw gradient {max(r["critic_grad_norm"] for r in R):.3f}; '
            f'max critic parameter update {max(r["critic_param_update_norm"] for r in R):.3f}; fixed-batch positive logit min {min(r["fixed_logits_pos"] for r in R):.1f}, '
            f'fixed critic loss first / last {R[0]["fixed_critic_loss"]:.4f} / {R[-1]["fixed_critic_loss"]:.4f}; actor scale median max {max(r["act_scale_median"] for r in R):.3f}', '']
    jumps = [r for r in R if r['critic_loss'] > 0.02 or r['fixed_logits_pos'] < T['rows'][0]['fixed_logits_pos'] - 10]
    first = jumps[0]['update'] if jumps else None
    L += [f'## {arm}: from update {T["source_step"]}, {T["updates"]} updates', '', f'First jump (training critic loss > 0.02 or fixed-batch positive logit down by > 10): update {first}', '']
    L += ['| update | critic loss (batch) | logits pos (batch) | fixed pos | fixed neg | fixed loss | \\|phi\\| | \\|psi\\| | critic grad | critic dparam | actor grad | actor dparam | q-term | scale med | saturation | reset f(mode) | reset f(logged) | batch max\\|s\\| | max\\|g\\| | goal y max | branch death share |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    sel = set()
    if first is not None:
      sel |= {u for u in range(first - 6, first + 16)}
    sel |= {R[0]['update']} | {r['update'] for r in R if (r['update'] - R[0]['update']) % 100 == 0} | {R[-1]['update']}
    for r in R:
      if r['update'] in sel:
        b = r['batch']; d = b['branch_outcome_share'] or {}
        L.append(f'| {r["update"]} | {r["critic_loss"]:.4f} | {r["logits_pos"]:+.1f} | {r["fixed_logits_pos"]:+.1f} | {r["fixed_logits_neg"]:+.1f} | {r["fixed_critic_loss"]:.4f} | {r["fixed_phi_norm"]:.2f} | {r["fixed_psi_norm"]:.2f} | '
                 f'{r["critic_grad_norm"]:.2f} | {r["critic_param_update_norm"]:.3f} | {r["actor_grad_norm"]:.2f} | {r["actor_param_update_norm"]:.3f} | {r["actor_q_term"]:.1f} | {r["act_scale_median"]:.3f} | {r["act_saturation"]:.2f} | '
                 f'{r["reset_f_mode"]:+.1f} | {r["reset_f_logged"]:+.1f} | {b["state_abs_max"]:.1f} | {b["goal_abs_max"]:.1f} | {b["goal_y_max"]:.1f} | {d.get("death", float("nan")):.2f} |')
    L.append('')
    pp = SPIKE / f'CF_s{s}' / arm / 'probe.json'
    if pp.exists():
      P = MP.read_json(pp)
      L += ['Probes (route: turned north within 100 steps; walking: reach from the detour entrances):', '', '| update | turned north | reaches hazard 1 in 100 steps | entrance reach | entrance timeout |', '|---:|---:|---:|---:|---:|']
      for k in sorted(P['checkpoints'], key=int):
        r = P['checkpoints'][k]; e = r.get('entry_probe', {})
        L.append(f'| {k} | {r["route_probe"]["turned_north"]:.3f} | {r["route_probe"]["failure_within_probe"]:.2f} | {e.get("reach", float("nan")):.3f} | {e.get("timeout", float("nan")):.3f} |')
      L.append('')
  (SPIKE / f'CF_s{s}' / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('mode', choices=('trace', 'rows', 'report'))
  ap.add_argument('--at', type=int, default=None, help='rows: the update whose critic batch is attributed')
  ap.add_argument('--seed', type=int, default=2)
  ap.add_argument('--ckpt', default=None)
  ap.add_argument('--updates', type=int, default=1000)
  ap.add_argument('--ckpt-every', type=int, default=50)
  ap.add_argument('--freeze-critic', action='store_true')
  ap.add_argument('--critic-clip', type=float, default=None, help='trace: clip_by_global_norm on the critic gradient before Adam (the Adam moments of the checkpoint are carried)')
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args()
  {'trace': mode_trace, 'rows': mode_rows, 'report': mode_report}[args.mode](args)


if __name__ == '__main__':
  main()
