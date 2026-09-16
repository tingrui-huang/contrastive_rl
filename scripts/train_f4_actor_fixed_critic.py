"""Step 2 of the seed-instability diagnosis: fixed critic, fresh actor.

Take the critic (q_params) of an existing query-coverage checkpoint, freeze
it, and train a NEW actor from a fresh initialization with the unchanged
actor objective of the sealed recipe -- (1 - 0.05) * critic term + 0.05 * BC
NLL, random_goals 0.5, alpha 0, tanh-normal policy -- on the same C replay,
same 30k x 10 update budget, Adam 3e-4.  The batch order, the actor
initialization and the reparameterization keys depend only on --actor-seed,
so the same actor seed sees byte-identical inputs under different critics.
No environment step is taken during training; the saved checkpoint carries
the frozen critic as q_params so scripts/eval_pointmaze_native_routes.py can
evaluate the policy exactly as the sealed arms were evaluated.

Later steps add switches that leave the objective untouched: --log-prob acme
(Step 6), --random-goals 0 (Step 7), and --bc-sampling balanced (Step 8: the
BC term's rows are drawn with the action regions flattened inside each
(state cell, goal cell) group, the critic term keeps the buffer's batch; see
GroupBalancedBCSampler).

Usage::

  python scripts/train_f4_actor_fixed_critic.py \\
      --critic outputs/pointmaze_ett_query_coverage_20260915_v1/crl/C/final.pkl \\
      --actor-seed 0 --out outputs/pointmaze_fixed_critic_actor_v1/critic_s0/actor_s0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
SEALED = ROOT / 'outputs' / 'pointmaze_ett_query_coverage_20260915_v1'
BC_COEF = 0.05
STEPS = 30_000
G, BATCH = 10, 256
LR = 3e-4


def sha256(path):
  with Path(path).open('rb') as f:
    return hashlib.file_digest(f, 'sha256').hexdigest()


SECTOR_NAMES_8 = ('R', 'UR', 'U', 'UL', 'L', 'DL', 'D', 'DR')


class GroupBalancedBCSampler:
  """Rows for the BC term only: the buffer's relabeling law, reweighted so
  that inside every (state cell, goal cell) group the recorded-action regions
  are more evenly represented.  The critic term keeps the buffer's own draws.

  Every (episode e, anchor i, future j) triple is enumerated with exactly the
  weight the TrajectoryBuffer's variable-length sampler gives it,
      w = 1/N * 1/(L_e - 1) * gamma^(j-i) / sum_{k=1}^{L_e-1-i} gamma^k,
  (checked against the buffer empirically at start-up).  Group key = the
  floor(xy) cell of the anchor's newest frame and of the relabeled goal's
  newest frame; region = the recorded action's angle in ``n_sectors`` bins
  rotated half a width so the cardinals AND the diagonals sit at bin centres
  (the D fix's diagonal queries are their own region), plus a wait bucket
  for |a| < wait_eps.  Within a group a region's share is ceiled at ``cap``:
      w' = w * min(share, cap) / share,  renormalised so the group's total
  mass is unchanged.  So the (state, goal) marginal of the BC rows is the
  original one; only the conditional over action regions is flattened, and
  regions below the ceiling keep their natural relative frequency (no
  amplification of near-empty regions -- the failure of the strict-uniform
  key balancing in outputs/pointmaze_absorbing_balanced_g2c_20260915_v1).
  ``cap=None`` reproduces the original law with independent rows.
  """

  def __init__(self, obs, act, lengths, discount, obs_dim, cell=1.0,
               n_sectors=8, wait_eps=0.1, cap=0.25, seed=0):
    obs = np.asarray(obs)
    act = np.asarray(act)
    lengths = np.asarray(lengths, np.int64)
    n_eps = len(obs)
    tr, ii, jj = [], [], []
    for e in range(n_eps):
      lt = int(lengths[e])
      i, j = np.meshgrid(np.arange(lt - 1), np.arange(lt), indexing='ij')
      m = j > i
      tr.append(np.full(int(m.sum()), e, np.int32))
      ii.append(i[m].astype(np.int16))
      jj.append(j[m].astype(np.int16))
    self.tr, self.ii, self.jj = (np.concatenate(x) for x in (tr, ii, jj))
    lt = lengths[self.tr]
    k = lt - 1 - self.ii.astype(np.int64)
    g = float(discount)
    w = (g ** (self.jj - self.ii).astype(np.float64)
         / (g * (1.0 - g ** k) / (1.0 - g)) / (lt - 1) / n_eps)
    s_cell = np.floor(obs[self.tr, self.ii, :2] / cell).astype(np.int64)
    g_cell = np.floor(obs[self.tr, self.jj, :2] / cell).astype(np.int64)
    a = act[self.tr, self.ii].astype(np.float64)
    mag = np.linalg.norm(a, axis=1)
    ang = np.arctan2(a[:, 1], a[:, 0])
    width = 2.0 * np.pi / n_sectors
    region = np.floor((ang + width / 2.0) / width).astype(np.int64) % n_sectors
    region = np.where(mag < wait_eps, n_sectors, region)
    key = np.stack([s_cell[:, 0], s_cell[:, 1], g_cell[:, 0], g_cell[:, 1]], 1)
    self.group_keys, g_id = np.unique(key, axis=0, return_inverse=True)
    g_id = g_id.ravel().astype(np.int64)
    n_reg = n_sectors + 1
    b_id = g_id * n_reg + region
    n_groups = len(self.group_keys)
    w_group = np.bincount(g_id, weights=w, minlength=n_groups)
    w_bucket = np.bincount(b_id, weights=w, minlength=n_groups * n_reg)
    share = w_bucket / np.repeat(w_group, n_reg)
    if cap is None:
      mult = np.ones_like(w_bucket)
    else:
      m = np.where(share > 0, np.minimum(share, cap) / np.maximum(share, 1e-300), 0.0)
      new_mass = np.bincount(np.repeat(np.arange(n_groups), n_reg),
                             weights=m * w_bucket, minlength=n_groups)
      mult = m * np.repeat(w_group / np.maximum(new_mass, 1e-300), n_reg)
    self.w = w * mult[b_id]
    self.w_original = w
    self.b_id, self.g_id, self.region = b_id, g_id, region
    self.n_reg, self.n_sectors, self.cap, self.cell = n_reg, n_sectors, cap, cell
    self.wait_eps = wait_eps
    self.w_bucket, self.w_bucket_new = w_bucket, w_bucket * mult
    self.cdf = np.cumsum(self.w / self.w.sum())
    self.cdf[-1] = 1.0
    self.rng = np.random.default_rng(seed)
    self.obs, self.act, self.obs_dim = obs, act, obs_dim
    self.stats = {
        'n_pairs': int(len(w)), 'n_groups': int(n_groups),
        'n_nonempty_buckets': int((w_bucket > 0).sum()), 'cap': cap,
        'cell': cell, 'n_sectors': n_sectors, 'wait_eps': wait_eps,
        'max_multiplier': float(mult.max()), 'min_nonzero_multiplier':
            float(mult[mult > 0].min()),
        'ess_original': float(w.sum() ** 2 / (w ** 2).sum()),
        'ess_reweighted': float(self.w.sum() ** 2 / (self.w ** 2).sum()),
        'group_marginal_max_abs_dev': float(np.abs(
            np.bincount(g_id, weights=self.w, minlength=n_groups) - w_group).max()),
    }

  def group_composition(self, s_cell, g_cell):
    """(original shares, reweighted shares) over regions for one group."""
    hit = np.nonzero((self.group_keys == np.array([*s_cell, *g_cell])).all(1))[0]
    if len(hit) == 0:
      return None
    sl = slice(hit[0] * self.n_reg, (hit[0] + 1) * self.n_reg)
    b, bn = self.w_bucket[sl], self.w_bucket_new[sl]
    names = list(SECTOR_NAMES_8 if self.n_sectors == 8 else
                 [f's{r}' for r in range(self.n_sectors)]) + ['wait']
    return ({n: float(v) for n, v in zip(names, b / b.sum())},
            {n: float(v) for n, v in zip(names, bn / bn.sum())},
            float(b.sum()))

  def bucket_of(self, traj, i, j):
    s_cell = np.floor(self.obs[traj, i, :2] / self.cell).astype(np.int64)
    g_cell = np.floor(self.obs[traj, j, :2] / self.cell).astype(np.int64)
    a = self.act[traj, i].astype(np.float64)
    mag = np.linalg.norm(a, axis=1)
    ang = np.arctan2(a[:, 1], a[:, 0])
    width = 2.0 * np.pi / self.n_sectors
    region = np.floor((ang + width / 2.0) / width).astype(np.int64) % self.n_sectors
    region = np.where(mag < self.wait_eps, self.n_sectors, region)
    key = np.stack([s_cell[:, 0], s_cell[:, 1], g_cell[:, 0], g_cell[:, 1]], 1)
    # groups are lexicographically sorted by np.unique -> searchsorted on a view
    flat = lambda arr: np.ascontiguousarray(arr).view(
        np.dtype((np.void, arr.dtype.itemsize * 4))).ravel()
    gk = flat(self.group_keys)
    pos = np.searchsorted(gk, flat(key))
    pos = np.minimum(pos, len(gk) - 1)
    ok = gk[pos] == flat(key)
    return np.where(ok, pos * self.n_reg + region, -1)

  def law_check(self, buffer, n=200_000, chunk=1000):
    """Max abs deviation between the buffer's empirical bucket shares and the
    enumerated original weights."""
    cnt = np.zeros(len(self.w_bucket))
    for _ in range(n // chunk):
      t, i, j = buffer.sampled_indices(chunk)
      b = self.bucket_of(t, i, j)
      if (b < 0).any():
        raise RuntimeError('buffer drew a (state, goal) group missing from '
                           'the enumeration')
      cnt += np.bincount(b, minlength=len(cnt))
    return float(np.abs(cnt / n - self.w_bucket).max())

  def sample(self, batch_size):
    from crl import losses as losses_mod
    pos = np.searchsorted(self.cdf, self.rng.random(batch_size), side='right')
    pos = np.minimum(pos, len(self.cdf) - 1)
    traj, i, j = self.tr[pos], self.ii[pos].astype(np.int64), self.jj[pos].astype(np.int64)
    state = self.obs[traj, i, :self.obs_dim].astype(np.float32)
    next_state = self.obs[traj, i + 1, :self.obs_dim].astype(np.float32)
    goal = self.obs[traj, j, :self.obs_dim].astype(np.float32)   # obs_to_goal(0, -1)
    return losses_mod.Transition(
        observation=np.concatenate([state, goal], 1),
        action=self.act[traj, i].astype(np.float32),
        reward=np.zeros((batch_size,), np.float32),
        discount=np.full((batch_size,), 0.95, np.float32),
        next_observation=np.concatenate([next_state, goal], 1),
        next_action=self.act[traj, i + 1].astype(np.float32))


def tree_sha(tree):
  import jax
  d = hashlib.sha256()
  for leaf in jax.tree_util.tree_leaves(tree):
    a = np.ascontiguousarray(np.asarray(leaf))
    d.update(str(a.dtype).encode())
    d.update(str(a.shape).encode())
    d.update(a.tobytes())
  return d.hexdigest()


def make_network(log_prob_mode='clip'):
  from crl import networks
  return networks.make_networks(
      8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
      use_image_obs=False, use_layer_norm=False, obs_scale=None,
      log_prob_mode=log_prob_mode)


def crl_config(replay, seed):
  from crl.config import Config
  return Config(
      env_name='point_two_route_swamp_windy_f4_v0', offline_dataset=str(replay),
      obs_dim=8, goal_dim=8, action_dim=2, max_episode_steps=50,
      start_index=0, end_index=-1, max_number_of_steps=STEPS,
      fail_bank_path='', fail_neg_alpha=0.0, obs_norm_mode='',
      obs_norm_z_scale=0.0, anchor_cut_mode='', balanced_sampling=False,
      use_td=False, use_cpc=False, use_gcbc=False, twin_q=False,
      bc_coef=BC_COEF, random_goals=0.5, entropy_coefficient=0.0,
      target_entropy=0.0, batch_size=BATCH, repr_dim=64,
      hidden_layer_sizes=(256, 256), discount=0.95, learning_rate=LR,
      actor_learning_rate=LR, num_sgd_steps_per_step=G, num_actors=0,
      guard_abort=True, jit=True, seed=int(seed))


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--critic', required=True,
                  help='checkpoint whose q_params are frozen and reused')
  ap.add_argument('--actor-seed', type=int, required=True)
  ap.add_argument('--replay', default=str(SEALED / 'replay_C.npz'))
  ap.add_argument('--steps', type=int, default=STEPS)
  ap.add_argument('--log-prob', choices=('clip', 'acme'), default='clip',
                  help="BC log-prob at the action boundary: 'clip' = this "
                       "port's historical rule (clip to 1-1e-6, density at "
                       "atanh), 'acme' = dm-acme 0.4.0's boundary-band rule "
                       "used by the original contrastive_rl actor")
  ap.add_argument('--random-goals', type=float, choices=(0.0, 0.5, 1.0),
                  default=0.5,
                  help="actor goal pairing, as crl/losses.py actor_loss: 0.5 = "
                       "the sealed recipe (batch doubled, second half with "
                       "rolled goals); 0.0 = the original offline pairing "
                       "(each state with its own relabeled future goal only); "
                       "1.0 = rolled goals only")
  ap.add_argument('--bc-sampling', choices=('shared', 'independent', 'balanced'),
                  default='shared',
                  help="rows of the BC term: 'shared' = the same buffer batch "
                       "as the critic term (all earlier steps); 'independent' "
                       "= a second batch from the same relabeling law; "
                       "'balanced' = a second batch with the action regions "
                       "flattened inside each (state cell, goal cell) group "
                       "(GroupBalancedBCSampler).  The critic term always "
                       "uses the buffer's own batch.  random_goals 0 only.")
  ap.add_argument('--bc-cap', type=float, default=0.25,
                  help='ceiling on one action region\'s share of its group')
  ap.add_argument('--bc-cell', type=float, default=1.0)
  ap.add_argument('--bc-sectors', type=int, default=8)
  ap.add_argument('--out', required=True)
  args = ap.parse_args(argv)
  if args.bc_sampling != 'shared' and float(args.random_goals) != 0.0:
    ap.error('--bc-sampling independent/balanced is defined for --random-goals 0')

  import jax
  import jax.numpy as jnp
  import optax
  from crl import checkpoint, offline_audit
  from crl import losses as losses_mod

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  if (out / 'final.pkl').exists():
    print(f'{out}: final.pkl exists', flush=True)
    return 0
  nets = make_network(args.log_prob)
  cfg = crl_config(args.replay, args.actor_seed)
  # buffer RNG = actor seed -> identical batch order across critics
  buffer, fp = offline_audit.build_offline_buffer(cfg.offline_dataset, cfg)
  _, critic_state = checkpoint.load_checkpoint(args.critic)
  q_params = critic_state.q_params

  key = jax.random.PRNGKey(int(args.actor_seed))
  key, k_init = jax.random.split(key)
  policy_params = nets.policy_network.init(k_init)
  optimizer = optax.adam(LR, eps=1e-7)
  opt_state = optimizer.init(policy_params)
  obs_dim = cfg.obs_dim

  random_goals = float(args.random_goals)
  bc_sampler = None
  bc_audit = None
  if args.bc_sampling != 'shared':
    with np.load(args.replay, allow_pickle=False) as d:
      bc_sampler = GroupBalancedBCSampler(
          d['obs'], d['act'], d['lengths'], cfg.discount, obs_dim,
          cell=args.bc_cell, n_sectors=args.bc_sectors,
          cap=None if args.bc_sampling == 'independent' else args.bc_cap,
          seed=10_000 + int(args.actor_seed))     # own stream: the critic
    # batch order stays byte-identical to the shared runs
    law_dev = bc_sampler.law_check(buffer)
    fork = {f'fork(1,3)->goal{g}': bc_sampler.group_composition((1, 3), g)
            for g in ((8, 3), (2, 3), (1, 2), (1, 3), (3, 3), (4, 3))}
    bc_audit = {'stats': bc_sampler.stats, 'law_check_max_abs_share_dev': law_dev,
                'fork_groups': fork}
    fo, fn, fm = fork['fork(1,3)->goal(8, 3)']
    print(f'BC rows: {args.bc_sampling} (cap {bc_sampler.cap}, cell '
          f'{bc_sampler.cell}, {bc_sampler.n_sectors} sectors + wait): '
          f'{bc_sampler.stats["n_pairs"]:,} (e,i,j) pairs, '
          f'{bc_sampler.stats["n_groups"]} groups, '
          f'{bc_sampler.stats["n_nonempty_buckets"]} buckets, max multiplier '
          f'{bc_sampler.stats["max_multiplier"]:.2f}, ESS '
          f'{bc_sampler.stats["ess_original"] / 1e6:.2f}M -> '
          f'{bc_sampler.stats["ess_reweighted"] / 1e6:.2f}M, law check max '
          f'share dev {law_dev:.1e}', flush=True)
    print('  fork (1,3) -> goal (8,3) [' + f'{fm:.4f} of BC mass]: '
          + ' '.join(f'{k} {fo[k]:.2f}->{fn[k]:.2f}' for k in fo if fo[k] > 0.005),
          flush=True)

  def actor_loss(params, transitions, k, bc_transitions=None):
    # goal pairing exactly as crl/losses.py actor_loss's three branches
    obs = transitions.observation
    state, goal = obs[:, :obs_dim], obs[:, obs_dim:]
    if random_goals == 0.0:
      new_state, new_goal, orig_action = state, goal, transitions.action
    elif random_goals == 0.5:
      new_state = jnp.concatenate([state, state], axis=0)
      new_goal = jnp.concatenate([goal, jnp.roll(goal, 1, axis=0)], axis=0)
      orig_action = jnp.concatenate([transitions.action, transitions.action], 0)
    else:
      new_state, new_goal = state, jnp.roll(goal, 1, axis=0)
      orig_action = transitions.action
    new_obs = jnp.concatenate([new_state, new_goal], axis=1)
    dist = nets.policy_network.apply(params, new_obs)
    action = nets.sample(dist, k)
    q = nets.q_network.apply(q_params, new_obs, action)
    q_term = -jnp.diag(q)                       # alpha = 0
    if bc_transitions is None:
      bc_dist, bc_action = dist, orig_action
    else:
      # BC term on its own rows (same law, or region-balanced within groups)
      bc_dist = nets.policy_network.apply(params, bc_transitions.observation)
      bc_action = bc_transitions.action
    bc_nll = -nets.log_prob(bc_dist, bc_action)
    loss = BC_COEF * bc_nll + (1 - BC_COEF) * q_term
    mode = jnp.tanh(dist.loc)
    return jnp.mean(loss), {
        'q_term': jnp.mean(q_term), 'bc_nll': jnp.mean(bc_nll),
        'policy_scale_median': jnp.median(dist.scale),
        'pre_tanh_loc_abs_mean': jnp.mean(jnp.abs(dist.loc)),
        'action_saturation_fraction':
            jnp.mean((jnp.abs(mode) > 0.99).astype(jnp.float32))}

  def one_update(carry, batch):
    params, opt_state, k = carry
    transitions, bc_transitions = batch
    k, sub = jax.random.split(k)
    (loss, aux), grads = jax.value_and_grad(actor_loss, has_aux=True)(
        params, transitions, sub, bc_transitions)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    aux['loss'] = loss
    return (params, opt_state, k), aux

  @jax.jit
  def multi_update(params, opt_state, k, trans_G, bc_G):
    (params, opt_state, k), aux = jax.lax.scan(
        one_update, (params, opt_state, k), (trans_G, bc_G))
    return params, opt_state, k, jax.tree_util.tree_map(jnp.mean, aux)

  def stack_G(batches):
    return losses_mod.Transition(*[
        jnp.asarray(np.stack([getattr(b, f) for b in batches], axis=0))
        for f in losses_mod.Transition._fields])

  def sample_G():
    trans = stack_G([buffer.sample(BATCH) for _ in range(G)])
    bc = (None if bc_sampler is None
          else stack_G([bc_sampler.sample(BATCH) for _ in range(G)]))
    return trans, bc

  print(f'fixed critic {args.critic} (q sha {tree_sha(q_params)[:12]}) | '
        f'actor seed {args.actor_seed} init sha {tree_sha(policy_params)[:12]} '
        f'| replay {args.replay} sha {fp["sha256"][:12]} | {args.steps} x {G} '
        f'updates, bc {BC_COEF} | log_prob {args.log_prob} | random_goals '
        f'{random_goals:g} | bc rows {args.bc_sampling}', flush=True)
  init_sha = tree_sha(policy_params)
  history = []
  t0 = time.time()
  for step in range(1, args.steps + 1):
    trans_G, bc_G = sample_G()
    policy_params, opt_state, key, aux = multi_update(
        policy_params, opt_state, key, trans_G, bc_G)
    if step % 1000 == 0 or step == args.steps:
      rec = {k: float(v) for k, v in aux.items()}
      rec['step'] = step
      history.append(rec)
      print(f'[step {step:6d}] loss {rec["loss"]:.4f} q_term {rec["q_term"]:.3f} '
            f'bc_nll {rec["bc_nll"]:.3f} scale {rec["policy_scale_median"]:.3f} '
            f'|loc| {rec["pre_tanh_loc_abs_mean"]:.2f} sat '
            f'{rec["action_saturation_fraction"]:.2f} '
            f'({(time.time() - t0) / step * (args.steps - step):.0f}s left)',
            flush=True)
  state = losses_mod.TrainingState(
      policy_optimizer_state=opt_state, q_optimizer_state=None,
      policy_params=policy_params, q_params=q_params, target_q_params=q_params,
      key=key)
  checkpoint.save_named(str(out), 'final', args.steps, state)
  with open(out / 'run_summary.json', 'w', encoding='utf-8') as f:
    json.dump({'critic': args.critic, 'critic_sha256': sha256(args.critic),
               'critic_q_tree_sha256': tree_sha(q_params),
               'actor_seed': args.actor_seed, 'actor_init_sha256': init_sha,
               'final_policy_sha256': tree_sha(policy_params),
               'replay': args.replay, 'replay_sha256': fp['sha256'],
               'steps': args.steps, 'sgd_steps_per_step': G, 'batch': BATCH,
               'bc_coef': BC_COEF, 'lr': LR,
               'log_prob_mode': args.log_prob, 'random_goals': random_goals,
               'bc_sampling': args.bc_sampling, 'bc_sampling_audit': bc_audit,
               'critic_frozen': True, 'env_steps_during_training': 0,
               'wall_seconds': time.time() - t0, 'history': history,
               'jax_devices': [str(d) for d in jax.devices()]}, f, indent=2)
  print(f'-> {out / "final.pkl"}', flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
