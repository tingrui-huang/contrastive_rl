#!/usr/bin/env python
"""AntMaze V6 -- the three-layer PAIRING check (user's plan after multi_futures, 2026-09-21): with the two oracle tables and the
trained critics / actors (CF = sealed draw, CF2 = draw 2, MF = both), no training.  The question: after mixing the two tables, did the
detour signal weaken IN THE DATA, did the CRITIC fail to learn it, or did the ACTOR not adopt it?  And, first of all: did the
sealed-table critic learn a repeatable consequence, or the particular (state, action) -> future pairing of its one draw?
Emphasis on the reset and early-fork states (start region, the west-column entry, the first pre-zone-1 metres), not only the mouth.

Layer 1 (data)   per anchor and per table, the EXACT probability that the critic's sampling law (P(m) ~ gamma^m within the path)
                 lands the positive goal near the task goal (reach radius 0.5; 2.0; the goal area) / in the far-route regions / on the
                 death frame; their average over the two tables; per-anchor repeatability corr(q_T1, q_T2); the agreement groups;
                 the 'detour signal' = the difference of these masses between anchors whose logged torque comes from a detour episode
                 and from a shortcut episode, within a stratum, per table and averaged; the ACTOR's actual relabelled goals (the
                 recorded-episode law) for the same rows.
Layer 2 (critic) the 15 critics on the SAME inputs: (a) the training NCE loss / accuracy with positives from T1, T2 and the mixture on
                 identical anchor batches; (b) the pairing test -- the critic's logit at the anchor's own (s, a, task goal) correlated
                 with the realised consequence under T1, T2 and the average, per stratum (own draw vs the other draw); (c) region read-
                 outs normalised per critic (softmax over a common goal set drawn from both tables), compared with the true masses.
Layer 3 (actor)  at the same state and the same goal (the task goal AND relabelled training goals), fixed candidate actions (the logged
                 torque; the modes of the CF / CF2 / MF / start / O policies at THIS state) under each critic, and the actor's full
                 objective (the sampled critic term with the twin-min + the BC term, as in crl.losses.actor_loss) for each policy under
                 each critic.  No cross-state teacher actions.
Outputs under pairing_check/: data.json, critic.json, actor.json, REPORT.md.
"""
from __future__ import annotations

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
import exp_v6_ett_futures as EF  # noqa: E402
import exp_v6_oracle_draw2 as D2  # noqa: E402
import exp_v6_multi_futures as MFD  # noqa: E402
from exp_v6_learned_ett import REGIONS, region_of, FAR  # noqa: E402

OUT = MP.OUT / 'pairing_check'
TABLES = {'T1': MP.OUT / 'branches_cf.npz', 'T2': D2.BRANCHES}
GAMMA = MP.GAMMA
STATE_DIM, GOAL_DIM, OBS_W, ACTION_DIM = MP.STATE_DIM, MP.GOAL_DIM, MP.OBS_W, MP.ACTION_DIM
RADII = {'reach0.5': 0.5, 'near2.0': 2.0}
SEEDS = (0, 1, 2, 3, 4)
FAMILIES = ('CF', 'CF2', 'MF')
BC = 0.05
STRATA = ('reset', 'start_early', 'west_entry', 'pre_zone1_early', 'pre_mouth', 'zone1', 'between', 'far_legs', 'all')


def ckpt_of(fam, s):
  return {'CF': EF.run_dir('CF', s) / 'final.pkl', 'CF2': D2.run_dir2(s) / 'final.pkl', 'MF': MFD.run_dir_mf(s) / 'final.pkl', 'O': EF.run_dir('O', s) / 'final.pkl'}[fam]


# ------------------------------------------------------------------ strata
def strata_masks(anchors):
  xy = anchors.state[:, :2].astype(np.float64); reg = region_of(xy); t = anchors.t.astype(np.int64)
  x, y = xy[:, 0], xy[:, 1]
  m = {'reset': t == 0,
       'start_early': (reg == REGIONS.index('start')) & (t >= 1) & (t <= 40),
       'west_entry': (reg == REGIONS.index('west_column')) & (y < 4.0),
       'pre_zone1_early': (reg == REGIONS.index('pre_zone1')) & (x < 4.5),
       'pre_mouth': (reg == REGIONS.index('pre_zone1')) & (x >= 4.5),
       'zone1': reg == REGIONS.index('zone1'),
       'between': reg == REGIONS.index('between'),
       'far_legs': np.isin(reg, FAR),
       'all': np.ones(anchors.n, bool)}
  return m, reg


def source_route(anchors, obs, lengths):
  """The anchor's source episode took the far route (any recorded row in the far-route regions) or the shortcut."""
  det = {}
  for e in np.unique(anchors.episode):
    r = region_of(obs[e, :lengths[e], :2].astype(np.float64)); det[int(e)] = bool(np.isin(r, FAR).any())
  return np.array([det[int(e)] for e in anchors.episode])


# ------------------------------------------------------------------ layer 1
def table_masses(path, anchors):
  """Per anchor: the mass of the critic's positive law on goal rows near the task goal / in the far regions / on the death frame."""
  with np.load(path, allow_pickle=False) as d:
    xy = np.ascontiguousarray(d['obs_rows'][:, :2]).astype(np.float64); off = d['offset'].astype(np.int64); L = d['length'].astype(np.int64); oc = d['outcome'].astype(str)
  K = len(L); row_of = np.repeat(np.arange(K), L); m = np.arange(len(row_of)) - np.repeat(off, L); after = m >= 1
  g = np.where(after, GAMMA ** m, 0.0); z = np.bincount(row_of, weights=g, minlength=K)
  dist = np.linalg.norm(xy - anchors.goal_xy[row_of].astype(np.float64), axis=1); reg = region_of(xy)
  out = {}
  for name, r in RADII.items():
    out[name] = np.bincount(row_of, weights=g * (dist <= r), minlength=K) / np.maximum(z, 1e-12)
  out['goal_area'] = np.bincount(row_of, weights=g * (reg == REGIONS.index('goal_area')), minlength=K) / np.maximum(z, 1e-12)
  out['far'] = np.bincount(row_of, weights=g * np.isin(reg, FAR), minlength=K) / np.maximum(z, 1e-12)
  term = off + L - 1
  out['death_frame'] = np.where(oc == 'death', g[term] / np.maximum(z, 1e-12), 0.0)
  out['outcome'] = oc
  return out


def recorded_masses(anchors, obs, lengths):
  """The ACTOR's relabelled-goal law on the anchor's own recorded row: P(m) ~ gamma^m over 1..L_e - 1 - t within the recorded episode."""
  K = anchors.n; res = {k: np.zeros(K) for k in list(RADII) + ['goal_area', 'far']}
  for k in range(K):
    e, t = int(anchors.episode[k]), int(anchors.t[k]); L = int(lengths[e])
    n = L - 1 - t
    if n < 1:
      continue
    mm = np.arange(1, n + 1); g = GAMMA ** mm; g /= g.sum()
    xy = obs[e, t + mm, :2].astype(np.float64); dist = np.linalg.norm(xy - anchors.goal_xy[k].astype(np.float64), axis=1); reg = region_of(xy)
    for name, r in RADII.items():
      res[name][k] = float((g * (dist <= r)).sum())
    res['goal_area'][k] = float((g * (reg == REGIONS.index('goal_area'))).sum()); res['far'][k] = float((g * np.isin(reg, FAR)).sum())
  return res


def wmean(x, w):
  w = np.asarray(w, float); return float((w * x).sum() / max(w.sum(), 1e-12))


def corr(a, b):
  a = np.asarray(a, float); b = np.asarray(b, float)
  if len(a) < 20 or a.std() < 1e-12 or b.std() < 1e-12:
    return None
  return float(np.corrcoef(a, b)[0, 1])


def _rank(x):
  o = np.argsort(x, kind='stable'); r = np.empty(len(x), float); r[o] = np.arange(len(x), dtype=float); return r


def spearman(a, b):
  a = np.asarray(a, float); b = np.asarray(b, float)
  if len(a) < 20 or a.std() < 1e-12 or b.std() < 1e-12:
    return None
  return corr(_rank(a), _rank(b))


def layer_data(anchors, obs, lengths, masks, det):
  T = {name: table_masses(p, anchors) for name, p in TABLES.items()}
  R = recorded_masses(anchors, obs, lengths)
  W = anchors.weight
  keys = list(RADII) + ['goal_area', 'far', 'death_frame']
  res = {'strata_n': {s: int(m.sum()) for s, m in masks.items()}, 'strata_detour_share': {s: float(det[m].mean()) if m.sum() else None for s, m in masks.items()}, 'by_stratum': {}}
  agree = T['T1']['outcome'] == T['T2']['outcome']
  for s, m in masks.items():
    if m.sum() < 30:
      continue
    row = {'n': int(m.sum()), 'weight_share': float(W[m].sum() / W.sum()), 'outcome_agreement': float(agree[m].mean()),
           'outcome_pairs': {f'{a}->{b}': int(((T['T1']['outcome'][m] == a) & (T['T2']['outcome'][m] == b)).sum()) for a in ('success', 'death', 'timeout') for b in ('success', 'death', 'timeout')}}
    for key in keys:
      q1, q2 = T['T1'][key][m], T['T2'][key][m]; qa = 0.5 * (q1 + q2)
      row[key] = {'mean_T1': wmean(q1, W[m]), 'mean_T2': wmean(q2, W[m]), 'mean_avg': wmean(qa, W[m]),
                  'corr_T1_T2': corr(q1, q2), 'spearman_T1_T2': spearman(q1, q2),
                  'mean_abs_diff_T1_T2': float(np.abs(q1 - q2).mean()), 'share_anchors_disagree_gt_0.2': float((np.abs(q1 - q2) > 0.2).mean())}
      d, sc = det[m], ~det[m]
      if d.sum() >= 20 and sc.sum() >= 20:
        row[key]['detour_signal'] = {'T1': wmean(q1[d], W[m][d]) - wmean(q1[sc], W[m][sc]), 'T2': wmean(q2[d], W[m][d]) - wmean(q2[sc], W[m][sc]),
                                     'avg': wmean(qa[d], W[m][d]) - wmean(qa[sc], W[m][sc]), 'n_detour': int(d.sum()), 'n_shortcut': int(sc.sum()),
                                     'mean_detour_T1/T2': [wmean(q1[d], W[m][d]), wmean(q2[d], W[m][d])], 'mean_shortcut_T1/T2': [wmean(q1[sc], W[m][sc]), wmean(q2[sc], W[m][sc])]}
      if key in R:
        row[key]['actor_relabelled_goal_mass'] = {'mean': wmean(R[key][m], W[m]), 'detour_rows': (wmean(R[key][m][d], W[m][d]) if d.sum() else None), 'shortcut_rows': (wmean(R[key][m][sc], W[m][sc]) if sc.sum() else None)}
    res['by_stratum'][s] = row
  return res, T, R


# ------------------------------------------------------------------ nets / checkpoints
class Nets:
  def __init__(self):
    import jax
    import jax.numpy as jnp
    cfg = MP.recipe_config(0, OUT / '_cfg'); MP.fill_dims(cfg); self.cfg = cfg
    self.nets = MP.make_nets(cfg)
    n = self.nets
    self._reprs = jax.jit(lambda p, o, a: n.representation_network.apply(p, o, a))       # phi [B, d, 2], psi [B, d, 2]
    self._pol = jax.jit(lambda p, o: n.policy_network.apply(p, o))
    self._logp = jax.jit(lambda dp, a: n.log_prob(dp, a))
    self._sample = jax.jit(lambda dp, key: n.sample(dp, key))
    self.jnp = jnp; self.jax = jax

  def load(self, path):
    from crl import checkpoint
    _, st = checkpoint.load_checkpoint(path)
    return st.q_params, st.policy_params

  def logits(self, qp, obs_sa, act, obs_g):
    """f(s_i, a_i, g_j) for all i, j: [n, m, 2] (the goal columns of obs_sa are ignored by phi; obs_g carries the goals)."""
    jnp = self.jnp
    phi, _ = self._reprs(qp, jnp.asarray(obs_sa, jnp.float32), jnp.asarray(act, jnp.float32))
    dummy_a = jnp.zeros((len(obs_g), ACTION_DIM), jnp.float32)
    _, psi = self._reprs(qp, jnp.asarray(obs_g, jnp.float32), dummy_a)
    return np.asarray(jnp.einsum('idh,jdh->ijh', phi, psi))

  def diag_logits(self, qp, obs, act, chunk=512):
    out = []
    for i in range(0, len(obs), chunk):
      o, a = obs[i:i + chunk], act[i:i + chunk]
      f = self.logits(qp, o, a, o)                                             # [c, c, 2]
      out.append(np.einsum('iih->ih', f))
    return np.concatenate(out)                                                  # [n, 2]

  def policy(self, pp, obs):
    return self._pol(pp, self.jnp.asarray(obs, self.jnp.float32))

  def mode(self, pp, obs):
    return np.asarray(self.jnp.tanh(self.policy(pp, obs).loc))

  def log_prob(self, dp, a):
    return np.asarray(self._logp(dp, self.jnp.asarray(a, self.jnp.float32)))

  def sample(self, dp, key):
    return np.asarray(self._sample(dp, key))


def check_logits_match(N, qp, obs, act):
  """The repr-based f equals the q_network's pairwise logits (guards the einsum layout)."""
  import jax.numpy as jnp
  ref = np.asarray(N.nets.q_network.apply(qp, jnp.asarray(obs, jnp.float32), jnp.asarray(act, jnp.float32)))
  f = N.logits(qp, obs, act, obs)
  return float(np.abs(ref - f).max())


# ------------------------------------------------------------------ layer 2
def nce_fit(N, params, anchors, n_batches=30, seed=777):
  """Training-identical NCE batches (anchor by weight; positive by the law) with positives from T1 / T2 / the mixture; the anchor
  sequence is identical across the three sources (same stream seed)."""
  import optax
  import jax.numpy as jnp
  srcs = {'T1': MP.BranchFutures(anchors, TABLES['T1']), 'T2': MP.BranchFutures(anchors, TABLES['T2'])}
  srcs['mix'] = MP.MultiFutures([srcs['T1'], srcs['T2']], GAMMA)
  batches = {}
  for name, f in srcs.items():
    st = MP.CriticStream(anchors, f, 1024, GAMMA, seed); batches[name] = [st.sample()[0] for _ in range(n_batches)]
  I = np.eye(1024, dtype=np.float32)
  res = {}
  for cname, qp in params.items():
    res[cname] = {}
    for name, bl in batches.items():
      losses, accs = [], []
      for tr in bl:
        f = N.logits(qp, tr.observation, tr.action, tr.observation)           # [B, B, 2]
        l = np.asarray(optax.sigmoid_binary_cross_entropy(logits=jnp.asarray(f), labels=jnp.asarray(I)[:, :, None])).mean()
        acc = float((f.mean(-1).argmax(axis=1) == np.arange(1024)).mean())
        losses.append(float(l)); accs.append(acc)
      res[cname][name] = {'nce_loss': float(np.mean(losses)), 'cat_acc': float(np.mean(accs))}
  return res


def pairing_test(N, params, anchors, masks, T, det, sample_seed=11, per_stratum=3000):
  """Per critic: the logit at (s_k, a_k, task goal_k) vs the realised consequence under T1 / T2 / their average, per stratum."""
  rng = np.random.default_rng(sample_seed)
  sel = {}
  for s, m in masks.items():
    idx = np.nonzero(m)[0]
    sel[s] = idx if len(idx) <= per_stratum else np.sort(rng.choice(idx, per_stratum, replace=False))
  allidx = np.unique(np.concatenate(list(sel.values())))
  obs = anchors.obs31[allidx].astype(np.float32); act = anchors.action[allidx].astype(np.float32)
  scores = {}
  for cname, qp in params.items():
    f = N.diag_logits(qp, obs, act)                                            # [n, 2]
    scores[cname] = {'mean': f.mean(-1), 'min': f.min(-1)}
  pos = {int(k): i for i, k in enumerate(allidx)}
  res = {}
  for s, idx in sel.items():
    ii = np.array([pos[int(k)] for k in idx]); row = {'n': int(len(idx))}
    q1, q2 = T['T1']['reach0.5'][idx], T['T2']['reach0.5'][idx]; qa = 0.5 * (q1 + q2)
    s1, s2 = (T['T1']['outcome'][idx] == 'success').astype(float), (T['T2']['outcome'][idx] == 'success').astype(float)
    ga1, ga2 = T['T1']['goal_area'][idx], T['T2']['goal_area'][idx]
    for cname in params:
      sc = scores[cname]['mean'][ii]
      row[cname] = {'corr_q_T1': corr(sc, q1), 'corr_q_T2': corr(sc, q2), 'corr_q_avg': corr(sc, qa),
                    'spearman_q_T1': spearman(sc, q1), 'spearman_q_T2': spearman(sc, q2), 'spearman_q_avg': spearman(sc, qa),
                    'corr_success_T1': corr(sc, s1), 'corr_success_T2': corr(sc, s2), 'corr_goalarea_T1': corr(sc, ga1), 'corr_goalarea_T2': corr(sc, ga2),
                    'score_mean_detour_minus_shortcut': (float(sc[det[idx]].mean() - sc[~det[idx]].mean()) if det[idx].sum() >= 20 and (~det[idx]).sum() >= 20 else None)}
    row['_data'] = {'corr_q_T1_T2': corr(q1, q2), 'corr_success_T1_T2': corr(s1, s2)}
    res[s] = row
  return res


def region_readout(N, params, anchors, masks, T, sample_seed=13, n=1024, n_ref=4096):
  """Importance-corrected region probabilities per critic (user's correction after 6b03cdd): the NCE logit is a density ratio against the
  critic's TRAINING negative marginal, so the reference goal set must be drawn from that marginal -- anchors by weight over ALL anchors,
  the goal by the law from the critic's own table(s) (T1 for CF, T2 for CF2, the mixture for MF).  With G_ref ~ p_neg, the self-normalised
  estimate p(g in region | s, a) = sum_{g in region} exp f(s, a, g) / sum_g exp f(s, a, g) is consistent, and comparable to the true masses of
  the critic's own table(s).  Also the per-anchor correlation of the predicted goal-area mass with the true one under each table."""
  rng = np.random.default_rng(sample_seed)
  srcs = {'T1': MP.BranchFutures(anchors, TABLES['T1']), 'T2': MP.BranchFutures(anchors, TABLES['T2'])}
  cdf = np.cumsum(anchors.weight / anchors.weight.sum()); cdf[-1] = 1.0

  def ref_set(tables):
    ks = np.minimum(np.searchsorted(cdf, rng.random(n_ref), side='right'), anchors.n - 1)
    which = rng.integers(0, len(tables), size=n_ref); G = np.zeros((n_ref, 2), np.float32)
    for i, (k, w) in enumerate(zip(ks, which)):
      f = srcs[tables[w]]; nf = int(f.lengths[k] - 1); u = rng.random()
      mm = int(np.clip(np.ceil(np.log1p(-u * (1.0 - GAMMA ** nf)) / np.log(GAMMA)), 1, nf)); G[i] = f.goal_at(int(k), mm)
    return G, region_of(G.astype(np.float64))
  refs = {'CF': ref_set(('T1',)), 'CF2': ref_set(('T2',)), 'MF': ref_set(('T1', 'T2'))}
  res = {'reference': 'goal set of 4096 drawn from the critic family own training goal marginal (anchor by weight, table(s) of the family, m by the law)',
         'reference_region_share': {fam: {REGIONS[r]: float((GR == r).mean()) for r in range(len(REGIONS)) if (GR == r).any()} for fam, (G, GR) in refs.items()}}
  for s, m in masks.items():
    idx = np.nonzero(m)[0]
    if len(idx) < 100:
      continue
    idx = idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))
    obs = anchors.obs31[idx].astype(np.float32); act = anchors.action[idx].astype(np.float32)
    row = {'n': int(len(idx)), 'true_mass': {reg: {'T1': float(T['T1'][key][idx].mean()), 'T2': float(T['T2'][key][idx].mean()), 'avg': float(0.5 * (T['T1'][key][idx] + T['T2'][key][idx]).mean())} for reg, key in (('goal_area', 'goal_area'), ('far', 'far'))}}
    for cname, qp in params.items():
      fam = cname.split('_')[0]; G, GR = refs[fam]
      obs_g = np.concatenate([np.zeros((len(G), STATE_DIM), np.float32), G], axis=1)
      f = N.logits(qp, obs, act, obs_g).mean(-1)                               # [n, n_ref]
      f = f - f.max(axis=1, keepdims=True); p = np.exp(f); p /= p.sum(axis=1, keepdims=True)
      pm = {REGIONS[r]: float(p[:, GR == r].sum(axis=1).mean()) for r in range(len(REGIONS)) if (GR == r).any()}
      pga = p[:, GR == REGIONS.index('goal_area')].sum(axis=1); pfar = p[:, np.isin(GR, FAR)].sum(axis=1)
      own = {'CF': 'T1', 'CF2': 'T2', 'MF': 'avg'}[fam]
      true_ga = {'T1': T['T1']['goal_area'][idx], 'T2': T['T2']['goal_area'][idx], 'avg': 0.5 * (T['T1']['goal_area'][idx] + T['T2']['goal_area'][idx])}
      row[cname] = {'pred_region_mass': pm, 'own_reference': own, 'pred_goalarea_mean': float(pga.mean()), 'true_goalarea_mean_own': float(true_ga[own].mean()),
                    'corr_pred_goalarea_vs_T1': corr(pga, true_ga['T1']), 'corr_pred_goalarea_vs_T2': corr(pga, true_ga['T2']), 'corr_pred_goalarea_vs_avg': corr(pga, true_ga['avg']),
                    'corr_pred_far_vs_T1': corr(pfar, T['T1']['far'][idx]), 'corr_pred_far_vs_T2': corr(pfar, T['T2']['far'][idx])}
    res[s] = row
  return res


# ------------------------------------------------------------------ layer 3
def layer_actor(N, params, pols, anchors, obs, lengths, masks, det, sample_seed=17, per_stratum=384, n_relabel=2, n_samples=64):
  """Same state, same goal: fixed candidate actions under each critic, and each policy's full actor objective under each critic."""
  import jax
  rng = np.random.default_rng(sample_seed)
  strata = ('reset', 'start_early', 'west_entry', 'pre_zone1_early', 'pre_mouth', 'zone1')
  res = {}
  key = jax.random.PRNGKey(sample_seed)
  for s in strata:
    idx = np.nonzero(masks[s])[0]
    if len(idx) < 30:
      continue
    idx = idx if len(idx) <= per_stratum else np.sort(rng.choice(idx, per_stratum, replace=False))
    # goal sets: the task goal; relabelled goals from the recorded law (n_relabel per anchor)
    goal_sets = {'task': anchors.goal_xy[idx].astype(np.float32)}
    for r in range(n_relabel):
      g = np.zeros((len(idx), 2), np.float32)
      for i, k in enumerate(idx):
        e, t = int(anchors.episode[k]), int(anchors.t[k]); L = int(lengths[e]); nf = max(L - 1 - t, 1)
        u = rng.random(); mm = int(np.clip(np.ceil(np.log1p(-u * (1.0 - GAMMA ** nf)) / np.log(GAMMA)), 1, nf))
        g[i] = obs[e, min(t + mm, L - 1), :2]
      goal_sets[f'relabel{r}'] = g
    row = {'n': int(len(idx)), 'detour_share': float(det[idx].mean())}
    for gname, G in goal_sets.items():
      o31 = np.concatenate([anchors.state[idx].astype(np.float32), G], axis=1)
      a_log = anchors.action[idx].astype(np.float32)
      cands = {'logged': a_log}
      dists = {}
      for pname, pp in pols.items():
        dp = N.policy(pp, o31); dists[pname] = dp; cands[pname] = N.mode(pp, o31)
      out = {'relabel_goal_far_share': float(np.isin(region_of(G.astype(np.float64)), FAR).mean()) if gname != 'task' else None}
      for cname, qp in params.items():
        # fixed candidates: twin-min logit at (s, a_cand, g)
        qc = {}
        for aname, a in cands.items():
          f = N.diag_logits(qp, o31, a); qc[aname] = f.min(-1)
        names = list(qc); Qm = np.stack([qc[a] for a in names], axis=1)
        best = np.array(names)[Qm.argmax(axis=1)]
        cres = {'candidate_mean_Qmin': {a: float(v.mean()) for a, v in qc.items()}, 'argmax_share': {a: float((best == a).mean()) for a in names},
                'margin_CFmode_minus_startmode_mean': float((qc['pi_CF'] - qc['pi_start']).mean()), 'share_CFmode_gt_startmode': float((qc['pi_CF'] > qc['pi_start']).mean()),
                'margin_MFmode_minus_startmode_mean': float((qc['pi_MF'] - qc['pi_start']).mean()), 'margin_CFmode_minus_MFmode_mean': float((qc['pi_CF'] - qc['pi_MF']).mean()),
                'share_CFmode_gt_MFmode': float((qc['pi_CF'] > qc['pi_MF']).mean())}
        # the full actor objective of each policy under this critic: (1 - bc) * E_{a~pi}[-Qmin] + bc * (-log pi(a_logged))
        obj = {}
        for pname, dp in dists.items():
          qs = []
          for j in range(n_samples):
            key, sub = jax.random.split(key)
            a = N.sample(dp, sub); qs.append(N.diag_logits(qp, o31, a).min(-1))
          qterm = -np.mean(qs, axis=0); nll = -N.log_prob(dp, a_log)
          obj[pname] = {'q_term_mean': float(qterm.mean()), 'bc_nll_mean': float(nll.mean()), 'objective_mean': float(((1 - BC) * qterm + BC * nll).mean())}
        cres['policy_objective'] = obj
        cres['objective_pi_MF_minus_pi_CF'] = {k: obj['pi_MF'][k] - obj['pi_CF'][k] for k in ('q_term_mean', 'bc_nll_mean', 'objective_mean')}
        cres['share_states_objective_prefers_pi_CF_over_pi_MF'] = None
        out[cname] = cres
      row[gname] = out
    res[s] = row
  return res


# ------------------------------------------------------------------ report
def write_report(data, critic, actor, fams):
  L = ['# The pairing check: data / critic / actor layers on the two oracle tables (no training)', '']
  L += ['## Layer 1 -- the data (anchor-weighted masses of the critic\'s positive law; T1 = sealed draw, T2 = draw 2)', '',
        '| stratum | n | detour share | agreement | reach0.5 T1 / T2 / avg | corr(T1,T2) reach0.5 | goal_area T1 / T2 | far T1 / T2 | death_frame T1 / T2 | detour signal reach0.5 T1 / T2 / avg | detour signal far T1 / T2 | actor relabelled: reach0.5 / far (detour rows, shortcut rows) |',
        '|---|---:|---:|---:|---|---:|---|---|---|---|---|---|']
  for s, r in data['by_stratum'].items():
    q = r['reach0.5']; ds = q.get('detour_signal'); dsf = r['far'].get('detour_signal'); ar = q.get('actor_relabelled_goal_mass', {}); arf = r['far'].get('actor_relabelled_goal_mass', {})
    L.append(f"| {s} | {r['n']} | {data['strata_detour_share'][s]:.2f} | {r['outcome_agreement']:.3f} | {q['mean_T1']:.3f} / {q['mean_T2']:.3f} / {q['mean_avg']:.3f} | {q['corr_T1_T2']} | {r['goal_area']['mean_T1']:.3f} / {r['goal_area']['mean_T2']:.3f} | {r['far']['mean_T1']:.3f} / {r['far']['mean_T2']:.3f} | {r['death_frame']['mean_T1']:.3f} / {r['death_frame']['mean_T2']:.3f} | "
             + (f"{ds['T1']:+.3f} / {ds['T2']:+.3f} / {ds['avg']:+.3f}" if ds else '') + ' | ' + (f"{dsf['T1']:+.3f} / {dsf['T2']:+.3f}" if dsf else '') + ' | '
             + (f"{ar.get('mean', 0):.3f} / {arf.get('mean', 0):.3f} ({(ar.get('detour_rows') or 0):.3f}, {(ar.get('shortcut_rows') or 0):.3f}; far {(arf.get('detour_rows') or 0):.3f}, {(arf.get('shortcut_rows') or 0):.3f})") + ' |')
  L += ['', '## Layer 2a -- NCE fit on identical batches (loss / categorical accuracy), seed mean over the five critics of a family', '', '| critic family | positives T1 | positives T2 | positives mix |', '|---|---|---|---|']
  fit = critic['nce_fit']
  for fam in fams:
    cells = []
    for src in ('T1', 'T2', 'mix'):
      ls = [fit[f'{fam}_s{s}'][src]['nce_loss'] for s in SEEDS]; ac = [fit[f'{fam}_s{s}'][src]['cat_acc'] for s in SEEDS]
      cells.append(f'{np.mean(ls):.4f} / {np.mean(ac):.3f}')
    L.append(f'| {fam} | ' + ' | '.join(cells) + ' |')
  L += ['', '## Layer 2b -- the pairing test: corr(critic logit at (s, a, task goal), realised reach0.5 mass) per stratum; seed mean [min .. max]', '',
        '| stratum | data corr(T1,T2) | ' + ' | '.join(f'{fam}: vs T1 / vs T2 / vs avg' for fam in fams) + ' |', '|---|---:|' + '---|' * len(fams)]
  pt = critic['pairing']
  for s, r in pt.items():
    cells = []
    for fam in fams:
      v = {k: [r[f'{fam}_s{sd}'][k] for sd in SEEDS if r[f'{fam}_s{sd}'][k] is not None] for k in ('corr_q_T1', 'corr_q_T2', 'corr_q_avg')}
      cells.append(' / '.join((f"{np.mean(v[k]):.3f} [{min(v[k]):.2f}..{max(v[k]):.2f}]" if v[k] else 'n/a') for k in ('corr_q_T1', 'corr_q_T2', 'corr_q_avg')))
    L.append(f"| {s} | {r['_data']['corr_q_T1_T2']} | " + ' | '.join(cells) + ' |')
  L += ['', '## Layer 2c -- importance-corrected region probabilities (reference goal set = the critic family own training goal marginal): predicted goal-area / far mass vs the true masses; seed mean', '',
        '| stratum | true goal_area T1 / T2 / avg | true far T1 / T2 | ' + ' | '.join(f'{fam}: pred goal_area, pred far; corr(pred goal_area, T1 / T2 / avg)' for fam in fams) + ' |', '|---|---|---|' + '---|' * len(fams)]
  rr = critic['region']
  L.insert(len(L) - 2, f"Reference goal sets: {rr['reference']}; reference region shares {json.dumps({f: {k: round(v, 3) for k, v in d.items()} for f, d in rr['reference_region_share'].items()})}.")
  for s, r in rr.items():
    if s in ('reference', 'reference_region_share'):
      continue
    cells = []
    for fam in fams:
      pga = np.mean([r[f'{fam}_s{sd}']['pred_region_mass'].get('goal_area', 0.0) for sd in SEEDS]); pfar = np.mean([sum(r[f'{fam}_s{sd}']['pred_region_mass'].get(n, 0.0) for n in ('west_column', 'top_corridor', 'east_column')) for sd in SEEDS])
      cs = [np.mean([r[f'{fam}_s{sd}'][k] for sd in SEEDS if r[f'{fam}_s{sd}'][k] is not None] or [np.nan]) for k in ('corr_pred_goalarea_vs_T1', 'corr_pred_goalarea_vs_T2', 'corr_pred_goalarea_vs_avg')]
      cells.append(f'{pga:.3f}, {pfar:.3f}; ' + ' / '.join(f'{c:.3f}' for c in cs))
    L.append(f"| {s} | {r['true_mass']['goal_area']['T1']:.3f} / {r['true_mass']['goal_area']['T2']:.3f} / {r['true_mass']['goal_area']['avg']:.3f} | {r['true_mass']['far']['T1']:.3f} / {r['true_mass']['far']['T2']:.3f} | " + ' | '.join(cells) + ' |')
  L += ['', '## Layer 3 -- same state, same goal: candidate actions and the full actor objective (seed mean over the seed-matched critic / actor pairs)', '']
  for s, r in actor.items():
    L += [f"### {s} (n {r['n']}, detour-source share {r['detour_share']:.2f})", '',
          '| goal | critic | mean Qmin: logged / pi_CF / pi_CF2 / pi_MF / pi_start / pi_O | argmax share pi_CF / pi_MF / pi_start | share Q(pi_CF) > Q(pi_start) | share Q(pi_CF) > Q(pi_MF) | objective pi_MF - pi_CF: q_term / bc_nll / total |',
          '|---|---|---|---|---:|---:|---|']
    for gname in [k for k in r if k not in ('n', 'detour_share')]:
      for fam in fams:
        rows = [r[gname][f'{fam}_s{sd}'] for sd in SEEDS]
        cm = {a: np.mean([x['candidate_mean_Qmin'][a] for x in rows]) for a in rows[0]['candidate_mean_Qmin']}
        am = {a: np.mean([x['argmax_share'].get(a, 0.0) for x in rows]) for a in ('pi_CF', 'pi_MF', 'pi_start')}
        d = {k: np.mean([x['objective_pi_MF_minus_pi_CF'][k] for x in rows]) for k in ('q_term_mean', 'bc_nll_mean', 'objective_mean')}
        L.append(f"| {gname}{'' if gname == 'task' else ' (far share ' + format(r[gname]['relabel_goal_far_share'], '.2f') + ')'} | {fam} | " + ' / '.join(f"{cm[a]:.2f}" for a in ('logged', 'pi_CF', 'pi_CF2', 'pi_MF', 'pi_start', 'pi_O'))
                 + f" | {am['pi_CF']:.2f} / {am['pi_MF']:.2f} / {am['pi_start']:.2f} | {np.mean([x['share_CFmode_gt_startmode'] for x in rows]):.2f} | {np.mean([x['share_CFmode_gt_MFmode'] for x in rows]):.2f} | {d['q_term_mean']:+.3f} / {d['bc_nll_mean']:+.3f} / {d['objective_mean']:+.3f} |")
    L.append('')
  L += ['Reading conventions: a critic\'s logits are comparable only within that critic; region read-outs are normalised per critic over the same goal set; the actor objective is crl.losses.actor_loss with the twin-min, 64 samples, bc 0.05 -- lower is better, so a NEGATIVE "pi_MF - pi_CF" total means the MF policy scores better than the CF policy under that critic at that goal.', '']
  return '\n'.join(L)


def run(args):
  OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz'); obs, act, lengths, _ = MP.load_dataset()
  masks, reg = strata_masks(anchors); det = source_route(anchors, obs, lengths)
  print('strata', {s: int(m.sum()) for s, m in masks.items()}, flush=True)
  data, T, R = layer_data(anchors, obs, lengths, masks, det)
  MP.write_json(OUT / 'data.json', data); print(f'layer 1 done {time.time() - t0:.0f}s', flush=True)
  N = Nets()
  params = {}
  for fam in FAMILIES:
    for s in SEEDS:
      params[f'{fam}_s{s}'] = N.load(ckpt_of(fam, s))[0]
  # the seed-matched policies for layer 3 are loaded per seed below; here the layout check
  o = anchors.obs31[:256].astype(np.float32); a = anchors.action[:256].astype(np.float32)
  print('logit layout max diff', check_logits_match(N, params['CF_s0'], o, a), flush=True)
  critic = {'nce_fit': nce_fit(N, params, anchors, n_batches=args.nce_batches), 'pairing': pairing_test(N, params, anchors, masks, T, det), 'region': region_readout(N, params, anchors, masks, T)}
  MP.write_json(OUT / 'critic.json', critic); print(f'layer 2 done {time.time() - t0:.0f}s', flush=True)
  # layer 3: seed-matched (critic, actor) pairs; the candidate policies of the same seed + start + O
  _, pp_start = N.load(MP.START_CKPT)
  actor_all = {}
  for s in SEEDS:
    pols_s = {'pi_CF': N.load(ckpt_of('CF', s))[1], 'pi_CF2': N.load(ckpt_of('CF2', s))[1], 'pi_MF': N.load(ckpt_of('MF', s))[1], 'pi_start': pp_start, 'pi_O': N.load(ckpt_of('O', s))[1]}
    params_s = {f'{fam}_s{s}': params[f'{fam}_s{s}'] for fam in FAMILIES}
    res_s = layer_actor(N, params_s, pols_s, anchors, obs, lengths, masks, det, sample_seed=17 + s, per_stratum=args.actor_per_stratum)
    for st, row in res_s.items():
      actor_all.setdefault(st, {'n': row['n'], 'detour_share': row['detour_share']})
      for gname, out in row.items():
        if gname in ('n', 'detour_share'):
          continue
        actor_all[st].setdefault(gname, {'relabel_goal_far_share': out['relabel_goal_far_share']})
        for cname in params_s:
          actor_all[st][gname][cname] = out[cname]
    print(f'layer 3 seed {s} done {time.time() - t0:.0f}s', flush=True)
  MP.write_json(OUT / 'actor.json', actor_all)
  (OUT / 'REPORT.md').write_text(write_report(data, critic, actor_all, FAMILIES), encoding='utf-8')
  print(f'all done {time.time() - t0:.0f}s', flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=('run',)); ap.add_argument('--nce-batches', type=int, default=30); ap.add_argument('--actor-per-stratum', type=int, default=384)
  run(ap.parse_args(argv)); return 0


if __name__ == '__main__':
  sys.exit(main())
