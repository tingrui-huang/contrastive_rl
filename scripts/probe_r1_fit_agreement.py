"""Train-set counterpart of the round-1 gate: on the replay's OWN fit keys
(dense anchors: 3 candidates x 2 draws), does the critic order the
candidates the way their two-draw differences do?  Separates 'did not
learn the per-key differences' from 'learned them but they do not carry
to held-out states'.  usage: probe_r1_fit_agreement.py <replay.npz> <cont ckpt> <critic ckpt ...>"""
import os, sys, json
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
import build_v6_policy_replay as R
from crl import checkpoint
replay, cont = sys.argv[1], sys.argv[2]
with np.load(replay, allow_pickle=False) as d:
  o0 = d['obs'][:, 0]; a0 = d['act'][:, 0]
  aid = d['audit_anchor_id']; cand = d['audit_cand'].astype(str); draw = d['audit_draw']; sset = d['audit_kind'].astype(str)
  pg = d['audit_p_goal']
keys = {}
for i in range(len(pg)):
  k = (int(aid[i]), cand[i])
  e = keys.setdefault(k, {'obs0': o0[i], 'act0': a0[i], 'set': sset[i], 'p': {}})
  e['p'][int(draw[i])] = float(pg[i])
bundle = R.policy_bundle(cont)
scorers = {}
rng = np.random.default_rng(0)
scorers['random'] = lambda o, a: rng.standard_normal(len(o))
for ck in sys.argv[3:]:
  p = R.Path(ck); label = f'{p.parent.parent.name}/{p.parent.name}/{p.stem}'
  scorers[label] = R.critic_fn(bundle['nets'], checkpoint.load_checkpoint(p)[1].q_params)
ks, scores = R.score_table(keys, scorers)
sets = list(R.DENSE_SETS) + ['general']
# two draws per dense key: validated = sign(draw 0 ratio) == sign(draw 1 ratio), both beyond thr
m = R.gate_metrics(keys, ks, scores, sets, thr=0.3, halves=((0,), (1,)))
print('| scorer | set | validated pairs | agreement | samples-only (n) | weighted | pick gain (anchors) |')
print('|---|---|---:|---:|---:|---:|---:|')
f_ = lambda v: ('-' if v is None else f'{v:.2f}')
for name in scorers:
  for s in sets + ['dense_all']:
    x = m[name][s]
    print(f'| {name} | {s} | {x["n_validated_pairs"]} | **{f_(x["validated_agreement"])}** | {f_(x["validated_agreement_samples_only"])} ({x["n_validated_samples_only"]}) | {f_(x["weighted_agreement"])} | {f_(x["pick_gain"])} ({x["n_anchors_with_spread"]}) |')
