"""Does the policy-width margin on the replay's jitter goal frames predict the actor family?

audit_f4_goal_frames.py found that the two RIGHT-family critics differ from
the DOWN-family critics only in the width-0.75 margin Qbar(DOWN loc) -
Qbar(RIGHT loc) on the fork -> (8,3) rows whose goal frame jitters inside the
goal cell (83% of that group's mass); the point margin q(DOWN) - q(RIGHT) and
the 16-root probe do not separate them.  This script computes that margin for
every critic of Steps 9d-9f and prints it next to the family its fresh
balanced-BC actors fell into (DOWN / CORNER / RIGHT, from
policy_objective_audit/two_stage_actor_modes.json and notes Step 9f), so the
claim is tested on the ten critics the finding was not derived from.

Output: outputs/pointmaze_balanced_bc_joint_v1/right_family_audit/jitter_margin_all_critics.{md,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_f4_right_family import FIX, J, OUT, WIDTH, GOAL   # noqa: E402
from audit_f4_goal_frames import frame_type, DOWN_LOC, RIGHT_LOC   # noqa: E402

SEALED = ROOT / 'outputs' / 'pointmaze_ett_query_coverage_20260915_v1'
# family = fork mode of the three fresh balanced-BC actors under the task goal (Step 9f);
# mode lower-route rates on the new seeds in brackets
CRITICS = [
    ('D1 (4080)', FIX / 'seeds' / 'seed_1' / 'crl' / 'D' / 'final.pkl', 'DOWN', '1.00 x3'),
    ('joint30k s0', J / 'joint' / 'b30k' / 'seed_0' / 'balanced' / 'final.pkl', 'DOWN', '1.00 x3'),
    ('D0', FIX / 'seeds' / 'seed_0' / 'crl' / 'D' / 'final.pkl', 'CORNER x0.67', '1.00 x3'),
    ('joint30k s2', J / 'joint' / 'b30k' / 'seed_2' / 'balanced' / 'final.pkl', 'CORNER x0.85', '0.88/1.00/0.00'),
    ('D2', FIX / 'seeds' / 'seed_2' / 'crl' / 'D' / 'final.pkl', 'CORNER x0.95', '0.81/0.01/0.88'),
    ('O seed 0', J / 'control_O' / 'seed_0' / 'critic' / 'final.pkl', 'CORNER x1.00', '0.76/0.55/0.63'),
    ('joint30k s1', J / 'joint' / 'b30k' / 'seed_1' / 'balanced' / 'final.pkl', 'RIGHT', '0.00 x3'),
    ('joint300k s0', J / 'joint' / 'b300k' / 'seed_0' / 'balanced' / 'final.pkl', 'RIGHT', '0.00 x3'),
    ('O seed 1', J / 'control_O' / 'seed_1' / 'critic' / 'final.pkl', 'RIGHT', '0.00 x3'),
    ('O seed 2', J / 'control_O' / 'seed_2' / 'critic' / 'final.pkl', 'RIGHT', '0.00 x3'),
    ('C seed 2', SEALED / 'cross_seeds' / 'seed_2' / 'crl' / 'C' / 'final.pkl', 'RIGHT', '0.00 x3'),
    ('C seed 0', SEALED / 'crl' / 'C' / 'final.pkl', 'RIGHT (attribution 9a)', '0.22/0.00/0.26'),
]
O_DATASET = (ROOT / 'artifacts' / 'f4_p30_server_30076' / 'results' / 'datasets'
             / 'swamp_windy_f4_merged_s0.npz')


def main():
  import jax
  import jax.numpy as jnp
  from crl import checkpoint, networks
  from crl.bc_balanced import GroupBalancedBCSampler
  nets = networks.make_networks(
      8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
      use_image_obs=False, use_layer_norm=False, obs_scale=None, log_prob_mode='acme')
  rng = np.random.default_rng(777)
  rows = {}
  for tag, path in (('D', FIX / 'replay_D.npz'), ('O', O_DATASET)):
    with np.load(path, allow_pickle=False) as d:
      obs, act = d['obs'], d['act']
      lengths = d['lengths'] if 'lengths' in d.files else np.full(len(obs), obs.shape[1], np.int64)
    smp = GroupBalancedBCSampler(obs, act, lengths, 0.95, 8, cap=0.25, seed=777)
    w = smp.w_original
    s_cell = np.floor(obs[smp.tr, smp.ii, :2]).astype(np.int64)
    g_cell = np.floor(obs[smp.tr, smp.jj, :2]).astype(np.int64)
    f83 = (s_cell[:, 0] == 1) & (s_cell[:, 1] == 3) & (g_cell[:, 0] == 8) & (g_cell[:, 1] == 3)
    ww = np.where(f83, w, 0.0)
    cdf = np.cumsum(ww / ww.sum())
    pos = np.minimum(np.searchsorted(cdf, rng.random(5000), side='right'), len(cdf) - 1)
    tr, ii, jj = smp.tr[pos], smp.ii[pos].astype(np.int64), smp.jj[pos].astype(np.int64)
    S, G = obs[tr, ii, :8].astype(np.float32), obs[tr, jj, :8].astype(np.float32)
    gtype, _ = frame_type(G)
    jit = (gtype == 'jitter') | (gtype == 'parked')
    rows[tag] = np.concatenate([S, G], 1)[jit]
    print(f'{tag}: {jit.sum()} jitter rows of 5000 fork->(8,3) rows', flush=True)
  roots = np.load(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
                  / 'root_selection.npz')['state'].astype(np.float32)
  root_obs = np.concatenate([roots, np.broadcast_to(GOAL, roots.shape)], 1).astype(np.float32)
  eps = jnp.asarray(rng.standard_normal((256, 2)).astype(np.float32))

  @jax.jit
  def q_point(qp, o, a):
    phi, psi = nets.representation_network.apply(qp, o, a)
    return jnp.sum(phi * psi, axis=1)[:, 0]

  @jax.jit
  def qbar(qp, o, loc):
    a = jnp.tanh(loc[None, None, :] + WIDTH * eps[None])
    a = jnp.broadcast_to(a, (o.shape[0],) + a.shape[1:])
    o_rep = jnp.repeat(o[:, None, :], eps.shape[0], 1).reshape(-1, o.shape[1])
    phi, psi = nets.representation_network.apply(qp, o_rep, a.reshape(-1, 2))
    return jnp.mean(jnp.sum(phi * psi, axis=1)[:, 0].reshape(o.shape[0], -1), axis=1)

  def margins(qp, o):
    o = jnp.asarray(o)
    d_ = jnp.tile(jnp.array([[0.0, -1.0]], jnp.float32), (o.shape[0], 1))
    r_ = jnp.tile(jnp.array([[1.0, 0.0]], jnp.float32), (o.shape[0], 1))
    pm, wm = [], []
    for k in range(0, o.shape[0], 1024):
      pm.append(np.asarray(q_point(qp, o[k:k + 1024], d_[k:k + 1024]) - q_point(qp, o[k:k + 1024], r_[k:k + 1024])))
      wm.append(np.asarray(qbar(qp, o[k:k + 1024], jnp.asarray(DOWN_LOC)) - qbar(qp, o[k:k + 1024], jnp.asarray(RIGHT_LOC))))
    return float(np.concatenate(pm).mean()), float(np.concatenate(wm).mean())

  out = []
  L = ['# Width margin on the jitter goal frames versus the actor family', '',
       'Rows: fork -> (8,3) rows of the replay the critic was trained on (D replay for the D / '
       'joint / C critics, the original 0.05-rung dataset for the O critics), goal frame jittering '
       f'inside the goal cell.  Margins: point q(DOWN) - q(RIGHT); width = Qbar(DOWN loc) - Qbar(RIGHT loc) at width {WIDTH}.', '',
       '| critic | family of its fresh balanced-BC actors | mode lower (new seeds) | 16 roots x canonical: point / width | jitter frames: point / width |',
       '|---|---|---|---:|---:|']
  for name, path, fam, lower in CRITICS:
    if not path.exists():
      L.append(f'| {name} | {fam} | {lower} | missing | missing |')
      continue
    qp = checkpoint.load_checkpoint(path)[1].q_params
    o = rows['O' if name.startswith('O ') else 'D']
    rp, rw = margins(qp, root_obs)
    jp, jw = margins(qp, o)
    out.append({'critic': name, 'family': fam, 'lower': lower, 'root_point': rp, 'root_width': rw,
                'jitter_point': jp, 'jitter_width': jw})
    L.append(f'| {name} | {fam} | {lower} | {rp:+.2f} / {rw:+.2f} | {jp:+.3f} / **{jw:+.3f}** |')
    print(L[-1], flush=True)
  (OUT / 'jitter_margin_all_critics.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  (OUT / 'jitter_margin_all_critics.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
  print('\n'.join(L))
  return 0


if __name__ == '__main__':
  sys.exit(main())
