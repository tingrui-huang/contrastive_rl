"""D1 vs D2: what the actor actually optimizes at the fork.

The actor minimizes, per state,

    L(loc, scale) = (1 - bc) * E_{a ~ tanh N(loc, scale)}[ -q(s, a) ]
                    + bc * E_{a_data}[ -log pi(a_data | loc, scale) ]

so the relevant critic quantity is the SMOOTHED score Qbar(loc) = E[q(s, a)]
under the policy's own sampling width, not the location of the argmax of q.
For the repaired critics D1 (argmax = DOWN on every root, no fresh actor
detours) and D2 (fresh actors detour), at each of the 16 held-out roots:

  * raw q over the action square and its argmax (as before);
  * Qbar over a grid of pre-tanh locations, at the scale the fresh actor
    actually has at that root (and at a sharp reference scale), its argmax
    and its gradient at the actor's loc;
  * the BC term over the same loc grid, using the D-replay actions taken
    within 0.15 of the root (with the tanh-normal log-prob exactly as in
    crl/networks.py, boundary clipping included), its argmin and gradient;
  * the total objective, its argmin, and the y-gradient balance between the
    two terms at the actor's loc -- i.e. why the actor rests where it does.

Outputs: outputs/pointmaze_actor_objective_D1_vs_D2/{results.json,REPORT.md,
landscapes_root*.png}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
FIX = ROOT / 'outputs' / 'pointmaze_diagonal_coverage_fix_v1'
STEP4 = ROOT / 'outputs' / 'pointmaze_fixed_dcritic_actor_v1'
OUT = ROOT / 'outputs' / 'pointmaze_actor_objective_D1_vs_D2'
GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)
DOWN = np.array([0.0, -1.0], np.float32)
BC = 0.05
LOC_GRID = np.linspace(-6.0, 6.0, 49, dtype=np.float32)
ACT_GRID = np.linspace(-1.0, 1.0, 61, dtype=np.float32)
N_EPS = 128
SHARP_SCALE = 0.3
LOCAL_RADIUS = 0.15
CRITICS = {'D1': FIX / 'seeds' / 'seed_1' / 'crl' / 'D' / 'final.pkl',
           'D2': FIX / 'seeds' / 'seed_2' / 'crl' / 'D' / 'final.pkl'}
ACTORS = {'D1': [STEP4 / 'critic_D1' / f'actor_s{a}' / 'final.pkl' for a in range(3)],
          'D2': [STEP4 / 'critic_D2' / f'actor_s{a}' / 'final.pkl' for a in range(3)]}


def plain(v):
  if isinstance(v, dict):
    return {str(k): plain(x) for k, x in v.items()}
  if isinstance(v, (list, tuple)):
    return [plain(x) for x in v]
  if isinstance(v, np.ndarray):
    return v.tolist()
  if isinstance(v, np.generic):
    return v.item()
  return v


def make_network():
  from crl import networks
  return networks.make_networks(
      8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
      use_image_obs=False, use_layer_norm=False, obs_scale=None)


def physics_region(outcome_map, action):
  """Look up which route a grid-resolution action leads to (from Step 1)."""
  ix = int(np.clip(np.round((action[0] + 1) / 2 * 60), 0, 60))
  iy = int(np.clip(np.round((action[1] + 1) / 2 * 60), 0, 60))
  return {0: 'other', 1: 'lower', 2: 'shortcut'}[int(outcome_map[iy, ix])]


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--plot-roots', type=int, nargs='+', default=[0, 7, 14])
  args = ap.parse_args(argv)
  import jax
  import jax.numpy as jnp
  from crl import checkpoint, networks
  OUT.mkdir(parents=True, exist_ok=True)
  nets = make_network()
  with np.load(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
               / 'root_selection.npz', allow_pickle=False) as d:
    roots = d['state'].astype(np.float32)
  n = len(roots)
  obs = np.concatenate([roots, np.broadcast_to(GOAL, roots.shape)], 1).astype(np.float32)
  outcome_maps = np.load(ROOT / 'outputs' / 'pointmaze_actor_fork_diagnosis_v1'
                         / 'seed0_heldout_score_maps.npz')['outcome_maps']
  # local data actions per root from the D replay
  with np.load(FIX / 'replay_D.npz', allow_pickle=False) as d:
    r_xy = d['obs'][:, :50, :2].reshape(-1, 2)
    r_act = d['act'][:, :50, :].reshape(-1, 2).astype(np.float32)
  local_actions = []
  for i in range(n):
    sel = np.linalg.norm(r_xy - roots[i, :2], axis=1) <= LOCAL_RADIUS
    local_actions.append(r_act[sel])

  @jax.jit
  def q_of(q_params, o, a):
    phi, psi = nets.representation_network.apply(q_params, o, a)
    return jnp.sum(phi * psi, axis=1)[:, 0]

  gx, gy = np.meshgrid(ACT_GRID, ACT_GRID, indexing='xy')
  act_grid = np.stack([gx.ravel(), gy.ravel()], 1)
  lx, ly = np.meshgrid(LOC_GRID, LOC_GRID, indexing='xy')
  loc_grid = np.stack([lx.ravel(), ly.ravel()], 1)           # [G, 2]
  eps = np.asarray(jax.random.normal(jax.random.PRNGKey(0), (N_EPS, 2)),
                   np.float32)

  def qbar_grid(q_params, o_row, scale):
    """E_eps q(s, tanh(loc + scale*eps)) for every loc on the grid."""
    a = np.tanh(loc_grid[:, None, :] + scale[None, None, :] * eps[None])  # [G,E,2]
    a = a.reshape(-1, 2).astype(np.float32)
    o_rep = np.repeat(o_row[None], len(a), 0)
    q = np.asarray(q_of(q_params, jnp.asarray(o_rep), jnp.asarray(a)))
    return q.reshape(len(loc_grid), N_EPS).mean(1)

  def bc_grid(actions, scale):
    """E_{a_data}[-log pi(a | loc, scale)] on the loc grid, exact log-prob."""
    if len(actions) == 0:
      return np.full(len(loc_grid), np.nan)
    params = networks.TanhNormalParams(
        loc=jnp.asarray(loc_grid)[:, None, :],
        scale=jnp.asarray(np.broadcast_to(scale, (len(loc_grid), 1, 2))))
    lp = nets.log_prob(params, jnp.asarray(actions)[None])        # [G, N]
    return -np.asarray(lp).mean(1)

  def qbar_and_grad_at(q_params, o_row, loc, scale, key):
    """Qbar and dQbar/dloc at one loc (reparameterized, many eps)."""
    e = jax.random.normal(key, (2048, 2))
    def f(l):
      a = jnp.tanh(l[None] + scale[None] * e)
      return jnp.mean(q_of(q_params, jnp.repeat(jnp.asarray(o_row)[None], len(a), 0), a))
    val, grad = jax.value_and_grad(f)(jnp.asarray(loc))
    return float(val), np.asarray(grad)

  def bc_and_grad_at(actions, loc, scale):
    def f(l):
      params = networks.TanhNormalParams(loc=l[None], scale=jnp.asarray(scale)[None])
      return -jnp.mean(nets.log_prob(params, jnp.asarray(actions)))
    val, grad = jax.value_and_grad(f)(jnp.asarray(loc))
    return float(val), np.asarray(grad)

  results = {}
  for name, cpath in CRITICS.items():
    _, cst = checkpoint.load_checkpoint(cpath)
    q_params = cst.q_params
    actors = []
    for apath in ACTORS[name]:
      _, ast_ = checkpoint.load_checkpoint(apath)
      dist = nets.policy_network.apply(ast_.policy_params, jnp.asarray(obs))
      actors.append({'loc': np.asarray(dist.loc), 'scale': np.asarray(dist.scale)})
    per_root = []
    for i in range(n):
      o_row = obs[i]
      q_raw = np.asarray(q_of(q_params, jnp.asarray(np.repeat(o_row[None], len(act_grid), 0)),
                              jnp.asarray(act_grid)))
      k_raw = int(np.argmax(q_raw))
      # actor seed 0 as the reference policy at this root; others summarized
      loc0, scale0 = actors[0]['loc'][i], actors[0]['scale'][i]
      qb = qbar_grid(q_params, o_row, scale0)
      qb_sharp = qbar_grid(q_params, o_row, np.full(2, SHARP_SCALE, np.float32))
      bcg = bc_grid(local_actions[i], scale0)
      total = (1 - BC) * (-qb) + BC * bcg
      k_qb, k_qbs, k_bc, k_tot = (int(np.argmax(qb)), int(np.argmax(qb_sharp)),
                                  int(np.nanargmin(bcg)), int(np.nanargmin(total)))
      qb_val, qb_grad = qbar_and_grad_at(q_params, o_row, loc0, scale0,
                                         jax.random.PRNGKey(100 + i))
      bc_val, bc_grad = bc_and_grad_at(local_actions[i], loc0, scale0)
      # the actual balance at the actor's loc: total gradient in pre-tanh space
      g_critic = -(1 - BC) * qb_grad          # d/dloc of (1-bc)(-Qbar)
      g_bc = BC * bc_grad
      per_root.append({
          'root': i,
          'actor0_loc': loc0, 'actor0_mode': np.tanh(loc0), 'actor0_scale': scale0,
          'actors_mode_y': [float(np.tanh(a['loc'][i, 1])) for a in actors],
          'raw_argmax_action': act_grid[k_raw], 'raw_max': float(q_raw[k_raw]),
          'raw_argmax_region': physics_region(outcome_maps[i], act_grid[k_raw]),
          'q_down': float(np.asarray(q_of(q_params, jnp.asarray(o_row[None]), jnp.asarray(DOWN[None])))[0]),
          'qbar_argmax_loc': loc_grid[k_qb], 'qbar_argmax_mode': np.tanh(loc_grid[k_qb]),
          'qbar_max': float(qb[k_qb]),
          'qbar_argmax_region': physics_region(outcome_maps[i], np.tanh(loc_grid[k_qb])),
          'qbar_sharp_argmax_mode': np.tanh(loc_grid[k_qbs]),
          'qbar_sharp_argmax_region': physics_region(outcome_maps[i], np.tanh(loc_grid[k_qbs])),
          'qbar_at_actor': qb_val,
          'qbar_at_down_loc': float(qb[np.argmin(np.linalg.norm(loc_grid - np.array([0.0, -5.0]), axis=1))]),
          'bc_argmin_loc': loc_grid[k_bc], 'bc_argmin_mode': np.tanh(loc_grid[k_bc]),
          'bc_at_actor': bc_val, 'bc_min': float(bcg[k_bc]),
          'local_actions_n': int(len(local_actions[i])),
          'local_actions_mean': (local_actions[i].mean(0) if len(local_actions[i]) else None),
          'total_argmin_loc': loc_grid[k_tot], 'total_argmin_mode': np.tanh(loc_grid[k_tot]),
          'total_argmin_region': physics_region(outcome_maps[i], np.tanh(loc_grid[k_tot])),
          'grad_critic_term_at_actor': g_critic, 'grad_bc_term_at_actor': g_bc,
          'grad_total_at_actor': g_critic + g_bc,
          'landscapes': {'qbar': qb.reshape(len(LOC_GRID), len(LOC_GRID)),
                         'bc': bcg.reshape(len(LOC_GRID), len(LOC_GRID)),
                         'total': total.reshape(len(LOC_GRID), len(LOC_GRID)),
                         'q_raw': q_raw.reshape(len(ACT_GRID), len(ACT_GRID))},
      })
      print(f'{name} root {i}: raw argmax {act_grid[k_raw].round(2)} ({per_root[-1]["raw_argmax_region"]}) '
            f'| Qbar argmax mode {np.tanh(loc_grid[k_qb]).round(2)} ({per_root[-1]["qbar_argmax_region"]}) '
            f'| BC argmin mode {np.tanh(loc_grid[k_bc]).round(2)} | total argmin mode '
            f'{np.tanh(loc_grid[k_tot]).round(2)} ({per_root[-1]["total_argmin_region"]}) '
            f'| actor mode {np.tanh(loc0).round(2)} | grad_y critic {g_critic[1]:+.3f} bc {g_bc[1]:+.3f}',
            flush=True)
    results[name] = per_root

  # ---- summary + report
  def count(name, key, value):
    return int(sum(r[key] == value for r in results[name]))
  lines = ['# D1 vs D2: the objective the actor actually optimizes at the fork', '',
           f'L(loc) = (1-bc) * E_(a~tanh N(loc, scale))[-q(s,a)] + bc * E_(a_data)[-log pi(a_data|loc,scale)], '
           f'bc = {BC}. Qbar = the sampled-action mean score, on a pre-tanh loc grid '
           f'[-6, 6]^2 with {N_EPS} innovations, at the fresh actor\'s own scale at the root '
           f'(actor seed 0); BC = the tanh-normal NLL of the D-replay actions taken within '
           f'{LOCAL_RADIUS} of the root (thousands per root, boundary-clipped ones included); '
           'region = where the corresponding mode action physically leads (Step 1 physics map).', '',
           '| critic | raw argmax region (lower/shortcut/other) | Qbar argmax region @actor scale | '
           'Qbar argmax region @scale 0.3 | BC argmin mode (mean) | total argmin region | '
           'actor mode (mean) | fresh actors: mode y (3 seeds, root mean) |',
           '|---|---|---|---|---|---|---|---|']
  for name in CRITICS:
    rr = results[name]
    def reg(key):
      return f'{count(name, key, "lower")}/{count(name, key, "shortcut")}/{count(name, key, "other")}'
    bc_mode = np.mean([r['bc_argmin_mode'] for r in rr], 0)
    act_mode = np.mean([r['actor0_mode'] for r in rr], 0)
    ys = np.mean([r['actors_mode_y'] for r in rr], 0)
    lines.append(f'| {name} | {reg("raw_argmax_region")} | {reg("qbar_argmax_region")} | '
                 f'{reg("qbar_sharp_argmax_region")} | ({bc_mode[0]:+.2f},{bc_mode[1]:+.2f}) | '
                 f'{reg("total_argmin_region")} | ({act_mode[0]:+.2f},{act_mode[1]:+.2f}) | '
                 f'{ys[0]:+.2f} / {ys[1]:+.2f} / {ys[2]:+.2f} |')
  lines += ['', '## Values at the fresh actor\'s loc (mean over roots)', '',
            '| critic | Qbar(actor) | Qbar max | Qbar at loc (0,-5) (= DOWN) | q(DOWN) raw | BC(actor) | BC min | '
            'd/dloc_y of (1-bc)(-Qbar) | d/dloc_y of bc*BC | total d/dloc_y | d/dloc_x critic / bc |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
  for name in CRITICS:
    rr = results[name]
    m = lambda k: float(np.mean([r[k] for r in rr]))
    gc = np.mean([r['grad_critic_term_at_actor'] for r in rr], 0)
    gb = np.mean([r['grad_bc_term_at_actor'] for r in rr], 0)
    lines.append(f'| {name} | {m("qbar_at_actor"):.3f} | {m("qbar_max"):.3f} | {m("qbar_at_down_loc"):.3f} | '
                 f'{m("q_down"):.3f} | {m("bc_at_actor"):.3f} | {m("bc_min"):.3f} | {gc[1]:+.4f} | {gb[1]:+.4f} | '
                 f'{gc[1] + gb[1]:+.4f} | {gc[0]:+.4f} / {gb[0]:+.4f} |')
  lines += ['', 'Sign convention: loc_y more negative = more DOWN. A negative d/dloc_y '
            'means the term pushes loc_y down (towards the detour); the actor rests where '
            'the total is ~0 (or where tanh saturation kills the critic gradient).', '',
            '## Per root', '',
            '| critic | root | raw argmax | Qbar argmax mode (region) | BC argmin mode | '
            'total argmin mode (region) | actor mode | grad_y critic | grad_y bc |',
            '|---|---|---|---|---|---|---|---:|---:|']
  for name in CRITICS:
    for r in results[name]:
      lines.append(f'| {name} | {r["root"]} | ({r["raw_argmax_action"][0]:+.2f},{r["raw_argmax_action"][1]:+.2f}) '
                   f'| ({r["qbar_argmax_mode"][0]:+.2f},{r["qbar_argmax_mode"][1]:+.2f}) ({r["qbar_argmax_region"]}) '
                   f'| ({r["bc_argmin_mode"][0]:+.2f},{r["bc_argmin_mode"][1]:+.2f}) '
                   f'| ({r["total_argmin_mode"][0]:+.2f},{r["total_argmin_mode"][1]:+.2f}) ({r["total_argmin_region"]}) '
                   f'| ({r["actor0_mode"][0]:+.2f},{r["actor0_mode"][1]:+.2f}) '
                   f'| {r["grad_critic_term_at_actor"][1]:+.4f} | {r["grad_bc_term_at_actor"][1]:+.4f} |')
  (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
  slim = {name: [{k: v for k, v in r.items() if k != 'landscapes'} for r in rr]
          for name, rr in results.items()}
  (OUT / 'results.json').write_text(json.dumps(plain(slim), indent=2), encoding='utf-8')
  np.savez_compressed(OUT / 'landscapes.npz', loc_grid=LOC_GRID, act_grid=ACT_GRID,
                      **{f'{name}_root{r["root"]}_{k}': r['landscapes'][k]
                         for name, rr in results.items() for r in rr for k in r['landscapes']})
  print('\n'.join(lines[:12]), flush=True)

  # ---- plots for a few roots
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  for i in args.plot_roots:
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    for row, name in enumerate(CRITICS):
      r = results[name][i]
      L = r['landscapes']
      ext_a = [-1, 1, -1, 1]
      ext_l = [LOC_GRID[0], LOC_GRID[-1], LOC_GRID[0], LOC_GRID[-1]]
      panels = [('raw q(s, a) over the action square', L['q_raw'], ext_a, 'a'),
                (f'Qbar(loc) at actor scale ({r["actor0_scale"][0]:.2f},{r["actor0_scale"][1]:.2f})', L['qbar'], ext_l, 'loc'),
                ('BC term over loc (D-replay actions near root)', L['bc'], ext_l, 'loc'),
                ('total (1-bc)(-Qbar) + bc*BC over loc', L['total'], ext_l, 'loc')]
      for col, (title, Z, ext, space) in enumerate(panels):
        ax = axes[row, col]
        im = ax.imshow(Z, origin='lower', extent=ext, aspect='auto',
                       cmap='viridis' if col != 3 else 'viridis_r')
        fig.colorbar(im, ax=ax, shrink=0.8)
        if space == 'a':
          ax.scatter(*r['actor0_mode'], marker='*', s=160, c='red', edgecolors='k', label='actor mode')
          ax.scatter(*r['raw_argmax_action'], marker='x', s=90, c='white', label='raw argmax')
          ax.scatter(0, -1, marker='v', s=90, c='cyan', edgecolors='k', label='DOWN')
          ax.contour(ACT_GRID, ACT_GRID, (outcome_maps[i] == 1).astype(float), levels=[0.5], colors='cyan')
        else:
          ax.scatter(*r['actor0_loc'], marker='*', s=160, c='red', edgecolors='k', label='actor loc')
          key = ('qbar_argmax_loc', 'bc_argmin_loc', 'total_argmin_loc')[col - 1]
          ax.scatter(*r[key], marker='x', s=90, c='white', label='opt')
          ax.axhline(0, color='w', lw=0.4)
          ax.axvline(0, color='w', lw=0.4)
        ax.set_title(f'{name} root {i}: {title}', fontsize=9)
        ax.legend(fontsize=7, loc='upper left')
    fig.suptitle(f'root {i}: what the actor optimizes under critic D1 (top) and D2 (bottom); '
                 'loc panels are pre-tanh coordinates (loc_y = -5 ~ action y = -1)', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f'landscapes_root{i}.png', dpi=100)
    plt.close(fig)
  return 0


if __name__ == '__main__':
  sys.exit(main())
