"""Shared frozen-integrity checks + controller loading for the Stage-3A pilot.

This module NEVER modifies any frozen process. It only (a) recomputes
checksums, (b) compares live code constants to the freeze manifest, and
(c) builds the frozen walker + base controllers exactly as the gate/teacher
did. Both the collector and the auditor import from here so the two agree
byte-for-byte on what "the frozen pipeline" means.
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
  sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

FREEZE_PATH = 'artifacts/freeze_signature/litter_freeze.json'


def sha256_file(path, chunk=1 << 20):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for b in iter(lambda: f.read(chunk), b''):
      h.update(b)
  return h.hexdigest()


def git_commit():
  try:
    return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                          text=True).stdout.strip()
  except Exception:  # pylint: disable=broad-except
    return 'unknown'


def _approx(a, b, tol=1e-9):
  a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
  return a.shape == b.shape and bool(np.all(np.abs(a - b) <= tol))


def check_frozen_integrity(manifest_path=FREEZE_PATH):
  """A1 core. Returns (hard_ok, discrepancies, info).

  hard_ok is False only for a SUBSTANTIVE drift (checksum / geometry /
  collapse / handoff / lane / v_fast / layout / seed reuse). Pure manifest
  documentation mismatches are returned in `discrepancies` with
  severity='doc' and do NOT flip hard_ok.
  """
  from crl import d4rl_ant as D
  from crl import probe as P
  import walker_gate as WG

  man = json.load(open(manifest_path))
  disc = []
  hard_ok = True

  def hard(name, got, want, tol=1e-9):
    nonlocal hard_ok
    ok = _approx(got, want, tol) if isinstance(want, (int, float, list, tuple)) \
        else (got == want)
    if not ok:
      disc.append({'field': name, 'severity': 'hard', 'code': got,
                   'manifest': want})
      hard_ok = False
    return ok

  # (1) checkpoint checksums
  wpath = man['walker']['path']
  bpath = man['base_policy']['path']
  wsha, bsha = sha256_file(wpath), sha256_file(bpath)
  hard('walker.sha256', wsha, man['walker']['sha256'])
  hard('base_policy.sha256', bsha, man['base_policy']['sha256'])

  # (2) env geometry / collapse / layout constants (code vs manifest)
  env = man['env']
  hard('LITTER_ZONE_X', list(D.LITTER_ZONE_X), env['zone_x'])
  hard('LITTER_PILE_Y', list(D.LITTER_PILE_Y), env['pile_y'])
  hard('LITTER_PILE_HEIGHT', D.LITTER_PILE_HEIGHT, env['pile_height'])
  hard('LITTER_SKIRT_Y', list(D.LITTER_SKIRT_Y), env['skirt_y'])
  hard('LITTER_SKIRT_N', D.LITTER_SKIRT_N, env['skirt_n'])
  hard('LITTER_SKIRT_HALF_XY', list(D.LITTER_SKIRT_HALF_XY), env['skirt_half_xy'])
  hard('LITTER_SKIRT_H0', list(D.LITTER_SKIRT_H0), env['skirt_h0'])
  hard('LITTER_SKIRT_H1', list(D.LITTER_SKIRT_H1), env['skirt_h1'])
  hard('LITTER_REEF_X', list(D.LITTER_REEF_X), env['reef_x'])
  hard('LITTER_REEF_PAIRS', D.LITTER_REEF_PAIRS, env['reef_pairs'])
  hard('LITTER_REEF_Y', list(D.LITTER_REEF_Y), env['reef_y'])
  hard('LITTER_REEF_H', list(D.LITTER_REEF_H), env['reef_h'])
  hard('LITTER_SLICK_Y', list(D.LITTER_SLICK_Y), env['slick_y'])
  hard('LITTER_SLICK_H', D.LITTER_SLICK_H, env['slick_h'])
  hard('LITTER_SLICK_FRICTION', D.LITTER_SLICK_FRICTION, env['slick_friction'])
  hard('LITTER_LAYOUT_SEED', D.LITTER_LAYOUT_SEED, env['layout_seed'])
  hard('LITTER_HIDE_Z', D.LITTER_HIDE_Z, env['inactive_side_hide_z'])
  hard('LITTER_COLLAPSE_FORCE', D.LITTER_COLLAPSE_FORCE, env['collapse_force'])
  hard('LITTER_COLLAPSE_SPEED', D.LITTER_COLLAPSE_SPEED, env['collapse_speed'])

  # (3) commands / handoff / lane / v_fast
  cmd = man['commands']
  hard('WG.LANE', WG.LANE, cmd['lane'])
  hard('probe.V_FAST', P.V_FAST, cmd['v_fast'])
  hard('WG.HANDOFF_X', WG.HANDOFF_X, cmd['handoff']['handoff_x'])
  hard('unstick.window', WG.STALL_WINDOW, cmd['unstick_middle_slow_only']['window'])
  hard('unstick.min_dx', WG.STALL_MIN_DX, cmd['unstick_middle_slow_only']['min_dx'])
  hard('unstick.nudge_y', WG.NUDGE_Y, cmd['unstick_middle_slow_only']['nudge_y'])
  hard('unstick.nudge_steps', WG.NUDGE_STEPS,
       cmd['unstick_middle_slow_only']['nudge_steps'])

  # slow_v: the frozen QUALIFIED teacher's blind speed is WG.SLOW_V (code),
  # which equals probe.V_SLOW. The manifest records commands.slow_v for the
  # GATE middle_slow validation arm (0.8, --slow-v). These are two distinct
  # roles; flag any divergence as a DOC discrepancy (data generation uses the
  # code value, never the manifest value).
  teacher_blind_v = WG.SLOW_V
  if not _approx(teacher_blind_v, cmd['slow_v']):
    disc.append({
        'field': 'commands.slow_v', 'severity': 'doc',
        'code_teacher_blind_v': teacher_blind_v,
        'manifest_slow_v': cmd['slow_v'],
        'note': ('manifest slow_v documents the GATE middle_slow arm '
                 '(--slow-v 0.8); the frozen QUALIFIED teacher blind policy '
                 'uses WG.SLOW_V=probe.V_SLOW=%.3g. Data generation uses the '
                 'code value. Manifest should split gate_middle_slow_v from '
                 'teacher_blind_v before full collection.' % teacher_blind_v)})

  # (4) u_side semantics sanity (code truth: side_u={'pos':1,'neg':0})
  info = {'walker_sha256': wsha, 'base_sha256': bsha,
          'teacher_blind_v': teacher_blind_v, 'v_fast': P.V_FAST,
          'lane': WG.LANE, 'handoff_x': WG.HANDOFF_X,
          'collapse_force': D.LITTER_COLLAPSE_FORCE,
          'collapse_speed': D.LITTER_COLLAPSE_SPEED,
          'manifest_git_frozen_code_commit': man.get('frozen_code_commit'),
          'current_git_commit': git_commit()}
  return hard_ok, disc, info


def seed_reuse(consumed, *seed_lists):
  used = set()
  for lst in seed_lists:
    used |= set(int(s) for s in lst)
  clash = sorted(used & set(int(s) for s in consumed))
  return clash


def load_controllers(walker_path, base_path):
  """Frozen walker + base policy, exactly as gate/teacher built them."""
  import jax
  import jax.numpy as jnp
  from crl import networks as networks_mod
  from crl import checkpoint as ckpt_mod
  from crl import probe as P
  from crl import envs as envs_mod
  from verify_offline_d4rl import build_offline_cfg

  cfg = build_offline_cfg()
  envs_mod.make_env('offline_ant_umaze', cfg, seed=1)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  base_step, st = ckpt_mod.load_checkpoint(base_path)
  params = st.policy_params

  @jax.jit
  def base_act(o):
    return jnp.tanh(nets.policy_network.apply(params, o).loc)

  wparams, wmeta = P.load_residual(walker_path)
  walker = P.WalkerController(wparams)
  return cfg, walker, base_act, int(base_step), wmeta
