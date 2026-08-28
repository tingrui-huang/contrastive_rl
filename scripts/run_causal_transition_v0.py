"""V0 launcher for the causal-transition redesign: clean CRL + failure-aware negatives.

This is the ONLY sanctioned entry point for the V0 baseline on the
``feature/causal-transition-v0`` branch. It exists because
``scripts/naive_rockfall_v2_crl.py`` has permissive defaults -- every
authoritative knob (dataset, p_active, horizon, reset-fix) falls back to an
OLDER, EASIER setting when the corresponding flag is omitted. See
``notes/CAUSAL_TRANSITION_V0.md`` for the full inventory.

(The related hazard -- the rockfall overrides being ad-hoc attributes read back
with ``getattr(config, 'rockfall_*', default)``, so a dropped or misspelled
assignment degraded silently -- is FIXED on this branch: they are now declared
fields on ``crl.config.Config`` with the identical fallback values, so they also
appear in the startup Config banner. ``assert_env_matches`` below stays as the
end-to-end guard.)

What this launcher does differently:

  * the authoritative configuration is FROZEN in module constants; there is no
    flag to point it at a different dataset, density, horizon or severity;
  * a hard provenance gate runs first (SHA-256 of the clean dataset and the
    settled bank, bank shape, bank manifest checks, explicit refusal of the
    LEGACY bank), mirroring ``scripts/run_failneg_settledbank_h800.sh``;
  * after the env is built it ASSERTS that every rockfall override actually
    landed on the env object, which is what the silent-getattr path cannot do
    for itself;
  * ``--alpha`` accepts only the two pre-registered arms, 0.1 (failure-aware)
    and 0.0 (byte-identical baseline).

Deliberately NOT wired here (deprecated for this redesign, see the V0 note):
Flow candidate generation, ``crl.static_worstcase``, ``crl.pessimistic_positive``,
``wc_positive`` / ``wc_table`` / ``wc_rho_mode`` / ``wc_p_wc``, and any
replacement of contrastive POSITIVE goals by generated worst-case states. None
of those modules exist on this branch and nothing here imports them.

The new causal transition model F_theta(s, x, x') -> s_next is NOT implemented
yet; this launcher trains the V0 baseline only.

Usage:
  python scripts/run_causal_transition_v0.py --check-only     # gate only, no jax work
  python scripts/run_causal_transition_v0.py --smoke          # gate + 5-update in-process check
  python scripts/run_causal_transition_v0.py --run            # production run (default 300k)
  python scripts/run_causal_transition_v0.py --run --alpha 0  # baseline arm
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# AUTHORITATIVE V0 CONFIGURATION -- frozen. Nothing below is settable by flag.
# ---------------------------------------------------------------------------
ENV_NAME = 'offline_ant_umaze_rockfall'

_SPLIT = 'artifacts/rockfall_v2_p30_h800_resetfix/failure_split'
CLEAN_NPZ = os.path.join(
    _SPLIT, 'antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz')
CLEAN_SHA = '6bec8a52e771569c4edc14ff0c7319df4322fe6d74e6e69a6a7074fc76be1852'

BANK_DIR = 'artifacts/settled_failure_bank_alpha01'
BANK_NPZ = os.path.join(BANK_DIR, 'failure_bank_settled.npz')
BANK_MANIFEST = os.path.join(BANK_DIR, 'bank_manifest.json')
BANK_SHA = '8c50202403317e8adc67670be716fffe5c3b3cb891017bfbe6eb070acd61d0ce'
# The pre-patch "healthy-looking frozen pose" bank. Refused explicitly so this
# launcher cannot silently reproduce the superseded experiment.
LEGACY_BANK_SHA = ('8d35b76ada59199e6ba22250a02bbdda931ff885beb08c2d8d146ec'
                   'd14b41481')
BANK_SHAPE = (16, 29)

P_ACTIVE = 0.30
HORIZON = 800
RESET_FIX = True
SEVERITY = (0.80, 0.15, 0.05)          # v2.1 local-detour eval lethality
# The training/eval env runs WITHOUT the death-settle patch. The 80 settle
# substeps belong to BANK CONSTRUCTION only (see the V0 note); leaving this at
# the env default 0 keeps V0 byte-comparable with the existing H800 runs.
DEATH_SETTLE_SUBSTEPS = 0

ALPHA_ARMS = {0.1, 0.0}                # 0.1 = failure-aware, 0.0 = baseline
DEFAULT_ALPHA = 0.1
DEFAULT_STEPS = 300_000


def _sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(1 << 20), b''):
      h.update(block)
  return h.hexdigest()


def _git_commit():
  try:
    out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=_ROOT,
                         capture_output=True, text=True, timeout=15)
    return out.stdout.strip() or '(unknown)'
  except Exception:                                  # noqa: BLE001
    return '(unknown)'


def provenance_gate(alpha, allow_cpu=False, require_backend=True):
  """Refuse to proceed unless every authoritative artifact is exactly right.

  Both artifacts are verified regardless of ``alpha`` -- the alpha=0 baseline
  arm must be reproducible against the same frozen tree as the alpha=0.1 arm.
  Only the WIRING of the bank into the config depends on alpha.
  """
  missing = [p for p in (CLEAN_NPZ, BANK_NPZ, BANK_MANIFEST)
             if not os.path.isfile(os.path.join(_ROOT, p))]
  if missing:
    raise SystemExit('ABORT: authoritative artifact(s) not found:\n  '
                     + '\n  '.join(missing))

  clean_sha = _sha256(os.path.join(_ROOT, CLEAN_NPZ))
  bank_sha = _sha256(os.path.join(_ROOT, BANK_NPZ))
  with np.load(os.path.join(_ROOT, BANK_NPZ)) as b:
    bank_shape = tuple(int(v) for v in b['goals'].shape)
  with open(os.path.join(_ROOT, BANK_MANIFEST)) as f:
    manifest = json.load(f)

  backend, devices = '(not queried)', ''
  if require_backend:
    import jax
    backend, devices = jax.default_backend(), str(jax.devices())

  print('=' * 70)
  print('CAUSAL-TRANSITION V0 PROVENANCE GATE')
  print('  git commit        :', _git_commit())
  print('  env               :', ENV_NAME)
  print('  p_active          :', P_ACTIVE)
  print('  horizon           :', HORIZON)
  print('  reset fix         :', RESET_FIX)
  print('  severity (v2.1)   :', SEVERITY)
  print('  clean dataset     :', CLEAN_NPZ)
  print('  clean dataset sha :', clean_sha)
  print('  failure bank      :', BANK_NPZ)
  print('  failure bank sha  :', bank_sha)
  print('  failure bank size :', '%d x %d' % bank_shape)
  print('  fail_neg_alpha    :', alpha,
        '(baseline arm -- bank NOT wired)' if alpha == 0.0 else '')
  if require_backend:
    print('  jax backend       :', backend, '|', devices)
  print('=' * 70)

  if clean_sha != CLEAN_SHA:
    raise SystemExit('ABORT: clean dataset sha is not the authoritative sha\n'
                     f'  expected {CLEAN_SHA}\n  got      {clean_sha}')
  if bank_sha == LEGACY_BANK_SHA:
    raise SystemExit('ABORT: this is the LEGACY (pre-settle) failure bank -- '
                     'refusing to run')
  if bank_sha != BANK_SHA:
    raise SystemExit('ABORT: failure bank sha is not the settled-bank sha\n'
                     f'  expected {BANK_SHA}\n  got      {bank_sha}')
  if bank_shape != BANK_SHAPE:
    raise SystemExit(f'ABORT: bank must be {BANK_SHAPE}, got {bank_shape}')
  if alpha not in ALPHA_ARMS:
    raise SystemExit(f'ABORT: alpha must be one of {sorted(ALPHA_ARMS)}')
  if manifest['bank']['sha256'] != bank_sha:
    raise SystemExit('ABORT: bank manifest sha drift')
  if not all(manifest['checks'].values()):
    failed = [k for k, v in manifest['checks'].items() if not v]
    raise SystemExit(f'ABORT: bank validation not clean: {failed}')
  if manifest['heldout_fresh_deaths']['n'] != 40:
    raise SystemExit('ABORT: held-out fresh-death list missing/short')
  if require_backend and not allow_cpu and backend != 'gpu':
    raise SystemExit('ABORT: no GPU backend (pass --allow-cpu only for a '
                     'CPU smoke/check)')

  print('PROVENANCE GATE PASSED')
  return {'clean_sha': clean_sha, 'bank_sha': bank_sha,
          'bank_shape': bank_shape, 'backend': backend}


def build_v0_config(steps=DEFAULT_STEPS, seed=0, ckpt_dir='',
                    alpha=DEFAULT_ALPHA):
  """The frozen V0 config: faithful offline recipe + authoritative rockfall env."""
  from verify_offline_d4rl import build_offline_cfg
  from rockfall_v2_teacher import SEVERITY_V2

  # Catch drift between this module's literal and the teacher's constant.
  if tuple(SEVERITY_V2) != SEVERITY:
    raise SystemExit(f'ABORT: rockfall_v2_teacher.SEVERITY_V2={SEVERITY_V2} '
                     f'disagrees with the V0 pin {SEVERITY}')
  if alpha not in ALPHA_ARMS:
    raise SystemExit(f'ABORT: alpha must be one of {sorted(ALPHA_ARMS)}')

  cfg = build_offline_cfg(max_steps=steps, ckpt_dir=ckpt_dir)
  cfg.env_name = ENV_NAME
  cfg.offline_dataset = CLEAN_NPZ
  cfg.eval_goal_mode = 'd4rl'
  cfg.seed = int(seed)

  # Rockfall env overrides -- declared fields on Config (defaults None/False =
  # the legacy env behaviour), so they are type-checked at construction and
  # recorded in the startup banner. assert_env_matches() below still verifies
  # they LANDED on the env object, which the config layer cannot prove.
  cfg.rockfall_severity = SEVERITY
  cfg.rockfall_p_active = P_ACTIVE
  cfg.rockfall_max_steps = HORIZON
  cfg.rockfall_reset_fix = RESET_FIX
  cfg.max_episode_steps = HORIZON

  # Failure-aware negatives. alpha=0 leaves fail_bank_path empty so the critic
  # loss and gradients are byte-identical to the baseline.
  if alpha > 0.0:
    cfg.fail_bank_path = BANK_NPZ
    cfg.fail_neg_alpha = float(alpha)
  else:
    cfg.fail_bank_path = ''
    cfg.fail_neg_alpha = 0.0
  return cfg


def assert_env_matches(cfg, seed=0):
  """Build the env and verify every override actually landed on the object.

  This is the check the getattr-based override path cannot perform for itself.
  """
  from crl import envs as envs_mod
  env = envs_mod.make_env(cfg.env_name, cfg, seed=seed)
  problems = []
  if abs(float(env.p_active) - P_ACTIVE) > 1e-12:
    problems.append(f'p_active={env.p_active} (want {P_ACTIVE})')
  if tuple(float(p) for p in env.severity_probs) != SEVERITY:
    problems.append(f'severity_probs={env.severity_probs} (want {SEVERITY})')
  if int(env.max_episode_steps) != HORIZON:
    problems.append(f'max_episode_steps={env.max_episode_steps} '
                    f'(want {HORIZON})')
  if bool(getattr(env._env, 'full_reset', False)) != RESET_FIX:  # noqa: SLF001
    problems.append(f'full_reset={getattr(env._env, "full_reset", None)} '  # noqa: SLF001
                    f'(want {RESET_FIX})')
  if int(getattr(env, 'death_settle_substeps', 0)) != DEATH_SETTLE_SUBSTEPS:
    problems.append(f'death_settle_substeps={env.death_settle_substeps} '
                    f'(want {DEATH_SETTLE_SUBSTEPS})')
  if int(cfg.obs_dim) != 29 or int(cfg.action_dim) != 8:
    problems.append(f'obs_dim={cfg.obs_dim} action_dim={cfg.action_dim} '
                    '(want 29/8)')
  if problems:
    raise SystemExit('ABORT: rockfall overrides did NOT land on the env '
                     '(silent-getattr failure):\n  ' + '\n  '.join(problems))
  print('ENV OVERRIDE CHECK PASSED  '
        f'(p_active={env.p_active} H={env.max_episode_steps} '
        f'severity={env.severity_probs} full_reset={env._env.full_reset} '  # noqa: SLF001
        f'obs={cfg.obs_dim} act={cfg.action_dim})')
  return env


def smoke(alpha, seed=0, n_updates=5):
  """5-update in-process check: bank reaches the loss and the identity holds."""
  import time

  import jax
  import optax

  from crl import losses as losses_mod
  from crl import networks as networks_mod
  from crl.replay import TrajectoryBuffer, obs_to_goal

  cfg = build_v0_config(steps=DEFAULT_STEPS, seed=seed, ckpt_dir='',
                        alpha=alpha)
  assert_env_matches(cfg, seed=seed)

  fail_bank = None
  if alpha > 0.0:
    with np.load(os.path.join(_ROOT, BANK_NPZ)) as fb:
      bank = np.asarray(fb['goals'], np.float32)
    fail_bank = obs_to_goal(bank, cfg.start_index, cfg.end_index,
                            cfg.goal_indices)

  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=False, use_layer_norm=cfg.use_layer_norm)
  init_state, update_step = losses_mod.build_learner(
      nets, cfg, obs_to_goal, optax.adam(cfg.actor_learning_rate),
      optax.adam(cfg.learning_rate), fail_bank=fail_bank)

  with np.load(os.path.join(_ROOT, CLEAN_NPZ), allow_pickle=True) as d:
    obs, act = d['obs'], d['act']
  buf = TrajectoryBuffer(
      capacity_steps=obs.shape[0] * obs.shape[1], ep_len_obs=obs.shape[1],
      full_obs_dim=obs.shape[2], action_dim=act.shape[2], obs_dim=cfg.obs_dim,
      start_index=cfg.start_index, end_index=cfg.end_index,
      discount=cfg.discount, seed=cfg.seed,
      goal_indices=tuple(range(cfg.obs_dim)))
  for k in range(obs.shape[0]):
    buf.add_episode(obs[k], act[k])

  state = init_state(jax.random.PRNGKey(cfg.seed))
  state, m = update_step(state, buf.sample(cfg.batch_size))    # compile
  m = {k: float(v) for k, v in m.items()}

  if alpha > 0.0:
    assert m['fail_bank_size'] == BANK_SHAPE[0], m['fail_bank_size']
    assert abs(m['fail_neg_alpha'] - alpha) < 1e-6, m['fail_neg_alpha']
    lhs = (m['critic_pos_term'] + m['critic_neg_ord_term']
           + m['critic_neg_fail_term'])
    assert abs(lhs - m['critic_loss']) < 1e-5, (lhs, m['critic_loss'])
    extra = ('| fail_bank_size %d | logits_fail_neg %.3f | '
             'L = pos + %.1f*ord + %.1f*fail verified'
             % (m['fail_bank_size'], m['logits_fail_neg'], 1 - alpha, alpha))
  else:
    assert 'critic_neg_fail_term' not in m, (
        'alpha=0 must not produce failure-negative metrics')
    extra = '| baseline arm: no failure-negative branch in the loss'

  t0 = time.time()
  for _ in range(n_updates):
    state, m = update_step(state, buf.sample(cfg.batch_size))
  jax.block_until_ready(state.q_params)
  print('SMOKE OK | backend %s %s | %.1f updates/s'
        % (jax.default_backend(), extra, n_updates / (time.time() - t0)))


def main():
  ap = argparse.ArgumentParser(
      description='V0 baseline launcher (clean CRL + failure-aware negatives). '
                  'The dataset, p_active, horizon, reset-fix and severity are '
                  'FROZEN and cannot be overridden by flag.')
  mode = ap.add_mutually_exclusive_group(required=True)
  mode.add_argument('--check-only', action='store_true',
                    help='provenance gate + config/env construction only')
  mode.add_argument('--smoke', action='store_true',
                    help='gate + env check + 5 in-process learner updates')
  mode.add_argument('--run', action='store_true',
                    help='the production training run')
  ap.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                  help=f'failure-negative mixture weight; one of '
                       f'{sorted(ALPHA_ARMS)} (default {DEFAULT_ALPHA})')
  ap.add_argument('--steps', type=int, default=DEFAULT_STEPS)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--ckpt-dir', default='',
                  help='default: causal_v0_a<alpha>_s<seed>_<steps>k')
  ap.add_argument('--resume', action='store_true')
  ap.add_argument('--allow-cpu', action='store_true',
                  help='permit a non-GPU backend (smoke/check only)')
  args = ap.parse_args()

  alpha = float(args.alpha)
  if alpha not in ALPHA_ARMS:
    raise SystemExit(f'ABORT: --alpha must be one of {sorted(ALPHA_ARMS)}; '
                     f'got {alpha}. The V0 arms are pre-registered.')
  if args.run and args.allow_cpu:
    raise SystemExit('ABORT: --allow-cpu is for --smoke/--check-only only')

  os.chdir(_ROOT)
  provenance_gate(alpha, allow_cpu=args.allow_cpu,
                  require_backend=not args.check_only)

  if args.check_only:
    cfg = build_v0_config(steps=args.steps, seed=args.seed, ckpt_dir='',
                          alpha=alpha)
    assert_env_matches(cfg, seed=args.seed)
    print('CHECK-ONLY COMPLETE (no training performed)')
    return

  if args.smoke:
    smoke(alpha, seed=args.seed)
    print('SMOKE COMPLETE (no checkpoints written)')
    return

  tag = ('a%s' % ('01' if alpha == 0.1 else '00'))
  ckpt_dir = args.ckpt_dir or (
      f'causal_v0_{tag}_s{args.seed}_{args.steps // 1000}k')
  os.makedirs(ckpt_dir, exist_ok=True)
  cfg = build_v0_config(steps=args.steps, seed=args.seed, ckpt_dir=ckpt_dir,
                        alpha=alpha)
  assert_env_matches(cfg, seed=args.seed)
  cfg.resume = bool(args.resume)

  from crl.train import train
  print(f'RUN causal-transition V0 | alpha={alpha} seed={args.seed} '
        f'steps={args.steps} | ckpt_dir={ckpt_dir}', flush=True)
  train(cfg)


if __name__ == '__main__':
  main()
