"""Offline CRL + ETT stage 0 on the V6 long two-rockfall benchmark.

EVERY hyperparameter lives in configs/ant_maze_configs.yml, as a flat map of
crl.config.Config field names. This script reads that file, hands the values to
Config, and calls ETT_train(). There is no second place to look and no
command-line knobs: to vary a run, edit the YAML (or point the script at
another one). The shipped YAML mirrors
scripts/train_rockfall_clock_v6_baseline.py --steps 100000 --seed 0 (its
defaults: the _gxy env, horizon 800, the p_active 0.40 far30 dataset and the V6
module's t0 ranges).

Nothing in this launcher is version-specific -- the config file is what picks
the benchmark, and the other benchmarks are runnable unchanged through their
own configs:
  python ant_maze/train_ctl_ant_maze.py configs/ant_maze_v7_configs.yml
  python ant_maze/train_ctl_ant_maze.py configs/ant_maze_v5_configs.yml
V7 is V6 with a signed death position: the torso XY the observation reports is
negated from the fatal rock contact onward, so a death lands outside the
reachable map. It reads the same rockfall_p_active_* / rockfall_t0_* fields as
V6, which is why the two configs differ by little more than the env name.

Outputs (checkpoints, metrics.json, tb/) go to the ckpt_dir the YAML builds
from the ${now:} resolver registered below, i.e.
./logs/antmaze_v6_<timestamp>.

Usage: python ant_maze/train_ctl_ant_maze.py [CONFIG_FILE]
"""
import datetime
import os
import shutil
import sys

from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    'now', lambda fmt='%Y%m%d_%H%M%S':
    datetime.datetime.now().strftime(fmt))

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from crl.config import Config  # noqa: E402
from crl.ETT_train import ETT_train as ETT_train    # noqa: E402

DEFAULT_CONFIG = os.path.join(_ROOT, 'configs', 'ant_maze_configs.yml')


def load_config(path):
  """Builds the run Config from `path`. The YAML is the only source of values."""
  values = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
  # YAML sequences arrive as lists; the dataclass fields are tuples.
  return Config(**{k: tuple(v) if isinstance(v, list) else v
                   for k, v in values.items()})


def hazard_banner(cfg):
  """The latent knobs this run actually carries, V6/V7's pair or V5's single one.

  V6 replaced V5's one rockfall_p_active with two independent per-zone hazard
  coins and two independent reset-time clocks, so the run line has to report
  whichever set the config filled in: the V6/V7 branch of envs.make_env never
  reads rockfall_p_active, and the V5 branch never reads the _1/_2 fields. V7
  reuses V6's fields exactly and adds one of its own, the signed death
  position, whose unset default is the V7 benchmark; a V7 config that turns it
  off is an audit run reproducing V6 and not a benchmark number, so the banner
  names the field only then. V6 itself never reads it.
  """
  parts = []
  if cfg.rockfall_p_active is not None:
    parts.append(f'p_active {cfg.rockfall_p_active}')
  if cfg.rockfall_p_active_1 is not None or cfg.rockfall_p_active_2 is not None:
    parts.append(f'p_active ({cfg.rockfall_p_active_1}, '
                 f'{cfg.rockfall_p_active_2})')
    parts.append(f't0 [{cfg.rockfall_t0_min_1}, {cfg.rockfall_t0_max_1}] / '
                 f'[{cfg.rockfall_t0_min_2}, {cfg.rockfall_t0_max_2}]')
  if cfg.rockfall_negate_death_xy is False:
    parts.append('negate_death_xy OFF (audit: V7 reproducing V6)')
  return ' | '.join(parts) if parts else 'env defaults'


def main():
  cfg_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
  cfg = load_config(cfg_path)
  if cfg.ckpt_dir:
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    # Snapshot the config INTO the run directory. ckpt_dir carries a ${now:}
    # timestamp, so the YAML on disk is free to move on between runs; without
    # this copy an A/B comparison months later cannot establish which arm a
    # logs/ directory actually was.
    shutil.copyfile(cfg_path, os.path.join(cfg.ckpt_dir, 'config.yml'))
  # One line recording which benchmark this run actually ran: the env knobs
  # (horizon and latent density must match the dataset the YAML names), the
  # arm (bc_coef 0.05 = crl / 1.0 = gcbc, fail-neg alpha 0.0 = plain negatives)
  # and whether ETT stage 0 ran at all, since dyn_enable=False reduces the run
  # to plain offline CRL no matter what the dyn_* block says.
  if cfg.dyn_enable:
    ett = (f'ETT stage0 ON (fit {cfg.dyn_train_steps} steps, augment '
           f'{cfg.dyn_augment_frac:g})')
  else:
    ett = 'ETT stage0 OFF'
  print(f'{cfg.env_name} run on {cfg.offline_dataset}'
        f' | steps {cfg.max_number_of_steps} | seed {cfg.seed}'
        f' | horizon {cfg.rockfall_max_steps} | {hazard_banner(cfg)}'
        f' | bc_coef {cfg.bc_coef} | fail-neg alpha {cfg.fail_neg_alpha}'
        f' | {ett}',
        flush=True)
  ETT_train(cfg)


if __name__ == '__main__':
  main()
