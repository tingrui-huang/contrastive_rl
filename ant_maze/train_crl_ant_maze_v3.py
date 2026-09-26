"""Offline CRL + ETT v3 on the clock V6 long two-rockfall benchmark.

The same launcher as ant_maze/train_ctl_ant_maze_v2.py, calling
crl.ETT_train_v3: no done head and no generated-negative BCE. The generated
trajectories are mixed into every minibatch at config.dyn_augment_frac, and
the critic, actor and BC losses all read the mixed batch. Every
hyperparameter lives in the YAML; the default is
configs/ant_maze_v6_ett_v3_configs.yml.

Usage: python ant_maze/train_crl_ant_maze_v3.py [CONFIG_FILE]
"""
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from train_ctl_ant_maze import hazard_banner, load_config  # noqa: E402
from crl.ETT_train_v3 import ETT_train_v3                   # noqa: E402

DEFAULT_CONFIG = os.path.join(_ROOT, 'configs', 'ant_maze_v6_ett_v3_configs.yml')


def main():
  cfg_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
  cfg = load_config(cfg_path)
  if cfg.ckpt_dir:
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    shutil.copyfile(cfg_path, os.path.join(cfg.ckpt_dir, 'config.yml'))
  if cfg.dyn_enable:
    ett = (f'ETT v3 stage0 ON (fit {cfg.dyn_train_steps} steps, augment '
           f'{cfg.dyn_augment_frac:g}, no done head)')
  else:
    ett = 'ETT stage0 OFF'
  print(f'{cfg.env_name} run on {cfg.offline_dataset}'
        f' | steps {cfg.max_number_of_steps} | seed {cfg.seed}'
        f' | horizon {cfg.rockfall_max_steps} | {hazard_banner(cfg)}'
        f' | bc_coef {cfg.bc_coef} | fail-neg alpha {cfg.fail_neg_alpha}'
        f' | {ett}',
        flush=True)
  ETT_train_v3(cfg)


if __name__ == '__main__':
  main()
