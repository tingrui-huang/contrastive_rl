"""ETT v3: generated trajectories mixed into every minibatch, no done head.

The training loop is crl/ETT_train.py's, unchanged: the generated
trajectories live in their own buffer and ``dyn_augment_frac * batch_size``
rows of every minibatch are drawn from it, the rest from the frozen dataset.
The concatenated batch feeds ALL THREE losses -- the contrastive critic loss,
the actor loss and the BC loss.

What v3 removes, relative to crl/ETT_train_v2.py:

  * the done head. No ``P(done | s)`` classifier is fitted
    (``dyn_done_dataset`` is cleared), and rollouts are called with
    ``stop_on_done=False``, so a loaded pickle that carries a done head still
    runs every row the full ``dyn_rollout_steps``;
  * the generated-negative BCE term (``gen_neg_coef`` is forced to 0), which
    labeled the real batch 1 and the generated batch 0. Generated data is
    ordinary training data here, not a negative class.

Usage:
    python -m crl.ETT_train_v3 --offline_dataset path/to/data.npz
    python ant_maze/train_crl_ant_maze_v3.py configs/ant_maze_v6_ett_v3_configs.yml
"""
from crl.config import Config
from crl.ETT_train import ETT_train, _apply_overrides, _build_arg_parser


def ETT_train_v3(config: Config):
  if getattr(config, 'dyn_done_dataset', ''):
    print(f'  [ETT v3] dyn_done_dataset={config.dyn_done_dataset} -> '
          "'' (v3 fits no done head)", flush=True)
    config.dyn_done_dataset = ''
  if float(getattr(config, 'gen_neg_coef', 0.0) or 0.0) > 0.0:
    print(f'  [ETT v3] gen_neg_coef={config.gen_neg_coef} -> 0 (v3 has no '
          'generated-negative BCE; generated rows join the minibatch at '
          f'dyn_augment_frac={config.dyn_augment_frac:g})', flush=True)
    config.gen_neg_coef = 0.0
  return ETT_train(config, stop_on_done=False)


def main():
  args = _build_arg_parser().parse_args()
  config = _apply_overrides(Config(), args)
  ETT_train_v3(config)


if __name__ == '__main__':
  main()
