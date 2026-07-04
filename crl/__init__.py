"""Acme-free port of Contrastive RL (Eysenbach et al., 2022).

This package reimplements the training pipeline of the original ``contrastive/``
package WITHOUT DeepMind Acme / Launchpad / Reverb / TF, so it runs as a single
process on modern JAX (Colab-friendly). The *algorithm* (networks + losses) is a
faithful port of ``contrastive/networks.py`` and ``contrastive/learning.py``;
only the orchestration, replay buffer, and goal-relabeling data pipeline are
replaced.

Modules:
  config    -- hyperparameter dataclass (port of contrastive/config.py)
  networks  -- Haiku critic/actor + tanh-normal policy (port of networks.py)
  losses    -- NCE / CPC / C-learning / GCBC losses (port of learning.py)
  replay    -- numpy trajectory buffer + geometric future-goal sampler
               (port of the TF flatten_fn in contrastive/builder.py)
  envs      -- modern env registry (point_env; fetch via gymnasium-robotics)
  train     -- single-process train + eval loop
"""
