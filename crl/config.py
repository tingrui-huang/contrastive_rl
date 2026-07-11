"""Config for the Acme-free contrastive RL port.

Port of ``contrastive/config.py``: same algorithm hyperparameters, minus the
Acme/Reverb-specific fields. A few single-process orchestration knobs are added
(``updates_per_step``, ``random_steps``, eval cadence) that Acme previously
handled implicitly via Launchpad + the Reverb rate limiter.
"""
import dataclasses
from typing import Optional, Tuple, Union


@dataclasses.dataclass
class Config:
  """Configuration options for contrastive RL (single-process port)."""

  # --- Environment ---
  env_name: str = 'point_Small'
  max_number_of_steps: int = 1_000_000  # total ENV steps to train for.

  # These four are filled in from the env at startup (see envs.make_env).
  obs_dim: int = -1          # size of the STATE part of the observation.
  goal_dim: int = -1         # size of the GOAL part (== end_index-start_index).
  action_dim: int = -1
  max_episode_steps: int = -1

  # Which coordinates of the state form the goal. point: (0, -1) => full state.
  # fetch_reach: (0, 3) => gripper xyz.  fetch_push: (3, 6) => object xyz.
  start_index: int = 0
  end_index: int = -1

  # --- Loss options (identical defaults to the original paper) ---
  batch_size: int = 256
  actor_learning_rate: float = 3e-4
  learning_rate: float = 3e-4
  discount: float = 0.99
  # Entropy bonus coefficient. None => adaptive (SAC-style) alpha.
  entropy_coefficient: Optional[float] = 0.0
  target_entropy: float = 0.0
  tau: float = 0.005                       # target network Polyak coefficient.
  hidden_layer_sizes: Tuple[int, ...] = (256, 256)
  repr_dim: Union[int, str] = 64           # representation size.
  repr_norm: bool = False
  repr_norm_temp: bool = True

  # Algorithm selector flags (see losses.py). Defaults => contrastive_nce.
  use_cpc: bool = False        # CPC (softmax) instead of NCE (binary).
  use_td: bool = False         # C-learning (TD) instead of Monte-Carlo.
  add_mc_to_td: bool = False   # nce+c_learning hybrid (requires use_td).
  use_gcbc: bool = False       # goal-conditioned behavior cloning baseline.
  twin_q: bool = False
  random_goals: float = 0.5    # actor-loss goal mixing: 0.0 / 0.5 / 1.0.
  use_image_obs: bool = False

  # --- Replay ---
  min_replay_size: int = 10_000     # env steps before learning starts.
  max_replay_size: int = 1_000_000  # env steps kept in the buffer.

  # --- Single-process orchestration (replaces Launchpad + rate limiter) ---
  # Gradient steps performed per env step, once warmed up. The original ran the
  # learner asynchronously; here we set the sample/insert ratio explicitly.
  updates_per_step: int = 1
  # Batches sampled+applied per learner.step (was num_sgd_steps_per_step; kept
  # so throughput matches when you want it, but 1 is fine for correctness).
  num_sgd_steps_per_step: int = 1
  # Take uniformly random actions for this many initial env steps.
  random_steps: int = 10_000

  jit: bool = True
  seed: int = 0

  # --- Eval / logging ---
  eval_every_steps: int = 10_000
  eval_episodes: int = 20
  log_every_steps: int = 1_000
  tensorboard: bool = False     # mirror scalars to <ckpt_dir>/tb (optional).

  # --- Checkpointing (point ckpt_dir at a Google Drive folder on Colab) ---
  ckpt_dir: str = ''            # '' disables checkpointing.
  ckpt_every_steps: int = 0     # 0 => checkpoint on every eval.
  resume: bool = False          # resume params/optimizer from ckpt_dir/latest.pkl.
