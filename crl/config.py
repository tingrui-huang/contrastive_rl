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
  # Optional NON-CONTIGUOUS goal coordinates (overrides start/end when set;
  # filled from the env by make_env). Must start with the XY indices (0, 1) so
  # XY success/distance metrics stay comparable across goal representations.
  goal_indices: Optional[Tuple[int, ...]] = None

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
  # LayerNorm in the critic encoders + actor torso (Stabilizing-Contrastive-RL
  # arm). Default False = faithful google-research recipe (byte-identical net,
  # so faithful checkpoints load). The LayerNorm notebook sets this True.
  use_layer_norm: bool = False

  # Algorithm selector flags (see losses.py). Defaults => contrastive_nce.
  use_cpc: bool = False        # CPC (softmax) instead of NCE (binary).
  use_td: bool = False         # C-learning (TD) instead of Monte-Carlo.
  add_mc_to_td: bool = False   # nce+c_learning hybrid (requires use_td).
  use_gcbc: bool = False       # goal-conditioned behavior cloning baseline.
  twin_q: bool = False
  random_goals: float = 0.5    # actor-loss goal mixing: 0.0 / 0.5 / 1.0.
  use_image_obs: bool = False
  # Offline actor regularization (paper Eq 7-8 / WindyCorridor recipe):
  # loss = (1-bc_coef)*(alpha*logp - Q) + bc_coef*(-log pi(a_orig|s,g)).
  # 0.0 = pure online SAC-style actor (unchanged default); offline runs use 0.5.
  bc_coef: float = 0.0

  # offline_ant_umaze eval goal source. 'd4rl' (default) = the benchmark
  # goal_sampler (single U_MAZE goal cell + [0,1.5] noise, resampled per
  # episode) == the protocol the paper/D4RL score with. 'dataset' = replay the
  # empirical infos/goal (~2x noise, HARDER, provenance only). 'fixed' = single
  # (0.75, 8.75). No effect on non-offline-ant envs.
  eval_goal_mode: str = 'd4rl'

  # --- offline_ant_umaze_rockfall env overrides ---
  # These were previously set as AD-HOC attributes on the config instance and
  # read back in envs.make_env with getattr(config, 'rockfall_*', <default>).
  # That made a dropped or misspelled assignment degrade SILENTLY to the older,
  # easier setting instead of raising, and kept the values out of the startup
  # Config banner, so a run's log did not record which benchmark it actually
  # ran. Declaring them changes NO defaults: every value below is exactly the
  # fallback make_env already used, so unset runs stay byte-identical and the
  # v1 / p=0.20 / H=700 / legacy-reset experiments are untouched.
  #
  # Authoritative v2.1 H800 setting (see notes/CAUSAL_TRANSITION_V0.md):
  #   severity (0.80, 0.15, 0.05), p_active 0.30, max_steps 800,
  #   reset_fix True, death_settle_substeps 0.
  #
  # Per-run eval lethality. None => the env's frozen default 0.55/0.30/0.15.
  rockfall_severity: Optional[Tuple[float, ...]] = None
  # Hidden hazard-mask density. None => the env's frozen default P_ACTIVE=0.20.
  rockfall_p_active: Optional[float] = None
  # Episode horizon. None => the env's frozen default 700.
  rockfall_max_steps: Optional[int] = None
  # MuJoCo substeps integrated INSIDE the fatal transition (actor ctrl zeroed)
  # so the recorded post-fatal state is physically settled. None => 0 = legacy
  # freeze-at-contact. This is a DATASET/BANK-construction knob; training and
  # eval runs leave it at 0.
  rockfall_death_settle_substeps: Optional[int] = None
  # Canonical episode-independent full reset. False => legacy reset.
  rockfall_reset_fix: bool = False

  # --- Offline mode ---
  # Path to an .npz episode dataset (obs [N,L,obs+goal], act [N,L,A], see
  # scripts/collect_push_dataset.py). Non-empty => the buffer is preloaded once
  # and NO env interaction happens during training (env used for eval only);
  # 'steps' then count the gradient clock, not env steps.
  offline_dataset: str = ''

  # --- Failure-aware negative sampling (Part 1 experiment; OFF by default) ---
  # Path to a failure-state bank npz with 'goals' [N_bank, obs_dim] (states in
  # the learner state space; sliced to goal coords at load). '' disables.
  fail_bank_path: str = ''
  # Mixture weight alpha on the critic NEGATIVE distribution,
  # q_alpha = (1-alpha)*p_clean + alpha*q_fail. Loss-level mixture that keeps
  # the positive term and the total negative mass at their original weights:
  # L = L_pos + (1-alpha)*L_ordinary-neg + alpha*L_failure-neg, with
  # L_failure-neg the exact uniform expectation over the bank (all states
  # scored, no sampling). 0.0 => the fail branch is skipped entirely and the
  # critic loss/gradients are byte-identical to the baseline.
  fail_neg_alpha: float = 0.0

  # --- Anchor sampling ("scheme C"; OFF by default) ---
  # '' keeps the original fixed-length draw (anchor uniform over ALL rows).
  # 'arrival' restricts ANCHORS to rows before the episode parks -- the first
  # row within anchor_cut_radius of its own goal, or the start of a frozen
  # terminal run (an absorbing/dead state), whichever comes first. The
  # future-goal window is NOT truncated: the geometric relabeling law and the
  # set of samplable goals are unchanged, so this is not the 'lengths' path.
  # Rationale: on a fixed-length dataset most rows can be post-arrival idling,
  # which costs anchor budget without carrying a decision.
  anchor_cut_mode: str = ''
  anchor_cut_radius: float = 0.5

  # --- Balanced (s, a) anchor sampling (OFF by default) ---
  # Draw a (discretised state, action sector) bucket uniformly, then a row
  # uniformly inside it, instead of drawing rows uniformly. Uniform-over-rows
  # is really "weighted by how often the behaviour policy chose it", which
  # fights any method whose goal is to move the agent onto a rarely-taken
  # route. Continuous (s, a) pairs are almost all unique, so the discretisation
  # below is a required modelling choice, not a convenience.
  # NOTE: this changes the distribution the loss integrates over, shifting the
  # contrastive optimum by a goal-only term -- harmless with a single fixed
  # commanded goal, but re-derive before any cross-goal comparison.
  balanced_sampling: bool = False
  balanced_cell_size: float = 1.0
  balanced_action_sectors: int = 4
  # Weight ceiling per bucket: a bucket is drawn with probability
  # proportional to min(count, cap). Strict uniform (cap None) amplifies a
  # one-row bucket by ~1000x on continuous data; the cap leaves small
  # buckets at their natural relative frequency and flattens only the
  # over-represented ones. 0 => strict uniform.
  balanced_cap: int = 0

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
  # Number of data-collection actors. >1 replicates the original recipe's
  # multi-actor collection with N logical env instances (distinct seeds/RNGs)
  # stepped in lockstep in-process, one batched policy forward per step.
  # env-step accounting and the learner-updates-per-TOTAL-env-step ratio are
  # unchanged (the budget is TOTAL across actors, as in acme's layout).
  num_actors: int = 1
  # Snapshot the replay buffer to <ckpt_dir>/replay.npz when train() exits
  # (incl. guard aborts) and restore it on --resume, so staged runs keep
  # their data. Off by default (large file).
  save_replay: bool = False

  jit: bool = True
  seed: int = 0

  # --- Numerical guard (opt-in): abort training when the learner state blows
  # up (non-finite actor/critic losses, logits, alpha, or parameters, or
  # |actor_loss| above the threshold). Off by default.
  guard_abort: bool = False
  guard_actor_loss_max: float = 1e6

  # --- Eval / logging ---
  eval_every_steps: int = 10_000
  eval_episodes: int = 20
  log_every_steps: int = 1_000
  tensorboard: bool = False     # mirror scalars to <ckpt_dir>/tb (optional).

  # --- Checkpointing (point ckpt_dir at a Google Drive folder on Colab) ---
  ckpt_dir: str = ''            # '' disables checkpointing.
  ckpt_every_steps: int = 0     # 0 => checkpoint on every eval.
  resume: bool = False          # resume params/optimizer from ckpt_dir/latest.pkl.
  # Extra named milestone checkpoints saved as <step>.pkl the first eval at or
  # past each step (in ADDITION to init/early/mid/final/latest/best). Empty =>
  # legacy behavior. Used by the image-conedir qualification (10k..70k).
  ckpt_milestone_steps: Tuple[int, ...] = ()
  # best.pkl update rule. False (default) => save when success >= best (legacy,
  # ties overwrite). True => save only on STRICT improvement success > best, so
  # best.pkl stays at the earliest checkpoint that reached the top success.
  best_strict_improvement: bool = False
  # FetchPush image runs: compute eval success/final_dist/min_dist from the
  # SIMULATOR object-goal coordinates (physical) instead of flattened image-L2.
  # No effect on non-FetchPush or state-obs runs.
  physical_eval_push: bool = False
