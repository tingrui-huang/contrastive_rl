"""Config for the Acme-free contrastive RL port.

Port of ``contrastive/config.py``: same algorithm hyperparameters, minus the
Acme/Reverb-specific fields. A few single-process orchestration knobs are added
(``updates_per_step``, ``random_steps``, eval cadence) that Acme previously
handled implicitly via Launchpad + the Reverb rate limiter.
"""
import dataclasses
from typing import Any, Optional, Tuple, Union


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
  # offline_antmaze_rockfall_clock_v6 eval goal POSITION, an (x, y) pair.
  # Set => every eval episode commands this fixed point (no noise) and
  # success is reaching it; it overrides eval_goal_mode. None => the V6
  # benchmark goal from eval_goal_mode. Read by the V6 branch of make_env only.
  eval_goal: Optional[Tuple[float, float]] = None
  # offline_antmaze_rockfall_clock_v6 SUCCESS REGION. None => the AntMaze
  # test, dist(torso xy, goal xy) <= 0.5 (the frozen V6 benchmark). A float h
  # or a pair (hx, hy) => a box of size 2hx x 2hy centred on the goal: success
  # iff |dx| <= hx and |dy| <= hy ((1.0, 0.5) = a 2 x 1 goal box). Read by the
  # V6 branch of make_env only.
  eval_goal_box_half: Optional[Any] = None

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

  # --- rockfall_clock_v6_long_two_rockfall overrides ---
  # V6 has two independently sampled hazards and two independently sampled
  # reset-time absolute clocks.  None preserves the explicit defaults in
  # crl/rockfall_clock_v6.py; V6 scripts always materialise all six values in
  # their run metadata/config banner.
  rockfall_p_active_1: Optional[float] = None
  rockfall_p_active_2: Optional[float] = None
  rockfall_t0_min_1: Optional[int] = None
  rockfall_t0_max_1: Optional[int] = None
  rockfall_t0_min_2: Optional[int] = None
  rockfall_t0_max_2: Optional[int] = None

  # --- rockfall_clock_v7_long_two_rockfall override ---
  # V7 reuses every rockfall_p_active_* / rockfall_t0_* field above, because
  # it is V6 with one change.  This is the one field that is only V7's:
  # None -> True -> the benchmark's negated death position.  Set it False
  # only in an audit that wants V7 to reproduce V6 exactly.
  rockfall_negate_death_xy: Optional[bool] = None
  # Burst length, in env steps, of one zone's rockfall (read by BOTH the V6 and
  # the V7 env; the field lives here because V7 needed it first). None keeps
  # the module's ROCKFALL_STEPS = 72, the frozen benchmark that every existing
  # V6/V7 dataset, run and audit was produced at. A larger value makes
  # the sighted teacher wait longer at an ARMED mouth, so armed shortcut
  # episodes get longer while the detour is untouched -- which flips the
  # length ratio the uniform future-goal relabeler normalises by (crl/replay.py
  # draws a goal with mass 1/(L-t)). It changes the benchmark, so the
  # collector records it in the dataset meta and the V7 trainer refuses a
  # dataset whose value disagrees with the CLI. Nothing gates it on the V6
  # side, so a V6 run that sets it is evaluating on a different env from the
  # one its offline_dataset was collected on -- deliberate or not.
  rockfall_steps: Optional[int] = None

  # --- rockfall_v7_three_trap overrides ---
  # V7 replaces V6's two absolute clocks with three independent per-trap kill
  # coins (start / center / end of the shortcut).  Entering trap z kills with
  # probability p_kill_z.  None preserves the explicit defaults in
  # crl/rockfall_v7.py; V7 scripts always materialise all three values in
  # their run metadata/config banner.
  rockfall_p_kill_1: Optional[float] = None
  rockfall_p_kill_2: Optional[float] = None
  rockfall_p_kill_3: Optional[float] = None

  # --- Offline mode ---
  # Path to an .npz episode dataset (obs [N,L,obs+goal], act [N,L,A], see
  # scripts/collect_push_dataset.py). Non-empty => the buffer is preloaded once
  # and NO env interaction happens during training (env used for eval only);
  # 'steps' then count the gradient clock, not env steps.
  offline_dataset: str = ''

  # --- One-step transition model (ETT stage 0) ---
  # Fitted by crl/ETT_train.py on the frozen offline dataset BEFORE the first
  # contrastive gradient step: a plain MLP f(s_t, a_t) -> s_{t+1} over the
  # 29-dim learner STATE space (the goal half plays no part in dynamics).
  # Read ONLY by crl/causal_transition_model.py + ETT_train; crl/train.py
  # ignores them, so every existing CRL run is unaffected.
  # MASTER SWITCH for the whole stage: the causal-transition-model FIT and the
  # rollout SAMPLING that augments the dataset with it. False skips both
  # unconditionally -- no fit, no load from dyn_model_path, no augmentation
  # buffer even if dyn_augment_frac > 0 -- which makes crl/ETT_train.py behave
  # exactly like offline crl/train.py, so a vanilla-CRL baseline arm is one
  # line rather than a coordinated edit of dyn_train_steps, dyn_model_path and
  # dyn_augment_frac. The finer knobs still apply when it is True, and it is
  # deliberately NOT an error to leave dyn_augment_frac > 0 with the switch
  # off: that is the point, an augmented config can be run as its own baseline
  # without disturbing the numbers that define the augmented arm. The run
  # prints which of the two sub-stages the switch suppressed. Default True so
  # every existing config behaves as before.
  dyn_enable: bool = True
  dyn_train_steps: int = 20_000
  dyn_hidden_layer_sizes: Tuple[int, ...] = (512, 512)
  dyn_learning_rate: float = 1e-3
  dyn_batch_size: int = 1024
  # Fraction of EPISODES (not transitions) held out for validation. Consecutive
  # transitions inside an episode are near-duplicates, so a transition-level
  # split would report an optimistically small generalization error.
  dyn_holdout_frac: float = 0.1
  # Regress the residual s_{t+1} - s_t instead of the absolute next state. Over
  # one control step the state barely moves, so the residual target spares the
  # network from having to memorize the identity map.
  dyn_predict_delta: bool = True
  # Feed the dynamics head a COUNTERFACTUAL action alongside the factual one:
  # concat(s_t, a_t, a'_t) instead of concat(s_t, a_t). a' is the action that
  # COULD have been played at the same state s_t in place of the stored a_t --
  # it is NOT a_{t+1}, and nothing about it is a temporal successor, so it
  # lives in the same action space as a and at the same timestep. Nothing
  # counterfactual was ever executed in the environment, so every stored
  # transition is FACTUAL: a' IS a on every row of the frozen offline dataset,
  # and this widens the input by action_dim columns that are an exact copy of
  # the previous ones. It buys nothing on the fit itself; it is here so the
  # fitted artifact ACCEPTS an off-diagonal (genuinely counterfactual) query
  # f(s, a, a') at predict time, which is the interventional object the
  # causal-transition redesign wants. Read the identifiability warning in
  # crl/causal_transition_model.py before using such a query as evidence: the
  # data pins the model down only on the factual diagonal, and how it splits
  # its action dependence between the two blocks is an artifact of
  # initialization, not of the ant. Set to False for the historical (s, a) fit.
  # The field name is legacy -- it says next_action but means the
  # counterfactual action -- and is kept so existing configs and pickles load.
  dyn_next_action_input: bool = True
  dyn_log_every_steps: int = 1_000
  # Hidden activation. 'relu' is the historical default and is kept so that a
  # rerun of any existing config reproduces its previous fit bit-for-bit;
  # 'silu' fits this ant's contact dynamics measurably better (a ReLU net is
  # piecewise linear, which is a poor basis for smooth rigid-body flow).
  dyn_activation: str = 'relu'
  # Spectral normalization on EVERY layer of both heads (dynamics and blind
  # expert policy): each weight matrix is divided by its own largest singular
  # value and multiplied by dyn_spectral_norm_coef, which bounds the Lipschitz
  # constant of the network by coef ** n_layers. On a one-step dynamics model
  # that is a direct regularizer of the thing the model is FOR: it stops the
  # net from answering a small state perturbation with a large change of
  # predicted next state, which is what makes rollouts and any downstream
  # causal query blow up off the data manifold. Off by default, so every
  # existing config refits exactly as before.
  dyn_spectral_norm: bool = False
  # Target spectral norm per layer. 1.0 is the textbook setting; the output
  # layer is normalized too, so a strictly 1-Lipschitz stack can underfit a
  # standardized target -- raise this (1.5-3) before concluding the constraint
  # itself is wrong.
  dyn_spectral_norm_coef: float = 1.0
  # Power iterations used to estimate each layer's spectral norm. The estimate
  # is recomputed from scratch on every forward pass (no persistent iteration
  # vector, so a saved model predicts exactly as it trained), which is why this
  # is 10 rather than the 1 that state-carrying implementations use.
  dyn_spectral_norm_iters: int = 10
  # Adam learning-rate schedule. 'const' is the historical default (a fixed
  # dyn_learning_rate for the whole fit). 'cosine' warms up over
  # dyn_lr_warmup_frac of the run, then decays to dyn_lr_end_frac *
  # dyn_learning_rate -- which removes the late-training gradient noise floor
  # that leaves a constant-LR fit oscillating around its own optimum.
  dyn_lr_schedule: str = 'const'
  dyn_lr_warmup_frac: float = 0.02
  dyn_lr_end_frac: float = 0.01
  # Load a previously fitted model from this path instead of fitting (''=fit).
  dyn_model_path: str = ''

  # --- Off-diagonal SUPERVISION of the dynamics head (the negative dataset) --
  # Path to an episode .npz of FAILURE trajectories (obs/act/lengths, the same
  # schema as offline_dataset; the candidate files under
  # artifacts/v6_failneg/candidates_*/ are exactly this). '' leaves the
  # off-diagonal unsupervised, which is the historical behaviour.
  #
  # What it is for. The dynamics head reads concat(s, a, a'). The frozen
  # POSITIVE dataset is factual on every row (a' = a), so it pins the head
  # down only on the diagonal, and the off-diagonal was therefore the
  # network's initialization rather than anything a gradient produced -- the
  # identifiability warning in crl/causal_transition_model.py. This field is
  # what removes that warning: on every fit step a minibatch of NEGATIVE
  # transitions (s, a, s') is drawn, a counterfactual a' != a is drawn for it
  # (see dyn_off_diagonal_train_action) and the head is regressed
  # f(s, a, a') -> s' on it. So the two blocks now mean different things by
  # construction: a' = a returns the expert's own next state, a' far from a
  # returns the next state the failure data recorded.
  #
  # Read the file's obs width as obs_dim + anything: only the leading obs_dim
  # columns (the learner STATE) are used, so a negative collection stored
  # under an older goal contract (58-wide V6 candidates against a 31-wide _gxy
  # positive dataset) is loaded without conversion. It is NEVER added to the
  # frozen replay buffer -- the offline contract is untouched; it reaches the
  # supervised fit only.
  #
  # It needs dyn_next_action_input=True: with no a' block there is no
  # off-diagonal to supervise, and the fit raises rather than silently
  # training the diagonal twice.
  dyn_negative_dataset: str = ''
  # Weight of the off-diagonal MSE in the joint loss
  #   dyn_diag_mse + dyn_off_diagonal_coef * dyn_offdiag_mse
  #                + dyn_policy_coef * policy_loss.
  # Both dynamics terms are MSEs of the same standardized target, so 1.0 makes
  # a negative row count exactly as much as a positive one; the two minibatches
  # are the same size, so this is also the ratio of the two datasets' influence
  # regardless of how many negative episodes were collected.
  dyn_off_diagonal_coef: float = 1.0
  # Where the training a' comes from on the negative minibatch. It is redrawn
  # at EVERY step, so a negative transition is seen with many different
  # counterfactual actions over the fit rather than with one fixed partner.
  #   'uniform' -- drawn uniformly from the per-dimension action box measured
  #     on the positive train split, which is the actuator's own range and the
  #     widest coverage of the a' block available. Matches
  #     dyn_off_diagonal_action='uniform' at sample time.
  #   'shuffle' -- another negative row's stored action (a random non-self row
  #     of the same minibatch), which keeps a' on the action manifold at the
  #     cost of covering the block less evenly.
  # Whichever is used, a' is independent of s' given (s, a), so what the
  # off-diagonal learns is E[s' | s, a] over the NEGATIVE data marginalized
  # over a' -- a second dynamics branch selected by a' being unlike a, not a
  # response to the specific a'. That is the intended object here; do not read
  # an off-diagonal prediction as "what a' would have done".
  dyn_off_diagonal_train_action: str = 'uniform'
  # Fraction of NEGATIVE episodes held out for the off-diagonal validation
  # MSE, split by episode exactly as dyn_holdout_frac is on the positive side.
  dyn_negative_holdout_frac: float = 0.1

  # --- Done head of the causal transition model ---------------------------
  # Path to an npz of TERMINAL states (`states` [M, obs_dim], optional
  # `outcome` [M] 1 = success / 0 = failure), e.g. the output of
  # scripts/build_v6_done_states.py. '' = no done head, rollouts always run
  # dyn_rollout_steps. When set, a classifier P(done | s) on the learner state
  # is fitted against the non-terminal states of the positive and negative
  # data, and every rollout stops a row at the first generated state it flags
  # (that state becomes the row's last valid observation).
  dyn_done_dataset: str = ''
  dyn_done_hidden_layer_sizes: Tuple[int, ...] = (256, 256)
  # 0 = same as dyn_train_steps.
  dyn_done_train_steps: int = 0
  # Sample-time: P(done | s) >= this stops the row. The head is trained on
  # balanced batches, so 0.5 is the likelihood-ratio-1 point, not the natural
  # posterior; read done_val_fpr in the fit report before lowering it.
  dyn_done_threshold: float = 0.5

  # --- Blind expert-policy head (the second half of CausalTransitionModel) ---
  # Fitted jointly with the dynamics head by
  # crl.causal_transition_model.fit_causal: pi_phi(s_t) -> a_t on the SAME
  # minibatches. Its input is the learner state the agent sees, which does NOT
  # contain the rockfall latent the sighted teacher that collected the data
  # read at the band mouth -- so the head measures how much of the expert's
  # behaviour is reproducible without the hidden trap status.
  dyn_policy_hidden_layer_sizes: Tuple[int, ...] = (512, 512)
  # Weight of the policy loss in `dyn_mse + coef * policy_mse`. Both terms are
  # O(1) here (normalized state target, actions in [-1, 1]). Set to 0.0 to fit
  # the dynamics head alone, which reproduces `fit` exactly.
  dyn_policy_coef: float = 1.0
  # True: the head emits pre-squash logits and the action is a tanh mapped onto
  # the action box measured on the train split, so predictions are always
  # admissible. False: a linear head regresses standardized actions and clips
  # at predict time -- worth trying on this dataset, where a third of the
  # expert's torques sit exactly on the +-1 limit and a tanh head's gradient
  # there vanishes. (That "a third" is the v2.1 PILOT's saturation rate; on the
  # V5 far05 dataset it is 0.11% of action coordinates.) Read only by
  # dyn_policy_head='deterministic'; the mixture head models the action box by
  # CENSORING instead (see below), which stays defined on a saturated torque
  # where a squash's log-density is -inf.
  dyn_policy_squash: bool = True
  # Which blind expert head to fit.
  #   'mdn'           -- a mixture density network (the default): the head
  #     emits dyn_policy_components weights, means and log-sigmas and is
  #     fitted by maximum likelihood. This is the head the sampler needs,
  #     because p(a|s) here is the SIGHTED teacher's action distribution
  #     marginalized over the hidden rockfall latent and is therefore
  #     multi-modal by construction -- and an MSE regression on a multi-modal
  #     conditional returns the conditional MEAN, which at the band mouth is
  #     an average of "go straight" and "detour": a torque that does neither.
  #   'deterministic' -- the historical MSE regression, kept so the two can be
  #     compared on the same fit and so every pre-MDN config refits unchanged.
  dyn_policy_head: str = 'mdn'
  # k, the number of mixture components, read only by dyn_policy_head='mdn'.
  # Three is the number of branches the rockfall teacher has at the band mouth,
  # but note that mixture components are NOT identified with latent states --
  # they split and merge freely -- so k is chosen by held-out likelihood and
  # k=1 is the honest control: if it wins, the multi-modality is not in the
  # actions. Reported per fit as policy_val_nll.
  dyn_policy_components: int = 3
  # Floor and ceiling on the per-component, per-dimension log standard
  # deviation. The floor is what stops a component collapsing onto a single
  # data point, where the likelihood diverges and the fit is over; the ceiling
  # keeps a dead component from drifting to a flat prior the optimizer cannot
  # come back from. Applied by clipping the network's log-sigma output, so the
  # gradient is exactly zero outside the range rather than merely small.
  dyn_policy_log_sigma_min: float = -5.0
  dyn_policy_log_sigma_max: float = 2.0
  # Standard deviation of the spread applied to the mixture head's OUTPUT-LAYER
  # BIAS for the component means at init, in standardized action units (so 1.0
  # is one std of the dataset's action distribution). It exists because Haiku
  # initializes every bias to zero, which starts all k component means on top
  # of each other; the components then differ only through their random weight
  # rows, and a mixture whose components start identical is slow to separate
  # and can fail to. Set to 0.0 to reproduce the bare Haiku init.
  dyn_policy_init_spread: float = 1.0

  # --- Sample-time entropy of the expert head (does NOT touch the fit) ------
  # Both apply ONLY when actions are drawn from the mixture -- sample(),
  # sample_other_component() and the coin in diagonal_prob(). The likelihood,
  # the responsibilities, the component means and mode_action() are untouched,
  # so every fit diagnostic still reports the fitted head and these two say
  # what the SAMPLER does with it. They are therefore sweepable without
  # refitting: load a pickle and set the fields on model.policy.
  #
  # Why they exist. The fitted head on the V7 far30 rung is not overfitted --
  # train NLL -8.124 vs val -8.084, a 0.04-nat gap -- it is over-CONCENTRATED:
  # its weighted mean sigma is 0.098 raw torque while its own mode sits 0.210
  # from the stored action, about 2x more confident than its residual
  # justifies. A sharper expert than the data supports makes the generated
  # rollouts a narrow tube, which is not what the relabeler wants to draw
  # goals from.
  #
  # Multiplies every component's sigma at draw time. 1.0 is the fitted head.
  # ~2.0 puts the draw's spread level with the head's own mode error, which is
  # the calibrated setting; above that the draws leave the action manifold.
  dyn_policy_sample_sigma_scale: float = 1.0
  # Temperature on the mixture WEIGHT logits at draw time: log_w / T,
  # renormalized. T > 1 flattens the branch distribution, which raises the
  # branch entropy and so LOWERS the factual-diagonal coin sum_i w_i(s)**2 --
  # i.e. it makes the counterfactual off-diagonal fire more often.
  #
  # Raise this one with care, and read the coin-vs-position diagnostic first.
  # On V7 far30 the head's branch entropy is NOT uniformly collapsed: it
  # spends what it has on the teacher's DETOUR coin, which is
  # latent-independent and therefore not the confounder ETT is about. At the
  # fork the head reproduces that coin well (implied P(detour) 0.324 at t=50
  # against a true 0.300, and independent of the state as it must be), and a
  # temperature above 1 flattens that CORRECT 0.30 toward 0.5. The collapse
  # that matters is inside the two hazard bands, where the rockfall latent
  # decides wait-vs-go and the coin is 0.86 with 0.28 nats -- no better than
  # the open corridor, where there is nothing to decide. A uniform temperature
  # cannot tell those two apart; raising dyn_policy_components so the hazard
  # decision gets branches of its own can.
  dyn_policy_weight_temperature: float = 1.0

  # --- Counterfactual (off-diagonal) branch of the dynamics head ----------
  # Read by CausalTransitionModel.sample() and .rollout(). The dynamics head
  # takes concat(s, a, a') with a' the COUNTERFACTUAL action, and every step
  # emits either the FACTUAL diagonal f(s, a, a) or the counterfactual
  # off-diagonal f(s, a, a'):
  #   'none'    -- the diagonal only, which is the only query the fit ever
  #     supervised (a' = a on every stored transition).
  #   'mixture' -- the mixture head's own coin. With weights pi(s) the
  #     diagonal is taken with probability sum_i pi_i(s)^2 and the
  #     off-diagonal otherwise. That probability is the chance two independent
  #     draws of the hidden confounder land in the SAME branch, so the
  #     counterfactual fires exactly in proportion to how much the confounder
  #     matters at this state: at an unconfounded state one weight is ~1, the
  #     sum of squares is ~1 and it never fires; at a state where all k
  #     branches are equally likely it fires with probability 1 - 1/k. Unlike
  #     the discriminator coin it replaced, there is no temperature to tune.
  #   'agent'   -- the one the augmentation path uses. It runs the SAME
  #     sum_i pi_i(s)^2 coin as 'mixture' and differs only in where the
  #     counterfactual action comes from: with that probability the step is
  #     the diagonal f(s, a, a) on the blind expert head's own draw a, and
  #     otherwise it is f(s, a, a') with a' the action the CURRENT contrastive
  #     agent would take at the same state (its policy network run on
  #     concat(s, goal)), which is the failure-data branch that
  #     dyn_negative_dataset supervised. The coin is ANDed with actual
  #     agent/expert disagreement, since a step where the learner already
  #     plays the expert's action is the diagonal however the coin fell.
  #     Because a' moves with the learner, the generated trajectories go stale
  #     as it trains, which is what dyn_augment_refresh_every is for.
  # WARNING: the off-diagonal map is identified only when the fit actually saw
  # it -- i.e. when dyn_negative_dataset was set. Without one, no gradient has
  # ever supervised a' != a and those rows are the network's initialization,
  # not physics; the fitted artifact records which of the two it is
  # (off_diagonal_trained) and the run warns when a sampler asks for an
  # off-diagonal the fit never trained.
  dyn_off_diagonal: str = 'none'
  # Where a' comes from on the rows the coin sends off-diagonal.
  #   'uniform' -- drawn uniformly from the per-dimension action box measured
  #     on the train split. Off the action manifold, so those rows extrapolate
  #     twice over: an unidentified map at an input the data never visits.
  #   'mixture' -- drawn from a DIFFERENT component of the mixture head than
  #     the one the factual action came from, which keeps a' on the action
  #     manifold and gives the query a reading: "what if the confounder had
  #     been branch j instead of branch i". Needs dyn_policy_head='mdn' and
  #     k >= 2.
  # Ignored by dyn_off_diagonal='agent', which takes a' from the learner's own
  # policy rather than drawing one.
  # Whichever the off-diagonal training set draws a' from
  # (dyn_off_diagonal_train_action), the sampler has to be switched to match
  # it or the fit and the sampler disagree.
  dyn_off_diagonal_action: str = 'uniform'

  # --- Model-based rollouts (CausalTransitionModel.rollout) ---
  # T: how many steps the two fitted heads are iterated for when generating a
  # trajectory -- a_t = pi_phi(s_t) fed into s_{t+1} = f_theta(s_t, a_t),
  # started at the INITIAL state of a real dataset episode. A rollout inherits
  # nothing else from that episode: no real prefix, and never a state from
  # partway through one, so T is the full length of the generated trajectory
  # and every transition in it is the model's.
  # Why this needs to be a trajectory at all: the contrastive learner does not
  # consume transition pairs. crl/replay.py relabels the goal to a FUTURE state
  # drawn from the same trajectory with probability ~ discount**(j - i), so a
  # generated episode is only usable if it has an internal future to relabel
  # from.
  # The cost is compounding: each step feeds the policy head its own predicted
  # state, so error grows with T and nothing holds the rollout on the data
  # manifold. The default is set from the MEASURED closed-loop drift of the V5
  # far05 fit (512 rollouts off held-out episodes, no spectral norm), where the
  # full-state error relative to how far the real trajectory moved over the
  # same horizon runs 0.19 at T=1, 0.33 at T=5, 0.51 at T=7 and 0.82 at T=10,
  # and the torso-xy error runs 0.004 / 0.058 / 0.097 / 0.149 against a corridor
  # about 9 units long. Measured AT T=20 (1024 rollouts, held-out episodes)
  # those become 0.94 and 0.444.
  # Those numbers were measured on rollouts that branched off a real prefix.
  # They are still the right warning about compounding -- at T=20 the
  # full-state error is already 94% of the real displacement, so only the
  # torso-xy columns (the goal coordinates) stay anywhere close -- but they
  # are no longer measured on the rollouts this default produces, which start
  # at the reset state instead. Re-measure on the current shape before
  # reading them as the drift of a generated episode.
  # 70 is the DEFAULT and the intended ceiling: the whole row is generated
  # now, so T also sets how long a trajectory the relabeler gets to draw a
  # future from, and a short T (the old 5 or 20) leaves a generated episode
  # with almost no internal future while still being far short of the
  # ~100-step mean relabeling offset at discount=0.99.
  # Re-measure before changing it: RolloutTrajectories.drift() reports error
  # per horizon against the real episode from the same initial state and
  # scripts/eval_causal_transition_model.py plots the curve. Refitting with
  # dyn_spectral_norm=True is the lever that should extend the usable horizon,
  # since it is what bounds the per-step Lipschitz constant the compounding
  # runs on.
  dyn_rollout_steps: int = 70
  # Where a generated trajectory STARTS (crl/ETT_train.py augmentation path).
  #   'initial'     -- v1 (crl/causal_transition_model.py): the source
  #                    episode's reset state, no real prefix.
  #   'route_split' -- v2 (crl/causal_transition_model_v2.py): a detour
  #                    episode starts at its first state with
  #                    x > dyn_rollout_detour_min_x and
  #                    y < dyn_rollout_detour_max_y, a shortcut episode at its
  #                    first state inside hazard zone
  #                    dyn_rollout_shortcut_zone; the real prefix up to
  #                    that state is kept in the row, so a row holds
  #                    t0 + dyn_rollout_steps + 1 observations. Episodes with
  #                    no such state (or labelled neither route) are skipped.
  dyn_rollout_start: str = 'initial'
  dyn_rollout_detour_min_x: float = 20.0
  dyn_rollout_detour_max_y: float = 7.5
  # V6 hazard zone (1 or 2) whose first entered state starts a shortcut row.
  dyn_rollout_shortcut_zone: int = 2
  # Per-episode route labels for 'route_split': a dataset *_sidecar.npz
  # (route_realized). '' derives it from offline_dataset
  # (<stem without _gxy>_sidecar.npz); if that file is missing, episodes are
  # labelled by geometry (shortcut iff the path enters that hazard zone).
  dyn_rollout_route_labels: str = ''

  # Clip every GENERATED state into the environment's feasible box before it
  # is stored and before it is fed back into the next step (see
  # crl.causal_transition_model.CausalTransitionModel.rollout and
  # crl.d4rl_ant.ant_state_bounds, which derives the box for the ant maze
  # family from the live eval env: the maze walls for xy, the floor and the
  # wall height for z, the ant.xml joint stops widened by the solver's give
  # for the eight leg joints, and an explicit envelope for the quaternion and
  # the fourteen velocities, which nothing in the model bounds).
  #
  # WHY IT IS ON. Nothing in the fit stops a chain fed its own output from
  # leaving the state space -- the ant walks through a wall, the torso sinks
  # under the floor, a joint bends past its stop, a velocity runs away -- and
  # every later step inherits that state. The relabeler then hands the critic
  # future goals that no reset of this environment could produce, which is
  # worse than a merely inaccurate rollout: it is a goal distribution the
  # policy is never evaluated on. The box was checked against the frozen V6
  # dataset and does not bind on a single one of its 297,878 real states, so
  # turning this on cannot clip anything the environment itself produces.
  #
  # It is NOT a fix for a bad fit. Pinning a divergent state to the wall does
  # not make it data; the clip only stops one bad step from compounding.
  # RolloutTrajectories.clipped_frac reports how often it bound, and a large
  # value is a reason to shorten dyn_rollout_steps or refit (e.g. with
  # dyn_spectral_norm), not to widen the box.
  #
  # False reproduces every rollout generated before this existed, bit for bit:
  # no box is attached to the model and clip_state is the identity. It is also
  # what a NON-ant benchmark gets automatically -- crl/ETT_train.py prints a
  # note and carries on unclipped when the env has no box to offer.
  dyn_clip_states: bool = True

  # Project the xy of every GENERATED state onto the maze's traversable cells
  # (see crl.causal_transition_model.maze_free_cells and
  # TransitionModel.clip_xy), which is the part of the feasibility constraint
  # the box above cannot express.
  #
  # WHY IT IS SEPARATE FROM dyn_clip_states. The box is a product of
  # intervals, so the xy it enforces is the maze's BOUNDING box: on the
  # U-maze it contains the central bar, on V6/V7 it contains the whole ring
  # centre. A generated chain can therefore cut straight through the block in
  # the middle of the map and never once be "out of bounds". On a two-route
  # benchmark that is the one excursion that matters, because a shortcut
  # between the two routes is precisely the thing the benchmark is asking the
  # policy to choose between, and a relabeled goal taken from a state inside
  # a wall is a goal no reset of this environment could ever produce.
  #
  # WHAT IT DOES AND DOES NOT DO. It is the exact Euclidean projection onto
  # the union of the open cells, so it is the identity on every state the
  # environment itself can produce and moves only the ones inside a wall. It
  # corrects xy alone: the velocities stay as the dynamics head produced
  # them, so the next step is still the model's own answer. Like the box it
  # is not a fix for a bad fit -- RolloutTrajectories.clipped_frac counts
  # both constraints together, and a large value is a reason to shorten
  # dyn_rollout_steps or refit.
  #
  # Rocks and gates that open and close during an episode are NOT walls and
  # are not part of the floor plan; only the static maze geometry is.
  #
  # False reproduces every rollout generated before this existed, and is also
  # what a non-ant benchmark gets automatically -- crl/ETT_train.py prints a
  # note and carries on with the box alone when the env has no floor plan to
  # offer.
  dyn_clip_walls: bool = True

  # --- Offline dataset augmentation with model rollouts (OFF by default) ---
  # Fraction of every contrastive minibatch drawn from MODEL-GENERATED
  # trajectories instead of the real dataset. 0.0 (default) means the
  # augmentation buffer is never built and the sampling stream is bit-for-bit
  # what it was before this feature existed, so every existing offline run
  # reproduces exactly.
  #
  # Generated data CANNOT go into the offline buffer. That buffer is sized
  # exactly to the dataset, frozen (crl/offline_audit.py gate G8), and its
  # content hash is asserted after the transition-model fit, at every eval and
  # at the end of the run -- the offline contract is that the dataset never
  # grows or changes. So rollouts go into a SEPARATE TrajectoryBuffer and the
  # two are mixed per minibatch: `dyn_augment_frac * batch_size` rows from the
  # generated buffer, the rest from the frozen one. Both buffers relabel goals
  # the same way, which is the whole reason rollouts have to be trajectories.
  #
  # Read the drift numbers on dyn_rollout_steps before turning this on. At
  # T=20 a generated state is off by 94% of the real displacement in the full
  # state (0.444 in torso xy), so this trains the critic on states that are
  # dynamically plausible but not where the real ant went, and it is an
  # experiment rather than a free win. Rollouts start at a reset state and
  # carry no real prefix, so the whole of this buffer is model output: there
  # is no real-data anchor inside a generated trajectory beyond its first
  # observation.
  dyn_augment_frac: float = 0.0
  # How many trajectories to generate. Each contributes exactly
  # dyn_rollout_steps transitions.
  dyn_augment_episodes: int = 2000
  # How often the augmentation buffer is REGENERATED, in outer training-loop
  # iterations (one iteration = one episode's worth of gradient clock,
  # max_episode_steps steps). 1 (default) regenerates it at the top of every
  # iteration, which is what dyn_off_diagonal='agent' requires to mean
  # anything: a' is the CURRENT agent's action, so a buffer generated once
  # before the first gradient step is a buffer of counterfactuals against a
  # randomly initialized policy for the rest of the run. Raise it to trade
  # freshness for wall time -- each regeneration costs dyn_augment_episodes
  # rollouts of dyn_rollout_steps steps plus the buffer rebuild. The frozen
  # dataset is untouched by any of this; only the second buffer is rebuilt.
  dyn_augment_refresh_every: int = 1

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

  # --- Task-goal negative term (OFF by default) -----------------------------
  # Weight on an extra critic term that scores f(s, a, g_task) against the
  # anchor episode's OWN task goal and labels it with the episode's outcome:
  # 1 if that episode reached the goal, 0 if it did not. It supplies the one
  # label hindsight relabeling structurally cannot -- a failure trajectory
  # never has the task goal among its future states, so the contrastive loss
  # alone never pushes f(s, a, g_task) down for a transition that led to a
  # death. 0.0 => the branch is skipped and the loss is byte-identical to the
  # baseline. Requires the Monte-Carlo NCE critic and a buffer carrying
  # per-episode labels (see crl/replay.py add_episode).
  task_goal_coef: float = 0.0
  # Goal-coordinate distance at which an offline episode counts as having
  # reached its task goal, used to derive the outcome labels from the dataset
  # itself rather than from a field the dataset may not carry. 0.5 is the
  # AntMaze SUCCESS_DIST the environments already score with.
  task_goal_success_dist: float = 0.5

  # --- Death-trajectory hindsight negatives (OFF by default) ----------------
  # Weight on an extra critic term built ONLY from the failure ("death")
  # episodes of the offline dataset. Those episodes are relabeled exactly the
  # way every other anchor is -- draw an episode, an anchor time i, and a
  # future time j > i with probability proportional to discount**(j-i) -- but
  # the resulting pair is scored with label 0 instead of the 1 that hindsight
  # would give it:
  #
  #     L += death_neg_coef * mean_k BCE( f(s_k, a_k, g_k), 0 )
  #
  # The claim is that "this action took me to where I later died" is not a
  # competence to reward. It differs from fail_neg_alpha, which pairs banked
  # death STATES with arbitrary anchors: there the only discriminating input
  # is the goal, and in a 2-dim XY goal a death state is indistinguishable
  # from a safe crossing of the same band (measured AUC 0.505). Here the
  # anchor and the goal come from the SAME death trajectory, so the margin
  # has to be carried by sa_repr(s, a).
  #
  # The death set is read off the data, not off the collector: it is the
  # episodes the buffer's per-episode labels mark as not having reached their
  # task goal (see task_goal_success_dist). 0.0 => the branch is skipped and
  # the critic loss and its gradients are byte-identical to the baseline.
  # Requires the Monte-Carlo NCE critic and a labeled offline buffer.
  death_neg_coef: float = 0.0
  # Optional SEPARATE offline file for the death pool. Empty => the pool is
  # the not-reached episodes of offline_dataset itself (they are then also
  # ordinary anchors, so they reach the ordinary NCE and the actor's BC).
  # Set => the pool is the not-reached episodes of THIS file, loaded into its
  # own frozen buffer that only the death term draws from; offline_dataset
  # can then be positives-only and BC never sees a death trajectory.
  death_dataset: str = ''

  # --- Generated-trajectory negatives (crl/ETT_train_v2.py; OFF by default) --
  # Weight on an extra critic term that separates the real positive dataset
  # from the causal transition model's generated trajectories. Both batches
  # are relabeled by the ordinary geometric-future rule inside their own
  # trajectories, and each row is scored against its own relabeled goal:
  #
  #     L += gen_neg_coef * [ mean_real BCE( f(s, a, g), 1 )
  #                         + mean_gen  BCE( f(s, a, g), 0 ) ]
  #
  # The real side reuses the diagonal of the ordinary contrastive batch. In
  # ETT_train_v2 the generated batch reaches ONLY this term: the B x B
  # contrastive loss, the actor loss and the BC loss all read the real batch
  # alone. 0.0 => the branch is skipped and the loss is byte-identical to the
  # baseline. Requires the Monte-Carlo NCE critic.
  gen_neg_coef: float = 0.0

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
  # crl/ETT_train*.py only: number of independently seeded eval envs, each run
  # for eval_episodes. Env k is seeded config.seed + 10_000 + 100_003 * k, so
  # env 0 is the single eval env of earlier runs; 'success' pools all
  # episodes and success_seed<k> / success_std give the per-seed spread.
  eval_num_seeds: int = 3
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
