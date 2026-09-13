# Death/absorption investigation: Stage 1 stops at an unvalidated onset construction

**The requested major-cause claim remains unresolved.** Existing evidence shows model motion and positive continuation from some simulator-confirmed dead contexts. It does not identify how much dangerous-action value error, or lack of native policy improvement, is caused by death modeling. This investigation did not certify an A/B comparison that repairs native death onset on generated branches while preserving an interpretable baseline. Following the requested gate, no Stage 2 rollouts or Stage 3 actor training ran.

## Scope and provenance

The main checkout is AntMaze at `f290056c1559baedd98d4e5cc79cad3532b321b0`. Work used the existing `.worktrees/pointmaze-p30` checkout on `feature/pointmaze-causal-transition`. Both its HEAD and `git ls-remote origin refs/heads/feature/pointmaze-causal-transition` were `f85a5f44b176120333a1de0f83d6d0d02a2b290e`; there were no newer PointMaze commits. Tracked PointMaze files were clean. Existing untracked artifacts were preserved, including locally generated E26 reports. The committed E26 JSON and report code were checked directly.

The user-specified experiment ledger was read in UTF-8. A byte-preserved copy, source hashes and verification accompany this report. [PROTOCOL.md](PROTOCOL.md) was written after reconnaissance, before the formal construction audit or any new numerical experiment; it is not claimed to precede the historical evidence. No external paper or additional environment was used.

## What is new, and which evidence was reused

- E02 and E10 establish the observability and inspected frozen-family limitations. F4 shifts three known frames after generating new XY. A saved native-dead realization does not imply every latent context with the same visible conditioning is dead. Neither F4 insufficiency in general nor incompatibility of every Lipschitz model follows.
- E15 already reports all 32 late outside-goal roots native-dead, but model motion and some goal-reaching from those contexts. Those roots do not answer an alive decision-making question. No new death detector or post-death sampling was run.
- E19's calibrated region-NCE and MC updates both reduced independent model return. This differs from the rejected old raw-logit critic. E20/E21 already tested native policy integration, including substantial actor changes in E21. E21's original actor scored about 8.24-10.00 in the four models versus 3.32 natively; that aggregate discrepancy is motivation, not a death-attributable quantity.
- E22-E24 already cover broader response search, higher-precision contrast error and sign agreement. E25's oracle uses a broader family. E26 corrects its optimizer-only inference and finds no consistent spectral advantage; small supported seed-0 decreases remain real. None is a death-repair control.

The new work is a **joint event/latent-conditioning and intervention audit**, rather than a repeat of frozen-anchor absorption capacity. It tests whether the baseline can be retained while an event on its generated branch acquires validated native semantics. The exact candidate definitions and analytic derivations are in [CONSTRUCTION.md](CONSTRUCTION.md).

## Intervention and attribution limits

A is the original rectangle/Frobenius learned kernel, using E26's two inherited `s0_k0_u24` / `s1_k0_u24` starting models, frozen actor and nominal. It has 16 inherited diagonal offsets and 32 response coordinates. Its generated death rate is **undefined**, not zero.

The closest B adds the same latent bookkeeping in both arms, but activates persistent fixed XY/zero reward only in B after a branch-local death event. Historical F4 frames continue shifting. This isolates persistence **conditional on a specified event law**; the missing piece is validating that event law as native onset under the generated landing and natural action x'.

The native teacher uses the current hidden mask to select x'. The learned kernel supplies a nominal marginal and an independently sampled learned anchor, but no joint mask/anchor/teacher-memo law. Fresh p=.30 masks independent of x' define an internally consistent synthetic killed model, explicitly specified in the appendix. That diagnostic can measure the effect of its stipulated killing rule; it has not verified native onset. Lower returns under that wrapper follow mechanically from deleting nonnegative reward tails and cannot verify the hypothesis.

A complete native simulator branch supplies real death events. However, replacing learned movement with native physics changes action response, motion, collision geometry and conditioning too. Copying its death label to a diverged model branch is invalid. A simulator transplant is also a new intervention requiring its own latent law. These are not all impossible diagnostics; they answer narrower or confounded questions unless the branch construction is validated.

Adding a dead component while keeping the live kernel unchanged generally changes the visible diagonal law. If q is posterior dead mass, diagonal consistency requires `K_A = q*delta_absorb + (1-q)*K_live`. Keeping `K_live=K_A` fails wherever q>0 and K_A has nonabsorbing mass. Compensating the live law may be possible when its emitted atom mass supports q, but changes another mechanism and needs a sequential latent construction. A common augmented architecture does not eliminate that confound. The old XY-only action-Lipschitz proof also does not certify the discontinuous death bit or multi-step augmented process.

**Needed to proceed:** an explicit sequential joint law connecting generated motion, current mask, persistent death and x', with an exact A control and declared diagonal/constraint changes; or a simulator-assisted construction with independent onset validation for the stated population and acknowledged family changes. The inspected artifacts do not provide this bridge. This is not a demand for offline identification, a universal impossibility theorem, or a ban on hybrid diagnostics.

## Execution and conclusions

Ran source/ledger/history review, analytic construction checks, input hashing and artifact verification. **New simulator transitions, model draws/emitter calls and all training updates: zero.** No repair was implemented or empirically verified. Stage 2 and Stage 3 did not run because the onset/attribution gate remained unclosed; no substitute detector, critic fit, sweep or actor loop ran.

Value-error changes, dangerous-versus-safe contrast changes, and native-policy changes are **not measured**. The fraction of the baseline discrepancy removed is **undefined**. H1 (major contribution to decision-relevant continuation errors) and H2 (material native-policy benefit) both remain unresolved, neither rejected nor established. The narrower historical post-death mismatch remains supported.

The protocol fixes prospective major-effect criteria: at least .02 normalized dangerous-action error reduction, at least 25% of a newly matched positive baseline gap removed, and at least .01 action-contrast error reduction; downstream benefit additionally requires .02 normalized native return and five percentage points less actual absorbing failure versus both controls. No new result met or failed these numerical criteria because no numerical phase ran. Their purpose is to prevent interpreting an arbitrary small or merely significant change as a major cause.

E27 records this stop and marks superseded recommendations explicitly. Historical artifacts, production code and checkpoints were preserved. No commit or push was made.
