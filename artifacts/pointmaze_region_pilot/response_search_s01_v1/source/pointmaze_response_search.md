# Bounded response-only ETT diagnosis

Base: a439486, feature/pointmaze-causal-transition. One experiment, no extensions.
Baselines: final critic_s0/critic_s1 from 817e429's update_reference_s01_v1.
For each baseline freeze the entire diagonal network AND its 16 head offsets,
nominal policy, original actor and normalization. Only 32 response coefficients
may change. Rectangle geometry, samplewise action-Lipschitz bound 1, F4 history,
next-XY reward radius 2, gamma=.95 and external horizon 50 remain unchanged.
No actor/BC changes, native simulation, route/geometry repeat or architecture sweep.

Three starts per baseline: its existing response; two independent Gaussian
8x2x2 arrays (seeds 160000001/160000002 shared between baseline pairs), with each
matrix rescaled to Frobenius norm .5. These are feasible nonzero coefficients,
not guaranteed emitted sensitivity. No sensitivity floor or forced transition.
Diagonal offsets are copied bitwise from the appropriate baseline in every case.

Exactly 24 updates per start, 144 total (versus three previously). On every
iteration collect four current-kernel paths from each of the existing 36 training
roots through its actual remaining horizon. Restore the corresponding prior
critic_s{s}_u2 averaged-positive region critic and its Adam state at every start.
Refresh 1,000 steps on the first iteration and 300 thereafter: 47,400 steps total.
Keep architecture, normalization, uniform row sampling, analytically averaged
positive NCE, 1:31 weighting, batch 256 and critic Adam lr=.003 unchanged.
Freeze critic and optimizer during the response update and diagnostic MC checks.

Minimize the existing surrogate, with Q decoded as in the region critic:
E[H*.95^t*(.05*r(next_state)+.95*Q_(h-1)(next_state,next_action))].
H is the root path's remaining external horizon; t is within-path time; h=H-t.
Uniform paths then uniform time; 32 state/action queries; four successor draws
and four frozen-actor next-action draws. Frozen nominal x_prime is independently
sampled. Reuse identical queries and random streams for candidate comparisons.
Estimate a response gradient using eight Gaussian antithetic directions, sigma=.1.
Adam lr=.05, beta=(.9,.999), eps=1e-8; cap each update's response L2 at .25, then
project each raw 2x2 block onto its unit Frobenius ball. Signed probes use the
same projection. No diagonal penalty or diagonal-fit rejection is needed because
the diagonal law is immutable. Apply each finite feasible optimizer step; no
outcome-driven retries, line searches, or early stopping. Stop after 24 updates
per start, or preserve and stop the whole attempt on a numerical/contract error.

Matched MC diagnostics at updates 1,12,24 of every start (18 comparisons):
compare the proposed response versus its pre-update response on a NEW independent
draw of 32 queries from that iteration's frozen visitation. Use eight successor
draws and four next-action draws. Critic and MC share exactly the same states,
executed actions, candidate first successors and frozen-actor next actions.
Both candidates then continue under the SAME frozen PRE-UPDATE transition model;
only the first transition uses the candidate. Pair all random streams. Q_0=0.
MC is diagnostic only: it never changes the gradient, step, or candidate pool.
The vectorized MC implementation computes 48 steps per row, masks every reward
and state update beyond h-1, and charges ALL generated slots including padding.
This is conservative computation accounting, not permission to lengthen episodes.
Save raw rewards, horizons, integrands and parameter vectors for independent audit.

Selection pool per baseline: all three starts at updates 0,12,24 (9 candidates,
including the original baseline). After all optimization finishes, score each
with 16 new full-kernel rollouts per TRAIN root. Pick the lowest mean normalized
return, with lexicographic candidate-name tie breaking. No MC diagnostic or final
evaluation sample selects it. Selection can pick the baseline or a random start.
Seal winners, full pool and selection samples before independent final evaluation.

Independent final evaluation per baseline: baseline, the two random starts, all
three final iterates, their three pre-final (update 23) models, and the selected
winner (deduplicate; at most 10 models). The final-versus-pre-final comparisons
directly test whether the last audited local improvement transfers to full use.
Use 64 fresh full-kernel rollouts per 36 existing held-out final-context roots,
and 128 fresh full 50-step paths from public START. Each evaluated model acts at
EVERY step. Samples are independent of training, selection and local MC checks.
Previously inspected roots are reused; fresh randomness is not a claim of new
root-distribution validation. Normalize returns by .05 as in 817e429; lower is
more pessimistic. Public-reset results are separate from continuation-root results.

Hard cap 6,000,000 newly computed model transitions, no reserve spending:
982,656 critic-refresh trajectory transitions; 331,776 optimizer first transitions;
1,778,688 matched-MC computed transitions including padding; 491,328 selection
transitions; at most 2,167,040 held-out final transitions + 128,000 public-reset
transitions; at most 40,448 diagonal/constraint checks. Upper total 5,919,936.
Exactly 47,400 critic optimizer steps, 144 response updates, zero actor/nominal
updates and zero native transitions. No new experiment after reporting.

Uncertainty: 2,000 paired bootstrap replicates, seed 166000000. Local comparisons
cluster query means by root; additionally report conditional simulation intervals
using paired successor differences (next-action draws averaged inside successor).
Report critic delta, MC delta and their gap, with raw-sign and resolved-opposite
sign counts. A resolved sign needs BOTH root and conditional-MC intervals on one
side of zero. Full-rollout differences use paired root bootstrap stratified by
the existing three root groups plus conditional MC intervals, as in 817e429.
Public-reset intervals resample paired rollouts. Report every prespecified check,
not only disagreements. Intervals are descriptive, not simultaneous; selection
uncertainty is addressed by the independent final sample, not a claim of global
optimality. No pooling the two baseline models into a population seed guarantee.

Verify hashes of base diagonal, nominal and actor, exact 16-offset equality at
every step, paired diagonal samples against the baseline at every proposal, and
final samplewise geometry/history/Lipschitz constraints. Record parameter norms
without imposing a minimum. Read-only audit recomputes MC targets and final metrics.

Decision rules: independently lower winner/baseline full returns in both models
supports insufficient response search as a contributor within this fixed family;
resolved opposite critic/MC directions support local continuation-estimation
error. Report both if they coexist. Favor continuation-estimation diagnosis next
if such disagreement accompanies search failure; favor local-to-full objective
alignment if MC-validated local decreases fail to improve full rollouts. Otherwise
say this bounded search is inconclusive. Even a successful search establishes
neither global worst-case recovery nor native-policy benefit. No native-policy
benefit is measured because actor and native environment are not trained/evaluated.
