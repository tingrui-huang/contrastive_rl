# Does the current implementation still test the intended ETT approach?

**Only a restricted subproblem.** The intended shared-generator, two-active-loss
experiment has become optimization of 32 corrections around an immutable fitted
distribution. Its diagonal term cannot train anything. The resulting failures
do not test, much less falsify, joint diagonal fitting and pessimistic transition
selection under an action-Lipschitz constraint.

Reviewed commit **78151671affe7890122cbca34537c8954c5eb93b**, its predecessor
experiments, and their implementations. This design adds an isolated executable
reference protocol; no production model, checkpoint, actor, critic, nominal
model, or historical artifact changes. Only saved-data preparation and **45
deterministic component outputs** were evaluated. No training or rollout ran.

## Source authority and introduced restrictions

No original research-teacher recording or transcript was found in the repository
search or supplied conversation. “Teacher” collectors in this repository are
behavior policies, not that recording. The shared-generator formulation here
comes from **the user's retelling and present instruction**, not an authenticated
teacher quotation. The user was asked for the original source; it is not assumed
to contain additional requirements.

The stated requirements are execution x versus natural x_prime, diagonal
observational fitting, a pessimistic second loss, and an explicit action bound.
The prescribed L=1 and the current samplewise, full-all-action-pairs convention
are retained. Neither L nor off-diagonal causality is identified by diagonal data.

The repository added the following restrictions:

* **Exact freezing of an approximate diagonal law.** Fitting
  `Law(G_theta(s,a,a,z))` to observed successors is a statistical objective.
  Forcing `G_theta(s,a,a,z)=A_frozen(s,a,z)` for every coupled sample is a much
  stronger engineering choice. It preserves approximation errors as well as
  useful fit and makes the current L_diag constant.
* **Only 32 trainable coefficients**, eight fixed context gates, and a signed
  affine response. These were chosen for tractable parameter-perturbation search;
  they are not the shared neural generator described by the user.
* **Anchor-independent correction** `M(s)(x-x_prime)`. In particular it cannot
  adapt to the diagonal draw's moving component or random displacement.
* **Fixed convex projections.** These give a convenient valid action-bound proof,
  but impose additional response restrictions. The fork audit proved some were
  unnecessary; the optional segment removed particular restrictions, not all of
  them. Nonconvex or action-switched projection is not an automatic replacement.
* **Probability-one visible absorption as a gate.** This was the death audit's
  operational interpretation of “absorbing consequences,” not a demonstrated
  teacher requirement or a necessary prerequisite for every one-step ETT test.

The earlier scalar shared-network study did train two active losses, but used
supplied synthetic targets and initially violated L. The constrained spline
studies established different limited facts and exposed diagonal-fit tradeoffs.
The later convex specification explicitly acknowledges that it abandoned shared
diagonal training. Its return reductions, mixed geometry comparison, and two
null CRL integrations concern that restricted class. They do not locate a sole
failure of the second-loss idea. See [sources](SOURCES.md).

## Corrected reading of the death audit

7815167 proves that at its inspected hazard-interior conditions, a fixed
anchor-independent correction plus the upper box cannot collapse the diagonal's
positive moving support to the current interior position with probability one,
for any of the 32 coefficients. Exact diagonal preservation supplies a separate
necessary-condition obstruction to certain absorption. This is a scoped
class result, not merely failed optimization.

It **does not** prove absence of a persistent dead component in an alive/dead
mixture, inability of a jointly fitted generator to represent one, inadequacy of
F4 everywhere, or incompatibility of death with L=1 in every formulation. A
stationary atom is also not a certificate of persistent death. The earlier F4
study found strong established-death information (97.52% probe recall at age
3+) but weak immediate-postfatal identification (4.65% recall at age 0). It found
no exact contradictory histories in its finite dataset; that does not prove
population identifiability. Those are historical probe/diagonal results, not new
measurements of the current model.

The long-rollout loss creates additional obligations: appropriate propagation
of hidden-context uncertainty, temporally consistent absorption, meaningful
nominal actions at generated histories, and control of recursive distribution
shift. Repeatedly drawing a one-step mixture need not preserve a persistent
component. A one-step conditional objective needs a well-defined successor law,
diagonal fit, the declared within-context action coupling, and a justified
successor score. It does **not** first need a complete absorbing simulator or an
assumption that F4 is a sufficient Markov state for 50-step composition.

Accordingly, the previous recommendation to require a persistent latent-death
construction before *any* further pessimistic search was too broad. It remains
relevant to death-rate claims and long-rollout use; it is not the next mandatory
implementation step for a one-step diagnostic.

## One recommendation

Run one **local, shared-generator, immediate-task-reward experiment** near the
known reward boundary. Its exact objective is

    min_theta E_D ES(Law(G_theta(s,a,a,Z)), y_observed)
              + 0.01 E_(s,a)~D, x~Uniform([-1,1]^2), Z
                         1[||G_theta(s,x,a,Z)_XY - (8.5,3.5)||_2 < 2].

Use one trainable stochastic network for both evaluations; x_prime is the recorded
natural action a. The actor and nominal model are preserved and unused in this
conditional experiment. This deliberately tests uniform execution-action
coverage, not fixed-actor performance or nominal-marginalized rollouts. The
alternative arm sets the second weight to zero and still trains the same network.

“Bad” here means **missing the next task reward**, whose ordering follows directly
from the existing task definition. It does not mean death, a low critic logit,
distance from a bank, or reduced natural long-run return. At hazard locations,
immediate reward often ties death and survival; this objective has **no justified
death-specific preference** there. If death selection remains the required
outcome, the specific missing ingredient is a supported observable badness or
continuation criterion that distinguishes those outcomes under relevant
conditioning. The rejected critic and partially transferable readouts do not
currently supply it. This proposal is ready only as a local reward-pessimism
diagnostic, not a death-finding experiment.

The [protocol](PROTOCOL.md) specifies the 2,123-parameter shared generator, full
L=1 proof, live diagonal fitting, fixed budget and separate acceptance criteria.
An analytic all-action witness can lower immediate reward while leaving its own
diagonal unchanged; a deterministic implementation check passes. It is an
existence example, not an observed off-diagonal target. Preparation found 275
training episodes/1,042 rows and 26 historically held-out episodes/66 rows.
Small support and high-dimensional perturbation variance may make the bounded
result inconclusive; no budget expansion is authorized.

**Next step:** implement/review the isolated reference protocol as the proposed
experiment, then run it only under a separate execution instruction. Keep death
selection and downstream CRL outside its acceptance claim.
