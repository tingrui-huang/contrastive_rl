# Native benefit from the pessimistic ETT remains unresolved

**This bounded experiment does not establish that training against the learned
pessimistic ETT improves native performance.** Every primary return, success
and failure interval includes zero; this is not evidence of equivalence.

A stayed unchanged. B used the final fitted diagonal control with both action
channels equal to the executed action. C used the final critic-driven ETT
with independent frozen nominal actions. Two paired seeds received four rounds
of current-actor rollouts, averaged-NCE fitting, and 25 guarded decoded-Q actor
proposals per round. Native evaluation began only after all actors were sealed.

On 256 fresh paired native reset seeds, raw discounted return differences
(95% paired episode-bootstrap intervals) were:

- C versus A: seed 0 **+.1106 [-.0596, +.2862]**; seed 1 **+.0211 [-.0830, +.1410]**.
- C versus B: seed 0 **+.0771 [-.0348, +.2240]**; seed 1 **-.0587 [-.2076, +.0917]**.

A's mean return/success/failure were **3.3252 / 29.7% / 70.3%**. C reached
**3.4357 / 30.5% / 69.5%** in seed 0 and **3.3462 / 29.7% / 70.3%** in seed 1.
Success and absorbing failure are distinct evaluation endpoints.

**Independent model rollouts also show no resolved own-model gain.** B's
return changes were +.1813/+.2518; C's were -.0251/-.0035, all intervals
including zero. All five actors were evaluated on all four frozen models.
The models remain optimistic relative to native outcomes: A's model returns
ranged from 7.80 to 10.31, versus 3.33 natively. These are descriptive model
outputs, not a native oracle.

Only **37/400** actor proposals passed the fixed policy constraints. Final
state-bank KL ranged .0031-.0151; mean-action RMS changes were .0179-.0256.
Correctness tests verified the differentiable decoded-Q path, action gradients
against Torch, actor gradients against finite differences, and B's action
channel routing. Correct gradients do not imply accurate value estimates.

C inherits positive diagonal-fit degradation (.01497/.00810), although within
the prior .02 allowance. Limited adaptation, critic approximation and model
error remain possible explanations; this run does not isolate them. An ETT
trained against A is not asserted to remain worst-case for updated actors.

Completed **423,168/500,000 model transitions**, **64,000 native steps**,
10,800 critic steps, 16 passing tests and a passing stored-artifact audit.
No ETT retraining, extra sweep or budget extension followed.

[REPORT.md](REPORT.md) contains all native contrasts, cross-model intervals,
policy changes and limitations. [PROTOCOL.md](PROTOCOL.md) was sealed before
training and evaluation.
