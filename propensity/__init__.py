"""Continuous-action propensity / behavior-support module (staged prototype).

Standalone package for the continuous-action analogue of the discrete causal
propensity term ``P(A = a | S = s)``. For continuous actions that point mass is
zero, so the plan (following Junzhe Zhang et al.'s Causal Flow Q-Learning) is to
learn a behavior-action distribution ``mu(a | s)`` from the fixed offline
dataset and later compare behavior samples against target-policy samples with a
discriminator ``D(s, a)``.

``D(s, a)`` is NOT the literal continuous propensity density. It is a learned
behavior-support / distributional-overlap surrogate.

Staging (only Stage 1 exists in this package today):

  Stage 1  offline dataset interface     <- ``propensity.dataset`` (this commit)
  Stage 2  behavior flow model           (not implemented)
  Stage 3  discriminator + diagnostics   (not implemented)
  Stage 4  integration into causal CRL   (not implemented)

Nothing here touches the CRL critic, the CRL losses, or the target policy. The
package is pure numpy: it deliberately pulls in neither JAX/Haiku/Optax nor any
new third-party dependency. Read-only reuse of ``crl.offline_audit`` gives it
the repo's single source of truth for the frozen-dataset key vocabulary.
"""

__all__ = ['BehaviorDataset', 'BehaviorBatch']


def __getattr__(name):
  """Lazy re-export (PEP 562).

  Eagerly importing ``propensity.dataset`` here would make
  ``python -m propensity.dataset`` load the module twice (runpy warns about it),
  so the symbols are resolved on first access instead."""
  if name in __all__:
    from propensity import dataset as _dataset
    return getattr(_dataset, name)
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
