# Execution-action response diagnostic

The authoritative bounded diagnostic is
[f4_return_s01_horizon50/REPORT.md](f4_return_s01_horizon50/REPORT.md).

It evaluates the frozen control and both return-search checkpoints from commit
`03f7686a2277c684d3801a6dcb7f24bd81ea622c`. No model or objective was changed,
and no training was launched. Code, configurations and evaluation artifacts
are versioned; model checkpoints remain local.

The `smoke` directory preserves the initial small check. The first full run in
`f4_return_s01` is superseded: its one-step probes are valid, but one selected
simulator context was at timestep 48 and its five-step continuation exceeded
the original 50-step horizon. The final run excludes that context only from
the five-step comparison, retaining all 15 one-step references and 14 eligible
continuation contexts. No old result was overwritten.

The model-only probes use observed validation histories and nominal actions.
The simulator references use natural noisy teacher actions attached to their
own underlying contexts. Restoration and hidden audit metadata are clearly
separated and never enter model inputs. These individual pairs are not exact
conditional ETT distributions.
