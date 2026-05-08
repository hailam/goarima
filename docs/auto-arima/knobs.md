# Knobs that actually matter

| Knob | Default | Tune when |
|---|---|---|
| `MaxP / MaxQ / MaxCapP / MaxCapQ` | 5 / 5 / 2 / 2 | Want bigger model space |
| `MaxD / MaxCapD` | 2 / 1 | Default matches R |
| `MaxOrder` | 5 | p+q+P+Q cap |
| `MaxIter` | 100 | BFGS iters per fit |
| `IC` | AICc | Switch to AIC/BIC for very small / very large n |
| `NJobs` | NumCPU | Reduce on shared hosts; set 1 for deterministic stepwise |
| `Trace` | nil | Pass a callback to log every visited candidate |
| `ParsimonyDelta` | 0 | AICc tie-break threshold favouring simpler models |
| `Approximation` | false | True for R-style two-stage; ~40% faster |

For concurrency-related knobs (`GradientWorkers`, `numWorkers`), see
[`../concurrency/knobs.md`](../concurrency/knobs.md).
