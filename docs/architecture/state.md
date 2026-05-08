# Where state lives during a Fit

The hot path doesn't allocate. State is split between long-lived
structs and per-call workspaces:

| What | Where | Lifetime |
|---|---|---|
| Model parameters | `*ARIMA` fields | per-Fit |
| Centred working series, exog matrices | local in `Fit` | per-Fit |
| Kalman scratch (P, K, row0, newP, …) | `kalmanWorkspace` in `paramScratch` | per-likelihood-eval, pooled |
| Transparams scratch | `paramScratch` in `paramScratchPool` | per-likelihood-eval, pooled |
| Gardner stationary-cov scratch | `gardnerWorkspace` in `kalmanWorkspace` | per-call, pooled |
| BFGS state | `gonum/optimize` internals | per-Fit |

Pool definitions: `arima/param_scratch.go`. One `paramScratchPool`
acquisition gives you everything for one likelihood call — see
[`../concurrency/pools.md`](../concurrency/pools.md).
