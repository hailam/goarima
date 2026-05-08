# Parameter reflection — `arTransparams` / `maTransparams`

BFGS sees an unconstrained parameter space; we map back to the
stationary-AR / invertible-MA regions before each likelihood call.
The mapping is the partial-autocorrelation reflection:

- Each unconstrained `p_i` → `tanh(p_i)` (∈ (-1, 1))
- Recursion: `φ_k` for `k=1..p` built so all roots stay outside unit
  circle (AR) or invertibility holds (MA).

Lives at `arima/param_scratch.go:arTransparamsInto` /
`maTransparamsInto`. Workspace-pooled per
[`../concurrency/pools.md`](../concurrency/pools.md).

Without this BFGS routinely escapes the stationarity region on small
samples and the Kalman Joseph guard alone can't save it.
