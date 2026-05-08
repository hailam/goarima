# Diffuse initialisation — exact vs the κ-trick

For non-stationary state components (the differencing factors),
initial variance is mathematically infinite. Two approaches:

| Approach | Where | Trade-off |
|---|---|---|
| κ ≈ 1e6 (the "big number trick") | not used in goarima | Simple but conditioning-fragile; eigenvalues span 12 orders of magnitude |
| Koopman-Durbin Pinf/Pstar two-part filter | `arima/kalman_full.go` (`NonSimpleDifferencing` path) | Exact; mathematically clean; matches R |

The default path (simple differencing) dodges the issue by
pre-differencing y, so the ARMA Kalman runs on a stationary process
— see [`../kalman/dispatch.md`](../kalman/dispatch.md).

## Why this matters

The κ-trick gives the Kalman filter a starting `P_0` with one or
more diagonal entries at `1e6`. Within a few steps the filter
collapses these — but during those steps the gain `K` is tiny and
likelihoods can be off by tens of log units. BFGS sees a flatter
surface, takes more iterations to converge, and on bad days lands at
a different optimum than R does.

Pinf/Pstar avoids this by tracking the diffuse (`Pinf`) and proper
(`Pstar`) parts separately and switching to the standard recursion
once `Pinf` collapses. CDX-3 (sparse `zRow`) keeps it fast.
