# Fallback paths

When the fast path can't converge, the code falls back to a slower
but more robust alternative — never silently returns wrong results.

| Fast path | Fallback | When fallback fires |
|---|---|---|
| Smith doubling iteration | inclu2 Givens rotations | T not stable (ρ ≥ 1); BFGS pushed off the manifold |
| Cholesky GLS in HEGY p-value cubic | OLS cubic | Cholesky fails on rank-deficient Σ (degenerate stress cases) |
| `kalmanARMALikelihoodInto` returning Inf likelihood | BFGS retreats and tries a different step | F = P[0,0] ≤ 0 or NaN — usually means parameters left the stationarity region |

Joseph form (KAL-1) keeps `kalmanARMALikelihoodInto` itself stable
on its own — the Inf return is a *signal* to BFGS, not a crash.
