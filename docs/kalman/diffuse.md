# `kalmanARIMAFull` — the exact-diffuse path

Lives at `arima/kalman_full.go`. Two-part filter:

- **Pinf phase** — handles infinite initial variance for the
  non-stationary (differencing) components without the κ ≈ 1e6
  approximation. Runs until Pinf "collapses" (typically n_diffuse =
  d + D·m steps).
- **Pstar phase** — standard Kalman with finite cov.

## When this path fires

Set `m.NonSimpleDifferencing = true`. Matches R `stats::arima` and
`statsmodels(simple_differencing=False)`. The diffuse approach is
*numerically* cleaner than the κ-trick — eigenvalues stay bounded,
BFGS converges in fewer iterations.

## State-dim implications

State dim grows with differencing: `rd = r + d + D·m`. So the inner
kernel here is bigger than the simple-differencing path. CDX-3
exploits the natural sparsity of `zRow` (the observation row vector
— typically 2-3 nonzeros vs rd=27 for monthly Airline) so the
matvecs stay fast.

See also [`../numerical-stability/diffuse-init.md`](../numerical-stability/diffuse-init.md).
