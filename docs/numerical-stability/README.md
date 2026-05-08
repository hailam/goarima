# Numerical stability

ARIMA fitting is sensitive: BFGS pushes parameters toward the unit
circle, the state covariance walks across many orders of magnitude,
and small rounding errors compound across hundreds of timesteps.

## Contents

- [`joseph.md`](joseph.md) — Joseph-form Kalman update (KAL-1)
- [`gardner-dispatch.md`](gardner-dispatch.md) — Smith vs inclu2 vs pure-MA (GARD-OPT-1)
- [`transparams.md`](transparams.md) — AR / MA reflection mapping
- [`diffuse-init.md`](diffuse-init.md) — exact Pinf/Pstar vs the κ-trick
- [`fallbacks.md`](fallbacks.md) — Cholesky → OLS, Smith → inclu2
