# Kalman variants

Three likelihood paths exist. Picking between them is automatic from
`ARIMA.Method` and `ARIMA.NonSimpleDifferencing`.

## Contents

- [`dispatch.md`](dispatch.md) — which path fires when (with diagram)
- [`arma-likelihood.md`](arma-likelihood.md) — `kalmanARMALikelihoodInto` (the default hot path)
- [`diffuse.md`](diffuse.md) — `kalmanARIMAFull` (exact-diffuse, for `NonSimpleDifferencing=true`)
- [`css.md`](css.md) — `armaCSS` (warm-start path)
- [`optimisations.md`](optimisations.md) — what's been layered on the inner loop
