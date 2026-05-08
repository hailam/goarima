# Fit() flow

```mermaid
flowchart TD
    A[Fit(y, exog)] --> B[difference y per Order.D, Seasonal.D]
    B --> C[centre by mean]
    C --> D{Method?}
    D -->|CSS| E[armaCSS gradient/BFGS]
    D -->|ML| F[Kalman likelihood + BFGS]
    D -->|CSSML| G[CSS warm-start → ML refine]
    E --> H[unpack params, compute residuals]
    F --> H
    G --> H
    H --> I[uncentre + un-difference for fitted values]
```

The Method branch lives in `arima/arima.go:Fit`. CSSML is the default
because R uses it (GAP-1). See [`../policies/method-default.md`](../policies/method-default.md).

## Inputs and outputs

`Fit(y []float64, exog [][]float64) error` — populates the `*ARIMA`
struct in place.

After `Fit` succeeds, the struct holds:

- Fitted params accessible via methods: `m.Params()`, `m.Beta()`,
  `m.Resid()`, `m.FittedValues()`, `m.Sigma2()`, `m.LogLikelihood()`
- Seasonal AR/MA coefficients exposed as fields `m.Phi`, `m.Theta`
  (the non-seasonal `phi`/`theta` are private; access via `Params()`)
- Residuals are NaN in the first `d + D·m` warmup entries
- The centred working series for re-use by `Update` / `Predict`
- A lazy `*Summary` (via `m.Summary()`) with parameter SE / z-stats

## What happens before BFGS

The differencing test pipeline runs in `AutoArima` (not `Fit`). For a
direct `Fit`, the user passes `Order.D` and `Seasonal.D` explicitly;
`Fit` just applies them.

Pre-fit transforms applied in order:

1. Box-Cox if `m.Lambda != nil`
2. Differencing per `(d, D, m)`
3. Mean-centring
4. Exog projection out of y (OLS) — drift / intercept handled here
