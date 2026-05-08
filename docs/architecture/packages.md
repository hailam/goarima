# Packages

| Package | Lives at | What it owns |
|---|---|---|
| `arima` | `arima/` | `ARIMA` struct, `Fit`/`Predict`/`Update`/`Refit`, `AutoArima`, Kalman variants, unit-root tests, decompose. The bulk of the code. |
| `preprocessing` | `preprocessing/` | Box-Cox, log, Fourier, `DateFeaturizer` — sklearn-style transformers. |
| `pipeline` | `pipeline/` | Chain transformers + an ARIMA into one `Fit`/`Predict` surface. |
| `modelselection` | `modelselection/` | `RollingForecastCV`, `SlidingWindowForecastCV`, `CrossValScore/idate/Predict`. Goroutine-parallel folds. |
| `metrics` | `metrics/` | SMAPE, MAE, MSE. |
| `utils` | `utils/` | ACF, PACF, diff, diff_inv, endog/exog checks. |
| `datasets` | `datasets/` | Built-in series (AirPassengers, Wineind, …). |

## What's intentionally not its own package

- **Kalman** — three Kalman implementations live in the `arima`
  package. They share types and workspaces; splitting them would
  create circular helpers without buying anything.
- **Optimization** — we use `gonum.org/v1/gonum/optimize` directly;
  no wrapper.
- **Linear algebra** — flat `[]float64` slices in hot paths (Kalman,
  Gardner). `gonum/mat` is only imported where matrix algebra is the
  natural API (HEGY OLS, summary covariance).
