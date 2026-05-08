# Parity modes — picking the right knobs

| You want to match… | Set |
|---|---|
| `pmdarima.auto_arima(y).predict(n)` | default |
| `statsmodels.SARIMAX(y, simple_differencing=True)` | default |
| `R::stats::arima(y, ...)` / `forecast::Arima` | `NonSimpleDifferencing = true` |
| `statsmodels.SARIMAX(y, simple_differencing=False)` (statsmodels' default) | `NonSimpleDifferencing = true; DiffuseConvention = DiffuseStatsmodels` |

The default path is fastest and matches the most common Python
usage. The non-simple-differencing path uses an exact Kalman filter
with Gardner-Harvey-Phillips stationary-cov initialisation and is
needed only when an exact match to R or to statsmodels' default is
required.

Background: [`../kalman/dispatch.md`](../kalman/dispatch.md) and
[`../numerical-stability/diffuse-init.md`](../numerical-stability/diffuse-init.md).
