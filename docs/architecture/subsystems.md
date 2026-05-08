# Subsystem index — where to look in the code

| You want… | Start at |
|---|---|
| Differencing decision (d, D) | `arima/seasonality.go:NDiffs` and `:NSDiffs`. Pipeline: [`../auto-arima/differencing.md`](../auto-arima/differencing.md). |
| Kalman likelihood | Tree in [`../kalman/`](../kalman/). Files: `arima/kalman.go` (ARMA, PG-113), `arima/kalman_full.go` (exact diffuse), `arima/sarimax_kalman.go` (NonSimpleDifferencing entry). |
| Stationary covariance | `arima/getq0.go` — Smith vs inclu2 vs pure-MA. Dispatch rule: [`../numerical-stability/gardner-dispatch.md`](../numerical-stability/gardner-dispatch.md). |
| Stepwise / full / approximation search | `arima/auto.go`. Doc: [`../auto-arima/`](../auto-arima/). |
| HEGY p-values (60 RS tables) | `arima/hegy_dispatch.go`, `arima/hegy_rs_tables.go`. |
| Bootstrap CIs | `arima/bootstrap.go`, `arima/bootstrap_inference.go`. |
| Outlier detection (AO/LS) | `arima/outliers.go`. ADR: [`../decisions/0001-outlier-detection-chen-liu.md`](../decisions/0001-outlier-detection-chen-liu.md). |
| Box-Cox transform | `arima/boxcox.go`. |
| Drift / intercept | `arima/arima.go:Fit` (search `IncludeDrift`); `arima/arima.go:Predict` extends drift on forecast. |
| Save / load | `arima/serialize.go` — versioned JSON. |
