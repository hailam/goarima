# Forecast extras

| Concern | pmdarima | R | goarima |
|---|---|---|---|
| Box-Cox | `lambda` kwarg | `lambda` arg in `Arima` | `m.Lambda *float64` (nil = off). `Predict`, `FittedValues`, `PredictBoot`, `Simulate` all inverse-transform automatically. |
| Bootstrap CI | `predict(..., bootstrap=True, n_sims=...)` | not built-in | `m.PredictBoot(n, alpha, nSim, seed, futureExog)` |
| Bootstrap parameter inference | not built-in | not built-in | `m.BootstrapInference(opts)` returns mean / SE / CIs and the raw B×P sample matrix |
| Simulate | `simulate(...)` (burn-in hidden) | `simulate.Arima(...)` (burn-in hidden) | `m.Simulate(n, SimulateOpts{Seed: …, BurnIn: …})`. `BurnIn=0` → 100. |
| Drift | `with_intercept=True` + `d=1` adds drift | `include.drift=TRUE` | `RArima(opts.IncludeDrift = true)` sets `m.DriftIncluded`; `Predict` / `PredictBoot` / `Simulate` auto-extend the drift column so callers don't reconstruct `[n+1, n+2, …]` manually. |
| Outlier detection | not built-in | `tsoutliers::tso` (separate pkg) | `m.DetectOutliers(opts)` — Chen-Liu AO/LS detection. ADR: [`../decisions/0001-outlier-detection-chen-liu.md`](../decisions/0001-outlier-detection-chen-liu.md). |
