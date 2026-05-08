# Gotchas — things that bite Python/R porters

- **Order capitalisation.** `Order.P, D, Q` are the non-seasonal
  triple. `Seasonal.P, D, Q, M` are the seasonal triple. Duplicate
  uppercase `P` is unavoidable in Go.
- **Default seasonal test.** pmdarima → OCSB. R `auto.arima` → SEAS
  (Wang-Smith-Hyndman). goarima defaults to `NSDiffsSEAS` (R parity);
  switch to `NSDiffsOCSB` for pmdarima parity. The default flipped
  on 2026-05-07 — pre-flip code defaulted to OCSB.
- **`NonSimpleDifferencing=true`** is the switch that pivots
  goarima from "matches pmdarima/`statsmodels(simple_differencing=True)`"
  to "matches R `stats::arima`/`statsmodels(simple_differencing=False)`".
  See [`../policies/parity-modes.md`](../policies/parity-modes.md).
- **Warmup NaNs.** Fitted-values and residuals are NaN in the
  pre-differencing warmup region (first `d + D·m` entries). Strip
  before passing to ACF / Ljung-Box.
- **`Method` zero value.** Pre GAP-1 (2026-05-05), `AutoArimaOpts{}`
  silently ran CSS and gave biased estimates. Post-fix, the zero
  value is `MethodCSSML` matching R. If you save a model from old
  goarima the JSON is still loadable — `Method` is serialised by
  string name.
- **Box-Cox is auto-inverted everywhere.** `Predict`, `FittedValues`,
  `PredictBoot`, `Simulate` all return values on the original scale
  when `m.Lambda != nil`. No need to invert manually.
- **CSS AICc is on a different scale than ML AICc.** Don't compare
  across Methods — refit at the same Method first.
