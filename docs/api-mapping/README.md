# API mapping — pmdarima / R → goarima

goarima implements R `forecast::auto.arima` semantics on a
`stats::arima`-equivalent Kalman likelihood, with API names modelled
on pmdarima so Python users find them familiar. **When pmdarima and
R disagree, R wins** — see [`../policies/divergence-policy.md`](../policies/divergence-policy.md).

## Contents

- [`orders-and-fields.md`](orders-and-fields.md) — Order / Seasonal types, `P` capitalisation, NaN warmup
- [`fit-and-update.md`](fit-and-update.md) — Fit, Update, Refit, Predict
- [`forecast-extras.md`](forecast-extras.md) — Box-Cox, drift, bootstrap, simulate
- [`differencing-tests.md`](differencing-tests.md) — `nsdiffs(test=...)` mapping
- [`save-load.md`](save-load.md) — JSON serialisation
- [`gotchas.md`](gotchas.md) — things that bite Python/R porters
