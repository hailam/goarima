# goarima

A Go port of [pmdarima](https://github.com/alkaline-ml/pmdarima) and R's
[`stats::arima`](https://www.rdocumentation.org/packages/stats/topics/arima) /
[`forecast::Arima`](https://pkg.robjhyndman.com/forecast/reference/Arima.html) —
auto ARIMA, exact-diffuse Kalman filter, unit-root tests, and a sklearn-style
pipeline.

## Status

- 119 tests pass; `staticcheck -checks=all` and `go vet` clean.
- 1:1 numerical parity (verified at runtime against statsmodels) with three
  reference implementations:
  - `statsmodels SARIMAX(simple_differencing=True)` — default mode
  - `R::stats::arima` — `NonSimpleDifferencing=true, DiffuseConvention=DiffuseR`
  - `statsmodels SARIMAX(simple_differencing=False)` — `DiffuseConvention=DiffuseStatsmodels`
- Concurrency wins R/Python can't easily match: parallel `AutoArima` full-search,
  parallel cross-validation folds (goroutines, no GIL/pickling overhead).

## Install

```sh
go get github.com/hailam/goarima
```

## Quick start

```go
import (
    "github.com/hailam/goarima/arima"
    "github.com/hailam/goarima/datasets"
)

ap := datasets.LoadAirPassengers()

// Auto-select the order via stepwise search.
m, err := arima.AutoArima(ap, nil, arima.AutoArimaOpts{
    M: 12, MaxP: 3, MaxQ: 3, MaxCapP: 1, MaxCapQ: 1, IC: arima.AICc,
})

// Forecast 12 periods with 95 % CI.
fc, lo, hi, err := m.Predict(12, 0.05, nil)
```

### With exogenous regressors

```go
exog := /* [n_obs][k] matrix */
m, _ := arima.AutoArima(y, exog, arima.AutoArimaOpts{ /*...*/ })
fc, _, _, _ := m.Predict(n, 0.05, futureExog)
```

### R-style API

```go
m, _ := arima.RArima(wineind, arima.RArimaOpts{
    Order:    arima.Order{P: 1, D: 1, Q: 1},
    Seasonal: arima.SeasonalOrder{P: 0, D: 1, Q: 1, M: 12},
})
```

### Pipeline with date features

```go
import "github.com/hailam/goarima/preprocessing"

dates := /* []time.Time aligned with y */
feat := preprocessing.NewPipelineDateFeaturizer(dates, preprocessing.DailyStep)
pl, _ := pipeline.NewPipeline([]pipeline.Step{
    {Name: "dates", Exog: feat},
}, arima.NewARIMA(arima.Order{P: 1, D: 0, Q: 0}))
pl.Fit(y, nil)
fc, _, _, _ := pl.Predict(14, 0.05, nil) // dates auto-extend forward
```

### Pipeline (Box-Cox + Fourier + ARIMA)

```go
import (
    "github.com/hailam/goarima/preprocessing"
    "github.com/hailam/goarima/pipeline"
)

logT := preprocessing.NewLogEndogTransformer(0, preprocessing.NegRaise, 1e-16)
arimaModel := arima.NewARIMA(arima.Order{P: 1, D: 0, Q: 0})

pl, _ := pipeline.NewPipeline([]pipeline.Step{
    {Name: "log", Endog: logT},
}, arimaModel)
pl.Fit(y, nil)
fc, _, _, _ := pl.Predict(12, 0.05, nil)
```

### Parallel AutoArima (full search)

```go
m, _ := arima.AutoArima(y, nil, arima.AutoArimaOpts{
    M: 12, MaxP: 5, MaxQ: 5, FullSearch: true, NJobs: 0, // 0 = GOMAXPROCS
})
```

### Simulate samples from a fitted model

```go
// Generate 100 samples from a fitted ARIMA process; deterministic with seed.
samples, _ := m.Simulate(100, 0, 42, nil) // burnIn=0 → default 100
```

Mirrors statsmodels' `SARIMAX.simulate` and R's `arima.sim`. Output is on
the model's original scale (Box-Cox-inverted if applicable). For models
with exog, pass a `futureExog` matrix matching `n` rows.

### Save and load fitted models

```go
// Save
f, _ := os.Create("model.json")
m.Save(f)

// Load
g, _ := os.Open("model.json")
loaded, _ := arima.LoadARIMA(g)
fc, _, _, _ := loaded.Predict(12, 0.05, nil) // identical to original
```

Models also implement `json.Marshaler` / `json.Unmarshaler`, so
`json.Marshal(m)` and `json.Unmarshal(data, &m)` work directly. Format is
versioned (currently `1`); loading an unknown version fails fast with a
clear error.

## Module layout

| Package | Purpose |
|---|---|
| `arima` | ARIMA / SARIMAX, AutoArima, ADF/KPSS/PP, CH/OCSB, exact-diffuse Kalman, decompose |
| `preprocessing` | Box-Cox, Log, Fourier, DateFeaturizer |
| `modelselection` | RollingForecastCV, SlidingWindowForecastCV, parallel CrossVal{Score,idate,Predict} |
| `pipeline` | sklearn-style chain (transformers + ARIMA) |
| `metrics` | SMAPE, MAE, MSE |
| `utils` | ACF, PACF, diff, diff_inv, check_endog/exog |_
| `datasets` | airpassengers, austres, wineind, woolyrnq, lynx, WWWusage, ausbeer, gasoline, heartrate, taylor, sunspots, msft |

## Choosing a parity mode

| You want to match... | Set |
|---|---|
| `pmdarima.auto_arima(y).predict(n)` | default |
| `statsmodels.SARIMAX(y, simple_differencing=True)` | default |
| `R::stats::arima(y, ...)` / `forecast::Arima` | `NonSimpleDifferencing = true` |
| `statsmodels.SARIMAX(y, simple_differencing=False)` (default for that lib) | `NonSimpleDifferencing = true; DiffuseConvention = DiffuseStatsmodels` |

The default path is fastest and matches the most common Python usage. The
non-simple-differencing path uses an exact Kalman filter with Gardner-Harvey-
Phillips stationary-cov initialization and is needed only when an exact match
to R or to statsmodels' default is required.

## License

MIT.
