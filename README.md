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

Requires Go 1.26+.

### SIMD (optional, AMD64)

A handful of vectorizable hot loops route through
[`go-highway`](https://github.com/ajroetker/go-highway), a transitional
SIMD library that anticipates Go's upcoming native SIMD support:

- **ARM64 (Apple silicon, Graviton, etc.)**: NEON acceleration is on by
  default — no flags needed.
- **AMD64**: SIMD requires `GOEXPERIMENT=simd` at build time. Without
  the flag, the affected loops fall back to scalar Go (correct, just
  not faster).
- **All other architectures**: scalar fallback, transparent.

To enable on AMD64:
```sh
GOEXPERIMENT=simd go build ./...
```

The speedup is targeted (residual computation with non-trivial exog
matrices); end-to-end Fit and AutoArima times are unlikely to move
measurably on small models. The library is treated as a temporary
adapter — when Go's native SIMD lands as GA, this dependency goes away.
See `docs/decisions/0002-simd-go-highway.md` for the full investigation
and bench numbers.

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
samples, _ := m.Simulate(100, arima.SimulateOpts{Seed: 42})

// With exog, custom burn-in, etc.:
samples, _ = m.Simulate(100, arima.SimulateOpts{
    BurnIn:     200,         // 0 → 100 (matches pmdarima/R hidden default)
    Seed:       42,          // 0 → time-based
    FutureExog: futureX,     // required if model has exog
})
```

Mirrors statsmodels' `SARIMAX.simulate` and R's `arima.sim`. Output is on
the model's original scale (Box-Cox-inverted if applicable).

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

## Coming from pmdarima or R? API map

goarima ships as a port of *both* pmdarima and `forecast::auto.arima` /
`stats::arima`. Where pmdarima and R agree on shape/behavior, goarima matches
them. Where they disagree, goarima exposes a knob (e.g.
`DiffuseConvention`, `Method`) and picks a sensible default. The table below
lists the surface differences that bite users porting code over.

| Concern | pmdarima | R | goarima |
|---|---|---|---|
| AR/MA/seasonal orders | `order=(p,d,q), seasonal_order=(P,D,Q,m)` | `order=c(p,d,q), seasonal=list(order=c(P,D,Q), period=m)` | `Order{P,D,Q}` (non-seasonal) and `Seasonal{P,D,Q,M}`. **`Order.P` is the non-seasonal AR order** (lowercase `p` in the math); `Seasonal.P` is the seasonal AR order. Capitalization is forced by Go visibility rules. |
| Confidence interval | `alpha=0.05` (kwarg) | `level=95` | `Predict(n, alpha, futureExog)`. Pass `alpha=0` to skip CIs (returns `nil` for `lower`/`upper`). |
| Fitted values | `predict_in_sample(...)` returns `len(y)` array | `fitted(model)` returns `ts` of length `n` (NA in warmup) | `m.FittedValues()` returns `len(yTrain)` slice with `math.NaN()` in the first `d + D*m` warmup entries. |
| Residuals | `arima_res_.resid` length `len(y)` | `residuals(model)` length `n` (NA in warmup) | `m.Resid()` returns `len(yTrain)` slice with `math.NaN()` in warmup. Filter via `dropNaN()` before passing to non-NaN-aware stats (Ljung-Box, ACF). |
| Update / refresh | `model.update(y, X)` warm-starts MLE on existing params | `Arima(model = existing, x = new_y, xreg = new_X)` warm-starts | `m.Update(y, x)` warm-starts (fast); `m.Refit(y, x)` does a full cold re-fit. Neither re-searches orders — call `AutoArima` fresh for that. |
| Box-Cox | `lambda` kwarg | `lambda` arg in `Arima` | `m.Lambda *float64` (nil = off). `Predict`, `FittedValues`, `PredictBoot`, `Simulate` all inverse-transform automatically. |
| Bootstrap CI | `predict(..., bootstrap=True, n_sims=...)` | not built-in | `m.PredictBoot(n, alpha, nSim, seed, futureExog)` |
| Simulate | `simulate(...)` (burn-in hidden) | `simulate.Arima(...)` (burn-in hidden) | `m.Simulate(n, SimulateOpts{Seed: …, BurnIn: …})`. `BurnIn=0` → 100. |
| Drift | `with_intercept=True` + `d=1` adds drift | `include.drift=TRUE` | `RArima(opts.IncludeDrift = true)` sets `m.DriftIncluded`; `Predict`/`PredictBoot`/`Simulate` auto-extend the drift column so callers don't reconstruct `[n+1, n+2, …]` manually. |
| Seasonal differencing test | `nsdiffs(test='ocsb')` (default) | `nsdiffs(test='ocsb')` (default) | `AutoArimaOpts.SeasonalTest` defaults to `NSDiffsOCSB` (matches both); set to `NSDiffsCH` for legacy R behavior. |
| Estimator | `method='lbfgs'/'css-mle'` | `method='CSS'/'ML'/'CSS-ML'` | `Method` enum: `MethodCSS`, `MethodML`, `MethodCSSML` (default — same as R). |
| Save / load | pickle | `saveRDS` / `readRDS` | `m.Save(io.Writer)` / `arima.LoadARIMA(io.Reader)` write versioned JSON. `*ARIMA` also implements `json.Marshaler`/`json.Unmarshaler`. |
| Forecast variance for integrated models | grows with horizon ✓ | grows with horizon ✓ | `Predict` CI bands grow correctly (cumulative-sum psi for each unit-root factor). |

## License

MIT.
