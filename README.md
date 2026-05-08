# goarima

A Go implementation of the **Hyndman-Khandakar `auto.arima` algorithm**
([`forecast::auto.arima`](https://pkg.robjhyndman.com/forecast/reference/auto.arima.html))
with the same `stats::arima` Kalman likelihood it builds on. Surface API
mirrors [pmdarima](https://github.com/alkaline-ml/pmdarima) so existing
Python users feel at home — but R `forecast` is the canonical reference
when implementations diverge.

Includes auto ARIMA, the exact-diffuse Kalman filter, unit-root tests
(KPSS / OCSB / SEAS / CH), and a sklearn-style pipeline.

When R `forecast::auto.arima`, pmdarima, and goarima disagree, R
wins by default — pmdarima is a secondary reference, and goarima
keeps a divergence only when empirical evidence shows it's better.
Full policy and the parity-mode table live at
[`docs/policies/`](docs/policies/).

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

## Documentation

Architecture and implementation notes live under `docs/`:

| Topic | Folder |
|---|---|
| Package map, top-level Fit() flow, where state lives | [`docs/architecture/`](docs/architecture/) |
| Goroutines, sync.Pool, parallelGradient, locks | [`docs/concurrency/`](docs/concurrency/) |
| The three Kalman variants and when each fires | [`docs/kalman/`](docs/kalman/) |
| AutoArima search modes, differencing pipeline | [`docs/auto-arima/`](docs/auto-arima/) |
| Joseph form, transparams, Smith vs inclu2 dispatch | [`docs/numerical-stability/`](docs/numerical-stability/) |
| Coming from pmdarima or R? Field-by-field map | [`docs/api-mapping/`](docs/api-mapping/) |
| Divergence-decision policy, parity modes | [`docs/policies/`](docs/policies/) |
| Per-decision ADRs (Chen-Liu outliers, SIMD, …) | [`docs/decisions/`](docs/decisions/) |

## License

MIT. Dataset attributions in `NOTICE`.
