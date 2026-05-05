package datasets

// R-version dataset variants.
//
// goarima's existing dataset functions (LoadAusbeer, LoadGasoline) ship the
// values from pmdarima.datasets — the older snapshot that pmdarima itself
// distributes. R's `forecast` package distributes longer/newer snapshots of
// the same underlying public-source series:
//
//   forecast::ausbeer  → 218 obs (Q1 1956 – Q2 2010)  vs goarima 212 (pmdarima snapshot)
//   forecast::gasoline → 1378+ obs (continuously updated through 2017+) vs goarima 745 (pmdarima)
//
// These R-version variants are NOT bundled in the source tree. To populate
// them, run `go run ./tools/fetch_datasets`. That tool fetches the canonical
// values from the original publishers (Australian Bureau of Statistics for
// ausbeer, US Energy Information Administration for gasoline), validates the
// shape, and generates `datasets/ausbeer_r.go` / `datasets/gasoline_r.go`
// with the populated values.
//
// Why fetch-on-demand instead of bundling: the R-version snapshots are
// continuously updated upstream (especially gasoline, which is weekly EIA
// data). Bundling locks our copy to a specific date; fetch-on-demand lets
// users reproduce against whatever version of R/forecast they're testing
// against.

// loadAusbeerR is set by the generated ausbeer_r.go (created by
// `go run ./tools/fetch_datasets`). When unset (the default state in a
// fresh checkout), LoadAusbeerR returns a clear error.
var loadAusbeerR func() []float64

// loadGasolineForecastR is set by the generated gasoline_r.go. Same pattern.
var loadGasolineForecastR func() []float64

// LoadAusbeerR returns Australian quarterly beer production matching R's
// `forecast::ausbeer` (218 obs, Q1 1956 – Q2 2010, megalitres).
//
// Returns nil + a clear error message until the fetch tool has populated
// the data. To populate, run from repo root:
//
//	go run ./tools/fetch_datasets ausbeer
//
// For the shorter pmdarima-snapshot version (212 obs), use LoadAusbeer().
func LoadAusbeerR() ([]float64, error) {
	if loadAusbeerR == nil {
		return nil, errDatasetNotFetched("ausbeer", "ausbeer")
	}
	return loadAusbeerR(), nil
}

// LoadGasolineForecastR returns weekly US finished motor gasoline products
// supplied (thousands of barrels per day) matching R's `forecast::gasoline`
// (Feb 1991 – present, ~1378+ obs depending on snapshot).
//
// Returns nil + a clear error until the fetch tool has populated the data.
// To populate:
//
//	go run ./tools/fetch_datasets gasoline
//
// For the shorter pmdarima-snapshot version (745 obs), use LoadGasoline().
func LoadGasolineForecastR() ([]float64, error) {
	if loadGasolineForecastR == nil {
		return nil, errDatasetNotFetched("gasoline-forecast", "gasoline")
	}
	return loadGasolineForecastR(), nil
}
