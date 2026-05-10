package arima

import (
	"testing"

	"github.com/hailam/goarima/datasets"
)

// Mirrors `test-newarima2.R::test_that("tests for ndiffs()", ...)`.
//
// Note: R's `forecast::ndiffs` internally uses the `urca` package (not
// `tseries`). Our Go port follows pmdarima which ports `tseries`. Where R's
// answer (using urca) differs from what our tseries-based test gives, we
// document the divergence and assert the pmdarima value, since urca is not
// in scope of this port.
//
// R published expectations (matched 5/5 post-PG-110b):
//
//	ndiffs(AirPassengers, test = "kpss") == 1
//	ndiffs(AirPassengers, test = "adf")  == 1   PG-110 τ_μ default
//	ndiffs(AirPassengers, test = "pp")   == 1   PG-110b Z(τ) drift default
func TestNDiffsParityAirPassengers(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	cases := []struct {
		test NDiffsTest
		want int
	}{
		{NDiffsKPSS, 1},
		{NDiffsADF, 1}, // PG-110: τ_μ default matches R's urca-based ndiffs
		{NDiffsPP, 1},  // PG-110b: Z(τ) drift default matches R
	}
	for _, c := range cases {
		got, err := NDiffs(ap, NDiffsOpts{Test: c.test, MaxD: 2, Alpha: 0.05})
		if err != nil {
			t.Fatalf("test %v: %v", c.test, err)
		}
		if got != c.want {
			t.Errorf("ndiffs(AirPassengers, %v) = %d want %d", c.test, got, c.want)
		}
	}
}

// Mirrors `test-newarima2.R::test_that("tests for nsdiffs()", ...)`.
//
// R published expectations:
//
//	nsdiffs(AirPassengers, test = "ocsb") == 1
//	nsdiffs(AirPassengers, test = "ch")   == 0
//	nsdiffs(rep(1, 100))                  == 0
func TestNSDiffsParityAirPassengers(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	D, err := NSDiffs(ap, NSDiffsOpts{
		M: 12, MaxD: 1, Test: NSDiffsOCSB, MaxLag: 3, LagMethod: OCSBAIC,
	})
	if err != nil {
		t.Fatalf("OCSB: %v", err)
	}
	if D != 1 {
		t.Errorf("nsdiffs(AirPassengers, ocsb) = %d want 1", D)
	}
	D, err = NSDiffs(ap, NSDiffsOpts{
		M: 12, MaxD: 1, Test: NSDiffsCH, MaxLag: 3,
	})
	if err != nil {
		t.Fatalf("CH: %v", err)
	}
	if D != 0 {
		t.Errorf("nsdiffs(AirPassengers, ch) = %d want 0", D)
	}

	// Constant series → 0
	D, err = NSDiffs(make([]float64, 100), NSDiffsOpts{M: 12, MaxD: 1, Test: NSDiffsOCSB, MaxLag: 3})
	if err != nil {
		t.Fatal(err)
	}
	if D != 0 {
		t.Errorf("nsdiffs(constant) = %d want 0", D)
	}
}

// Wineind: KPSS=1; ADF/PP=0 (tseries-style); OCSB(AIC)=0 (R-style pull-back); CH=0.
// Verified against pmdarima.arima.utils.ndiffs / nsdiffs.
func TestParityWineind(t *testing.T) {
	wi := datasets.LoadWineind()
	if got, _ := NDiffs(wi, NDiffsOpts{Test: NDiffsKPSS, MaxD: 2, Alpha: 0.05}); got != 1 {
		t.Errorf("ndiffs(wineind, kpss) = %d want 1", got)
	}
	if got, _ := NDiffs(wi, NDiffsOpts{Test: NDiffsADF, MaxD: 2, Alpha: 0.05}); got != 0 {
		t.Errorf("ndiffs(wineind, adf) = %d want 0", got)
	}
	if got, _ := NDiffs(wi, NDiffsOpts{Test: NDiffsPP, MaxD: 2, Alpha: 0.05}); got != 0 {
		t.Errorf("ndiffs(wineind, pp) = %d want 0", got)
	}
	// CH on wineind: pmdarima → 0; we agree.
	if got, _ := NSDiffs(wi, NSDiffsOpts{M: 12, MaxD: 1, Test: NSDiffsCH, MaxLag: 3}); got != 0 {
		t.Errorf("nsdiffs(wineind, ch) = %d want 0", got)
	}
}

// Woolyrnq: KPSS=1; OCSB(AIC) m=4 → 0.
func TestParityWoolyrnq(t *testing.T) {
	wq := datasets.LoadWoolyrnq()
	if got, _ := NDiffs(wq, NDiffsOpts{Test: NDiffsKPSS, MaxD: 2, Alpha: 0.05}); got != 1 {
		t.Errorf("ndiffs(woolyrnq, kpss) = %d want 1", got)
	}
	if got, _ := NSDiffs(wq, NSDiffsOpts{
		M: 4, MaxD: 1, Test: NSDiffsOCSB, MaxLag: 3, LagMethod: OCSBAIC,
	}); got != 0 {
		t.Errorf("nsdiffs(woolyrnq, ocsb) = %d want 0", got)
	}
}

// Lynx: stationary (no trend); KPSS → 0.
func TestParityLynx(t *testing.T) {
	ly := datasets.LoadLynx()
	if got, _ := NDiffs(ly, NDiffsOpts{Test: NDiffsKPSS, MaxD: 2, Alpha: 0.05}); got != 0 {
		t.Errorf("ndiffs(lynx, kpss) = %d want 0", got)
	}
}

// KPSS-NDIFFS-1 acceptance test: airpassengers training (n=120)
// seasonally-differenced at lag 12 (n=108) sits right at the KPSS 5%
// threshold. The bug was that `NDiffs(KPSS)` used `tseries::kpss.test`'s
// `lshort=TRUE` lag formula `trunc(4*(n/100)^0.25)` (= 4 here, stat
// 0.328 — below 5% crit 0.463 → ndiffs=0), while
// `forecast::ndiffs(test="kpss")` calls `urca::ur.kpss` with
// `use.lag = trunc(3*sqrt(n)/13)` (= 2 here, stat 0.473 — above 5%
// crit → ndiffs=1).
//
// R reference values (verified 2026-05-10):
//
//	urca::ur.kpss(diff(AirPassengers[1:120], lag=12), type="mu",
//	              use.lag=2)@teststat == 0.4733
//	forecast::ndiffs(diff(AirPassengers[1:120], lag=12), test="kpss") == 1
func TestNDiffsParity_AirPassengersTrainSeasonallyDiffed(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	train := ap[:120]
	sd := make([]float64, len(train)-12)
	for i := range sd {
		sd[i] = train[i+12] - train[i]
	}
	got, err := NDiffs(sd, NDiffsOpts{Test: NDiffsKPSS, MaxD: 2, Alpha: 0.05})
	if err != nil {
		t.Fatal(err)
	}
	if got != 1 {
		t.Errorf("ndiffs(airpassengers train sd-12, kpss) = %d want 1 (matches forecast::ndiffs)", got)
	}
}

// SARIMA fit on wineind — mirrors `test_that("tests for a ts with the seasonal
// component", ...)` from forecast/test-arima.R. Asserts the model fits and
// produces forecasts of the expected length.
func TestSARIMAFitWineind(t *testing.T) {
	wi := datasets.LoadWineind()
	mdl := &ARIMA{
		Order:    Order{P: 1, D: 1, Q: 1},
		Seasonal: SeasonalOrder{P: 0, D: 1, Q: 1, M: 12},
		Method:   MethodCSSML,
		MaxIter:  100,
	}
	if err := mdl.Fit(wi, nil); err != nil {
		t.Fatalf("SARIMA fit: %v", err)
	}
	// 24 forecasts (2 seasonal cycles).
	fc, lo, hi, err := mdl.Predict(24, 0.05, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(fc) != 24 {
		t.Errorf("forecast len = %d want 24", len(fc))
	}
	for i := range fc {
		if !(lo[i] <= fc[i] && fc[i] <= hi[i]) {
			t.Errorf("CI ordering at h=%d: lo=%v fc=%v hi=%v", i, lo[i], fc[i], hi[i])
		}
	}
}

// auto_arima on woolyrnq — mirrors `as.character.Arima()` test
// (`auto.arima(woolyrnq)` should return a valid ARIMA model).
func TestAutoArimaWoolyrnq(t *testing.T) {
	wq := datasets.LoadWoolyrnq()
	mdl, err := AutoArima(wq, nil, AutoArimaOpts{
		M:        4,
		MaxP:     3,
		MaxQ:     3,
		MaxCapP:  1,
		MaxCapQ:  1,
		MaxOrder: 5,
		MaxD:     2,
		Alpha:    0.05,
		IC:       AICc,
		MaxIter:  60,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mdl == nil {
		t.Fatal("auto_arima returned nil")
	}
	// woolyrnq has trend → at least one diff expected.
	if mdl.Order.D == 0 && mdl.Seasonal.D == 0 {
		t.Errorf("expected d or D > 0 for woolyrnq, got order=%v seasonal=%v",
			mdl.Order, mdl.Seasonal)
	}
}

// auto_arima on WWWusage — sanity test: fit must succeed and produce
// reasonable forecasts.
//
// Divergence note: R's `auto.arima(WWWusage, stepwise=FALSE)$arma` returns
// (3,0,0,0,1,1,0) — i.e. ARIMA(3,1,0). That d=1 comes from R's `forecast::ndiffs`
// which uses `urca::ur.kpss` and returns 1 for WWWusage. Our tseries-based
// `NDiffs(WWWusage, kpss)` returns 0 (matches pmdarima). Without urca, our
// auto_arima settles at d=0 with a stationary AR/MA model; that is correct
// for a tseries-grounded port, just different from R.
func TestAutoArimaWWWusage(t *testing.T) {
	wu := datasets.LoadWWWusage()
	mdl, err := AutoArima(wu, nil, AutoArimaOpts{
		M: 0, MaxP: 5, MaxQ: 5, MaxOrder: 5, MaxD: 2,
		Alpha: 0.05, IC: AICc, MaxIter: 80,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mdl == nil {
		t.Fatal("auto_arima returned nil")
	}
	fc, _, _, err := mdl.Predict(10, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(fc) != 10 {
		t.Errorf("forecast len = %d want 10", len(fc))
	}
}
