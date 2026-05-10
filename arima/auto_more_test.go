package arima

import (
	"errors"
	"math"
	"math/rand/v2"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/hailam/goarima/datasets"
	"github.com/hailam/goarima/metrics"
)

// FullSearch must enumerate every combination and find at least the same or
// better IC than stepwise.
func TestAutoArimaFullSearch(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	y := make([]float64, 200)
	for i := 1; i < len(y); i++ {
		y[i] = 0.5*y[i-1] + rng.NormFloat64()
	}
	stepMdl, err := AutoArima(y, nil, AutoArimaOpts{
		MaxP: 3, MaxQ: 3, MaxOrder: 4, IC: AICc, MaxIter: 60,
	})
	if err != nil {
		t.Fatal(err)
	}
	fullMdl, err := AutoArima(y, nil, AutoArimaOpts{
		MaxP: 3, MaxQ: 3, MaxOrder: 4, IC: AICc, MaxIter: 60, FullSearch: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	// Full search visits at least as many combinations as stepwise — its IC
	// should be ≤ stepwise's (within a tiny tolerance for optimizer noise).
	if fullMdl.AICc() > stepMdl.AICc()+0.5 {
		t.Errorf("FullSearch IC=%v worse than stepwise=%v", fullMdl.AICc(), stepMdl.AICc())
	}
}

// FullSearch with NFits caps candidates and uses random sampling.
func TestAutoArimaNFits(t *testing.T) {
	y := datasets.LoadAusbeer()
	// drop trailing NaN
	y = y[:len(y)-1]
	mdl, err := AutoArima(y, nil, AutoArimaOpts{
		M: 4, MaxP: 3, MaxQ: 3, MaxCapP: 1, MaxCapQ: 1, MaxOrder: 4,
		IC: AICc, MaxIter: 30,
		FullSearch: true, NFits: 5, Seed: 42,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mdl == nil {
		t.Fatal("nil model")
	}
}

// Trace receives one line per candidate.
func TestAutoArimaTrace(t *testing.T) {
	y := simulateAR1(150, 0.4, 1.0, 1)
	var lines []string
	_, err := AutoArima(y, nil, AutoArimaOpts{
		MaxP: 2, MaxQ: 2, MaxOrder: 3, IC: AIC, MaxIter: 30,
		FullSearch: true, // ensures multiple candidates
		Trace:      func(s string) { lines = append(lines, s) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(lines) == 0 {
		t.Error("trace received no lines")
	}
	for _, l := range lines {
		if !strings.Contains(l, "ARIMA") {
			t.Errorf("trace line missing ARIMA: %q", l)
		}
	}
}

// out_of_sample_size scoring uses SMAPE on the holdout.
func TestAutoArimaOutOfSample(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	y := make([]float64, 200)
	for i := 1; i < len(y); i++ {
		y[i] = 0.6*y[i-1] + rng.NormFloat64()
	}
	mdl, err := AutoArima(y, nil, AutoArimaOpts{
		MaxP: 2, MaxQ: 2, MaxOrder: 3, IC: AIC, MaxIter: 30,
		OutOfSampleSize: 20,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mdl == nil {
		t.Fatal("nil model")
	}
	// Model should still be a valid AR-ish fit after holdout-based selection.
	if mdl.Order.P > 2 {
		t.Errorf("p=%d should be <=2", mdl.Order.P)
	}
}

// AUTOSEL-1 acceptance test: on co2 (n=468, m=12), AutoArima with
// OutOfSampleSize=24 and MASE scoring must reach a holdout MASE
// competitive with R's auto.arima (≤ 0.30 on the last 24 months).
//
// The bug report claimed goarima's default-AICc pick gave MASE=0.301
// vs Python pmdarima's MASE=0.217 — a 0.084 gap. After KPSS-NDIFFS-1
// shipped, the default pick already gives MASE ≈ 0.216 (proper
// seasonal-12 MASE), and OutOfSampleSize=24 + MASE scoring picks a
// neighbor with marginally better holdout MASE (≈ 0.215).
//
// This test asserts BOTH paths land below 0.30, with the OOS path
// not regressing more than 5% vs the default. R's auto.arima on the
// same train gives MASE ≈ 0.258.
func TestAutoArimaCO2HoldoutMASE(t *testing.T) {
	co2 := datasets.LoadCO2()
	const k = 24
	train := co2[:len(co2)-k]
	test := co2[len(co2)-k:]

	// Default-AICc path.
	mDefault, err := AutoArima(train, nil, AutoArimaOpts{
		M: 12, IC: AICc, MaxIter: 50,
	})
	if err != nil {
		t.Fatalf("default AutoArima: %v", err)
	}
	fcDefault, _, _, err := mDefault.Predict(k, 0, nil)
	if err != nil {
		t.Fatalf("default Predict: %v", err)
	}
	maseDefault, err := metrics.MASE(test, fcDefault, train, 12)
	if err != nil {
		t.Fatalf("default MASE: %v", err)
	}

	// OutOfSampleSize=24 + MASE scoring path.
	mOOS, err := AutoArima(co2, nil, AutoArimaOpts{
		M: 12, IC: AICc, MaxIter: 50,
		OutOfSampleSize: k,
		Scoring:         metrics.MASEScoring(train, 12),
	})
	if err != nil {
		t.Fatalf("OOS AutoArima: %v", err)
	}
	fcOOS, _, _, err := mOOS.Predict(k, 0, nil)
	if err != nil {
		t.Fatalf("OOS Predict: %v", err)
	}
	maseOOS, err := metrics.MASE(test, fcOOS, train, 12)
	if err != nil {
		t.Fatalf("OOS MASE: %v", err)
	}

	t.Logf("co2 default-AICc: Order=%v Seasonal=%v MASE=%.4f",
		mDefault.Order, mDefault.Seasonal, maseDefault)
	t.Logf("co2 OOS=24 MASE:  Order=%v Seasonal=%v MASE=%.4f",
		mOOS.Order, mOOS.Seasonal, maseOOS)

	const maseCeiling = 0.30
	if maseDefault > maseCeiling {
		t.Errorf("default MASE %.4f > ceiling %.4f", maseDefault, maseCeiling)
	}
	if maseOOS > maseCeiling {
		t.Errorf("OOS MASE %.4f > ceiling %.4f", maseOOS, maseCeiling)
	}
	// Tolerance: OOS must not be more than 5% worse than default
	// (different picks; small jitter expected, large regressions are
	// a genuine selector bug).
	if maseOOS > maseDefault*1.05 {
		t.Errorf("OOS MASE %.4f regressed > 5%% vs default %.4f", maseOOS, maseDefault)
	}
}

// error_action=ignore: a candidate that fails should not abort the whole run.
func TestAutoArimaErrorActionIgnore(t *testing.T) {
	y := simulateAR1(80, 0.5, 1.0, 1)
	mdl, err := AutoArima(y, nil, AutoArimaOpts{
		MaxP: 5, MaxQ: 5, MaxOrder: 8, IC: AICc, MaxIter: 20,
		ErrorAction: "ignore", FullSearch: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mdl == nil {
		t.Fatal("nil model")
	}
}

// Regression: AutoArima must pick D=1 on AirPassengers — both pmdarima and
// R's forecast::auto.arima default to OCSB which selects D=1 here. Pre-fix
// the search hard-coded the CH test which returned D=0, leaving the model
// to fit a non-seasonally-differenced SARIMA and producing visibly worse
// forecasts on monthly data.
func TestAutoArimaSelectsOCSBDByDefault(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	m, err := AutoArima(ap, nil, AutoArimaOpts{
		M: 12, MaxP: 2, MaxQ: 2, MaxCapP: 1, MaxCapQ: 1,
		MaxOrder: 5, MaxD: 1, IC: AICc, MaxIter: 50,
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.Seasonal.D != 1 {
		t.Errorf("AutoArima picked Seasonal.D = %d, want 1 (matches pmdarima/forecast::auto.arima with OCSB default)",
			m.Seasonal.D)
	}
}

// Short-series case (n < 4m) where OCSB and CH commonly disagree. Use the
// first 41 observations of AirPassengers — known real data with a stochastic
// seasonal that R's OCSB picks up (D=1) where CH does not (D=0). Pre-fix
// AutoArima silently used CH and underpicked D on this kind of series.
func TestAutoArimaShortSeasonalSeries(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	short := ap[:41]
	// Verify OCSB and CH disagree on this slice (otherwise the test isn't
	// actually exercising the bug).
	dOCSB, err := NSDiffs(short, NSDiffsOpts{
		M: 12, MaxD: 1, Test: NSDiffsOCSB, MaxLag: 3, LagMethod: OCSBAIC,
	})
	if err != nil {
		t.Fatal(err)
	}
	dCH, err := NSDiffs(short, NSDiffsOpts{
		M: 12, MaxD: 1, Test: NSDiffsCH, MaxLag: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if dOCSB == dCH {
		t.Skipf("OCSB and CH agree on short series (both = %d); test loses its bite", dOCSB)
	}
	// AutoArima default should land on the OCSB answer.
	m, err := AutoArima(short, nil, AutoArimaOpts{
		M: 12, MaxP: 1, MaxQ: 1, MaxCapP: 1, MaxCapQ: 1,
		MaxOrder: 3, MaxD: 1, IC: AICc, MaxIter: 30,
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.Seasonal.D != dOCSB {
		t.Errorf("short-series AutoArima picked Seasonal.D = %d, want OCSB result %d (got CH result %d instead — default flipped to CH again?)",
			m.Seasonal.D, dOCSB, dCH)
	}
}

// Explicit opt-in to NSDiffsCH (legacy R behavior) should still be honored.
func TestAutoArimaCHOptIn(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	m, err := AutoArima(ap, nil, AutoArimaOpts{
		M: 12, MaxP: 1, MaxQ: 1, MaxCapP: 1, MaxCapQ: 1,
		MaxOrder: 3, MaxD: 1, IC: AICc, MaxIter: 30,
		SeasonalTest: NSDiffsCH,
	})
	if err != nil {
		t.Fatal(err)
	}
	// On AirPassengers, CH returns D=0 — verifies the opt-in path actually
	// uses CH instead of silently dispatching OCSB.
	if m.Seasonal.D != 0 {
		t.Errorf("AutoArima with SeasonalTest=NSDiffsCH picked Seasonal.D = %d, want 0",
			m.Seasonal.D)
	}
}

// CD-F2: AutoArima accepts Lambda and threads it into every candidate fit.
// Verified by fitting with Lambda=0 (log) on AirPassengers and checking
// that m.Lambda is set on the returned model + that forecasts are positive
// (Box-Cox-inverted from log scale).
func TestAutoArima_Lambda(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	lambda := 0.0 // log
	mdl, err := AutoArima(ap, nil, AutoArimaOpts{
		M: 12, MaxP: 1, MaxQ: 1, MaxCapP: 1, MaxCapQ: 1,
		MaxOrder: 3, IC: AICc, MaxIter: 50,
		Lambda: &lambda,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mdl.Lambda == nil || *mdl.Lambda != 0 {
		t.Errorf("Lambda not threaded: got %v", mdl.Lambda)
	}
	// Forecast must be on the original positive scale (Box-Cox-inverted).
	fc, _, _, err := mdl.Predict(12, 0.05, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range fc {
		if v <= 0 {
			t.Errorf("fc[%d] = %g, want positive (Box-Cox-inverted from log)", i, v)
		}
	}
}

// CD-F2: AutoArima with NonSimpleDifferencing=true threads it into candidate
// fits. Verify by checking the returned model's flag is set.
func TestAutoArima_NonSimpleDifferencing(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	mdl, err := AutoArima(logAp, nil, AutoArimaOpts{
		M: 12, MaxP: 1, MaxQ: 1, MaxCapP: 1, MaxCapQ: 1,
		MaxOrder: 3, IC: AICc, MaxIter: 50,
		NonSimpleDifferencing: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !mdl.NonSimpleDifferencing {
		t.Error("NonSimpleDifferencing not threaded into returned model")
	}
}

// CD-F2: explicit Method choice threads into candidate fits.
func TestAutoArima_MethodCSS(t *testing.T) {
	y := simulateAR1(200, 0.5, 1.0, 7)
	mdl, err := AutoArima(y, nil, AutoArimaOpts{
		M: 0, MaxP: 1, MaxQ: 1, MaxOrder: 2, IC: AIC, MaxIter: 30,
		Method: MethodCSS,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mdl.Method != MethodCSS {
		t.Errorf("Method = %v, want MethodCSS", mdl.Method)
	}
}

// CD-F2: DiffuseConvention defaults to DiffuseR; verify DiffuseStatsmodels
// is honored when explicitly set alongside NonSimpleDifferencing.
func TestAutoArima_DiffuseStatsmodels(t *testing.T) {
	wi := datasets.LoadWineind()
	mdl, err := AutoArima(wi, nil, AutoArimaOpts{
		M: 12, MaxP: 1, MaxQ: 1, MaxCapP: 0, MaxCapQ: 1,
		MaxOrder: 3, IC: AICc, MaxIter: 50,
		NonSimpleDifferencing: true,
		DiffuseConvention:     DiffuseStatsmodels,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !mdl.NonSimpleDifferencing {
		t.Error("NonSimpleDifferencing not threaded")
	}
	if mdl.DiffuseConvention != DiffuseStatsmodels {
		t.Errorf("DiffuseConvention = %v, want DiffuseStatsmodels", mdl.DiffuseConvention)
	}
}

// Codex #C12: ErrorAction="raise" (default) must propagate fit errors even
// when other candidates succeed. Pre-fix the FullSearch path returned
// (best != nil, err == nil) when the best candidate happened to succeed,
// silently swallowing failures from sibling candidates.
//
// We can't easily inject a fit failure (every well-formed candidate succeeds
// on synthetic data), so we test the error-PROPAGATION wiring directly:
// a custom Scoring callback that returns an error on the first call. With
// "raise", the search must surface that error; with "ignore", it must not.
func TestAutoArimaErrorActionRaisePropagates(t *testing.T) {
	y := simulateAR1(150, 0.5, 1.0, 3)
	scoringFails := func(yt, yp []float64) (float64, error) {
		return 0, errors.New("synthetic scoring error")
	}
	// FullSearch path
	_, err := AutoArima(y, nil, AutoArimaOpts{
		MaxP: 1, MaxQ: 1, MaxOrder: 2, IC: AICc, MaxIter: 20,
		OutOfSampleSize: 10, Scoring: scoringFails,
		ErrorAction: "raise", FullSearch: true,
	})
	if err == nil {
		t.Error("FullSearch + raise: expected error to propagate, got nil")
	}
	// Stepwise path
	_, err = AutoArima(y, nil, AutoArimaOpts{
		MaxP: 1, MaxQ: 1, MaxOrder: 2, IC: AICc, MaxIter: 20,
		OutOfSampleSize: 10, Scoring: scoringFails,
		ErrorAction: "raise",
	})
	if err == nil {
		t.Error("Stepwise + raise: expected error to propagate, got nil")
	}
	// Same setup with ignore: must NOT propagate the synthetic scoring
	// error. (Search may still error with "no candidate succeeded" since
	// every candidate's scoring fails, but the error must not be the
	// synthetic-scoring one.)
	_, err = AutoArima(y, nil, AutoArimaOpts{
		MaxP: 1, MaxQ: 1, MaxOrder: 2, IC: AICc, MaxIter: 20,
		OutOfSampleSize: 10, Scoring: scoringFails,
		ErrorAction: "ignore", FullSearch: true,
	})
	if err != nil && strings.Contains(err.Error(), "synthetic scoring error") {
		t.Errorf("ignore mode propagated scoring error: %v", err)
	}
}

// Custom scoring function.
func TestAutoArimaCustomScoring(t *testing.T) {
	y := simulateAR1(150, 0.4, 1.0, 1)
	// AutoArima may invoke Scoring concurrently across neighbor candidates
	// in stepwise mode (and across the search box in FullSearch mode), so
	// any state captured in the closure must be safe for concurrent access.
	var called atomic.Bool
	scoring := func(yt, yp []float64) (float64, error) {
		called.Store(true)
		if len(yt) != len(yp) {
			return 0, errors.New("length mismatch")
		}
		s := 0.0
		for i := range yt {
			d := yt[i] - yp[i]
			s += math.Abs(d)
		}
		return s / float64(len(yt)), nil
	}
	mdl, err := AutoArima(y, nil, AutoArimaOpts{
		MaxP: 2, MaxQ: 2, MaxOrder: 3, IC: AIC, MaxIter: 30,
		OutOfSampleSize: 15,
		Scoring:         scoring,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !called.Load() {
		t.Error("custom scorer was never called")
	}
	if mdl == nil {
		t.Fatal("nil model")
	}
}

// MaxSteps caps the stepwise iterations.
func TestAutoArimaMaxSteps(t *testing.T) {
	y := simulateAR1(100, 0.3, 1.0, 1)
	mdl, err := AutoArima(y, nil, AutoArimaOpts{
		MaxP: 3, MaxQ: 3, MaxOrder: 4, IC: AIC, MaxIter: 30,
		MaxSteps: 1, // very tight
	})
	if err != nil {
		t.Fatal(err)
	}
	if mdl == nil {
		t.Fatal("nil model")
	}
}
