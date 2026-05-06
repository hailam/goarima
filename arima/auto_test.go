package arima

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/hailam/goarima/datasets"
)

// auto_arima must produce a reasonable model on austres (a non-stationary trend).
// Mirrors test_auto_arima_with_arima — auto-selected model should fit and
// produce forecasts.
func TestAutoArimaAustres(t *testing.T) {
	austres := datasets.LoadAustres()
	mdl, err := AutoArima(austres, nil, AutoArimaOpts{
		M:        4,
		MaxP:     3,
		MaxQ:     3,
		MaxCapP:  1,
		MaxCapQ:  1,
		MaxOrder: 5,
		MaxD:     2,
		Alpha:    0.05,
		Test:     NDiffsKPSS,
		IC:       AICc,
		MaxIter:  60,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mdl == nil {
		t.Fatal("auto_arima returned nil")
	}
	// austres requires at least one diff
	if mdl.Order.D == 0 && mdl.Seasonal.D == 0 {
		t.Errorf("austres should have d or D > 0; order=%v seasonal=%v",
			mdl.Order, mdl.Seasonal)
	}
	// Forecast should not error.
	fc, _, _, err := mdl.Predict(4, 0.05, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(fc) != 4 {
		t.Errorf("forecast len=%d", len(fc))
	}
}

// auto_arima on a known AR(1) should pick up p>=1.
func TestAutoArimaAR1(t *testing.T) {
	rng := rand.New(rand.NewPCG(123, 124))
	n := 300
	phi := 0.6
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = phi*y[i-1] + rng.NormFloat64()
	}
	mdl, err := AutoArima(y, nil, AutoArimaOpts{
		M:        0,
		MaxP:     3,
		MaxQ:     3,
		MaxOrder: 5,
		MaxD:     2,
		Alpha:    0.05,
		IC:       AICc,
		MaxIter:  80,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mdl.Order.P == 0 && mdl.Order.Q == 0 && mdl.Order.D == 0 {
		t.Errorf("expected non-trivial model, got %v", mdl.Order)
	}
}

// auto_arima must accept a too-short series only with an error.
func TestAutoArimaTooShort(t *testing.T) {
	y := []float64{1, 2, 3}
	if _, err := AutoArima(y, nil, AutoArimaOpts{}); err == nil {
		t.Error("expected error for very short series")
	}
}

// G-NEW-3c: with ParsimonyDelta set very high, the stepwise search must
// never grow the parameter count past the initial seed (since no
// higher-order neighbor can clear an arbitrarily large IC threshold).
// Same-or-fewer-parameter candidates are unaffected, so simplification
// must still be possible.
func TestAutoArimaParsimonyDeltaCapsParams(t *testing.T) {
	ap := datasets.LoadAirPassengers()

	// Default delta = 0 baseline.
	mDefault, err := AutoArima(ap, nil, AutoArimaOpts{
		M: 12, MaxP: 3, MaxQ: 3, MaxCapP: 1, MaxCapQ: 1,
		MaxOrder: 5, MaxD: 2, IC: AICc, MaxIter: 50,
	})
	if err != nil {
		t.Fatal(err)
	}
	defaultParams := mDefault.Order.P + mDefault.Order.Q +
		mDefault.Seasonal.P + mDefault.Seasonal.Q

	// Same search box but with an effectively-infinite parsimony gate:
	// no neighbor that adds a parameter can ever beat its current best.
	// The result must have <= the default-run param count, and crucially
	// no more than the seed's param count (StartP=2 → seed params = 2).
	mStrict, err := AutoArima(ap, nil, AutoArimaOpts{
		M: 12, MaxP: 3, MaxQ: 3, MaxCapP: 1, MaxCapQ: 1,
		MaxOrder: 5, MaxD: 2, IC: AICc, MaxIter: 50,
		StartP: 2, StartQ: 2, // seed at p=q=2 (4 params before P/Q)
		ParsimonyDelta: 1e9,
	})
	if err != nil {
		t.Fatal(err)
	}
	strictParams := mStrict.Order.P + mStrict.Order.Q +
		mStrict.Seasonal.P + mStrict.Seasonal.Q
	seedParams := 2 + 2 + 0 + 0
	if strictParams > seedParams {
		t.Errorf("ParsimonyDelta=1e9 must cap params at seed (%d); got %d",
			seedParams, strictParams)
	}
	t.Logf("default params=%d, strict params=%d (seed=%d)",
		defaultParams, strictParams, seedParams)
}

// GAP-2 regression: Approximation=true should produce a fitted model
// whose Method is the user's actual choice (default MethodCSSML), even
// though the candidate search ran under MethodCSS internally. Verifies
// the two-stage flow restores Method correctly on the final fit.
func TestGAP2_ApproximationRefitsAtUserMethod(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	m, err := AutoArima(logAp, nil, AutoArimaOpts{
		M: 12, MaxP: 3, MaxQ: 3, MaxCapP: 1, MaxCapQ: 1,
		MaxOrder:      5,
		MaxD:          2,
		IC:            AICc,
		MaxIter:       50,
		Approximation: true,
		// Method left as zero value → MethodCSSML
	})
	if err != nil {
		t.Fatal(err)
	}
	// Final fit must be on MethodCSSML scale (the user's effective Method),
	// not MethodCSS (the search-time override).
	if m.Method != MethodCSSML {
		t.Errorf("final model Method = %v, want MethodCSSML", m.Method)
	}
	// The picked (Order, Seasonal) should be sensible — at minimum a non-trivial fit.
	if m.LogLikelihood() == 0 {
		t.Error("final model has logL=0 — refit may have failed silently")
	}
}

// GAP-2: Approximation=true with explicit Method=MethodML should still
// run the search under MethodCSS but refit at MethodML.
func TestGAP2_ApproximationRespectsExplicitMethodML(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	m, err := AutoArima(logAp, nil, AutoArimaOpts{
		M: 12, MaxP: 2, MaxQ: 2, MaxCapP: 1, MaxCapQ: 1,
		MaxOrder:      5,
		MaxD:          2,
		IC:            AICc,
		MaxIter:       50,
		Approximation: true,
		Method:        MethodML,
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.Method != MethodML {
		t.Errorf("final model Method = %v, want MethodML (user's explicit choice)", m.Method)
	}
}

// AutoArimaOpts.GradientWorkers override — when set to 1, every
// candidate fit gets nWorkers=1 in parallelGradient, forcing serial
// gradient evaluation. Useful in environments where pthread_cond_signal
// cost exceeds the parallel-arithmetic speedup. Verifies the override
// propagates and produces a usable model.
func TestAutoArima_GradientWorkersOverride(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	m, err := AutoArima(logAp, nil, AutoArimaOpts{
		M: 12, MaxP: 2, MaxQ: 2, MaxCapP: 1, MaxCapQ: 1,
		MaxOrder:        4,
		MaxIter:         50,
		IC:              AICc,
		GradientWorkers: 1, // force serial gradient
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.GradientWorkers != 1 {
		t.Errorf("override should propagate; got m.GradientWorkers=%d", m.GradientWorkers)
	}
	if m.LogLikelihood() == 0 {
		t.Error("model not fitted (logL=0)")
	}
}

// GAP-3: Stationary=true must constrain the search to d=D=0 regardless
// of the input series's actual differencing requirement. AirPassengers
// would normally pick d=1 / D=1 (KPSS + OCSB both detect non-stationarity)
// — under Stationary=true, the picked model must have d=D=0.
func TestGAP3_StationaryConstrainsDifferencing(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	m, err := AutoArima(logAp, nil, AutoArimaOpts{
		M: 12, MaxP: 3, MaxQ: 3, MaxCapP: 1, MaxCapQ: 1,
		MaxOrder:   5,
		MaxIter:    50,
		IC:         AICc,
		Stationary: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.Order.D != 0 || m.Seasonal.D != 0 {
		t.Errorf("Stationary=true: got Order.D=%d Seasonal.D=%d, both must be 0",
			m.Order.D, m.Seasonal.D)
	}
}

// GAP-4: AllowDrift=BoolPtr(true) on a d>0 series must include the drift
// term even though the default for d>0 is no-intercept. Symmetric for
// AllowMean on d=0 series.
func TestGAP4_AllowDriftOverride(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	// Force drift on a (0,1,1)(0,1,1) candidate that defaults to no-intercept.
	m, err := AutoArima(logAp, nil, AutoArimaOpts{
		M: 12, MaxP: 1, MaxQ: 1, MaxCapP: 1, MaxCapQ: 1,
		MaxOrder:   3,
		MaxIter:    50,
		IC:         AICc,
		AllowDrift: BoolPtr(true),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !m.WithIntercept {
		t.Errorf("AllowDrift=true on d>0 model: WithIntercept must be true; got %v", m.WithIntercept)
	}
}

func TestGAP4_AllowMeanOverride(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	n := 200
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = 0.5*y[i-1] + rng.NormFloat64() // stationary AR(1)
	}
	// Force mean OFF on a stationary (d=0) series — opposite of the default.
	m, err := AutoArima(y, nil, AutoArimaOpts{
		M: 0, MaxP: 2, MaxQ: 2,
		MaxOrder:   4,
		MaxIter:    50,
		IC:         AICc,
		Stationary: true,
		AllowMean:  BoolPtr(false),
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.WithIntercept {
		t.Errorf("AllowMean=false: WithIntercept must be false; got %v", m.WithIntercept)
	}
}

// GAP-5: m.LjungBoxWithDF must accept an explicit fitdf and produce a
// p-value that differs from m.LjungBox(h)'s when fitdf differs from
// the auto-derived ARMA degrees-of-freedom.
func TestGAP5_LjungBoxWithDFOverride(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	m := NewARIMA(Order{P: 1, D: 1, Q: 1})
	m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
	m.MaxIter = 100
	if err := m.Fit(logAp, nil); err != nil {
		t.Fatal(err)
	}
	autoQ, autoP, err := m.LjungBox(24)
	if err != nil {
		t.Fatal(err)
	}
	// Override fitdf to a deliberately-different value.
	overrideQ, overrideP, err := m.LjungBoxWithDF(24, 5)
	if err != nil {
		t.Fatal(err)
	}
	if autoQ != overrideQ {
		t.Errorf("Q-stat must be invariant to fitdf; got auto=%g override=%g", autoQ, overrideQ)
	}
	// p-value depends on fitdf via the chi-squared df adjustment; if the
	// auto-derived fitdf and our override are different, p must change.
	armaDoF := m.Order.P + m.Order.Q + m.Seasonal.P + m.Seasonal.Q
	if armaDoF != 5 && autoP == overrideP {
		t.Errorf("p-value should depend on fitdf (auto=%d, override=5): got %g == %g",
			armaDoF, autoP, overrideP)
	}
}

// GAP-1 regression: AutoArimaOpts{}.Method (zero value) must resolve to
// MethodCSSML — matching documented default and pmdarima/R behaviour.
// Pre-fix the iota order made MethodCSS the zero value, silently
// downgrading every caller who didn't explicitly set Method.
func TestGAP1_MethodZeroValueIsCSSML(t *testing.T) {
	if (Method(0)) != MethodCSSML {
		t.Errorf("Method(0) = %v, want MethodCSSML — iota order must keep MethodCSSML at 0", Method(0))
	}
	var opts AutoArimaOpts
	if opts.Method != MethodCSSML {
		t.Errorf("AutoArimaOpts{}.Method = %v, want MethodCSSML", opts.Method)
	}
}

// G-NEW-3c: simplification must still work with a high ParsimonyDelta.
// Seed an over-specified model; the search should drop redundant terms
// because dropping a parameter only needs the legacy 1e-6 tolerance,
// not the parsimony delta.
func TestAutoArimaParsimonyDeltaAllowsSimplification(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	m, err := AutoArima(ap, nil, AutoArimaOpts{
		M: 12, MaxP: 3, MaxQ: 3, MaxCapP: 1, MaxCapQ: 1,
		MaxOrder: 6, MaxD: 2, IC: AICc, MaxIter: 50,
		StartP: 3, StartQ: 3, StartCapP: 1, StartCapQ: 1, // 8 params seed
		ParsimonyDelta: 1e9,
	})
	if err != nil {
		t.Fatal(err)
	}
	params := m.Order.P + m.Order.Q + m.Seasonal.P + m.Seasonal.Q
	if params >= 8 {
		t.Errorf("expected stepwise to simplify away from 8-param seed even "+
			"under ParsimonyDelta=1e9; got params=%d", params)
	}
	t.Logf("simplified to params=%d from seed=8", params)
}
