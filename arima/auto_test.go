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

// PG-91: parallelGradient's serial path (n<4 or nWorkers=1) must produce
// bit-equivalent gradients to its parallel path. Pre-fix the serial path
// used `save+eps` / `save-eps` direct subtraction while parallel used
// `+= eps; -= 2*eps` — differing by 1 ulp on the perturbed values. On
// ill-conditioned problems (m5_with_exog intermittent demand has many
// close-by local minima) that ulp drift pushed BFGS into a different
// basin, surfacing as same-orderKey/different-AICc when stepwise visited
// more candidates first. This test fits the same problem with several
// GradientWorkers values and asserts the AICc is identical regardless.
func TestParallelGradient_SerialMatchesParallel(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	// ARIMA(1,1,1)(1,1,1)[12] gives nFree = 4 — large enough that
	// the parallel path is reachable, while still letting the serial
	// path trigger via GradientWorkers=1.
	gws := []int{1, 2, 4, 8, 16}
	results := make([]float64, len(gws))
	for i, gw := range gws {
		m := NewARIMA(Order{P: 1, D: 1, Q: 1})
		m.Seasonal = SeasonalOrder{P: 1, D: 1, Q: 1, M: 12}
		m.MaxIter = 100
		m.GradientWorkers = gw
		if err := m.Fit(logAp, nil); err != nil {
			t.Fatalf("GW=%d: %v", gw, err)
		}
		results[i] = m.AICc()
	}
	// All variants must converge to the same AICc — the gradient
	// computation must be path-independent.
	for i := 1; i < len(results); i++ {
		if math.Abs(results[i]-results[0]) > 1e-9 {
			t.Errorf("GW sweep AICc inconsistency: GW=%d→%.6f, GW=%d→%.6f (gap=%.6f)",
				gws[0], results[0], gws[i], results[i], results[i]-results[0])
		}
	}
}

// PG-99: candidatesOut sink populates with ranked candidates from the
// CSS search, sorted ascending by IC. Used internally by the
// Approximation wrapper for fallback-on-refit-failure (R-aligned
// robustness). This test verifies the bookkeeping is correct so the
// fallback has the right inputs when it fires.
func TestAutoArima_RankedCandidatesPopulated(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	var ranked []rankedCandidate
	opts := AutoArimaOpts{
		M: 12, MaxP: 3, MaxQ: 3, MaxCapP: 2, MaxCapQ: 2,
		MaxOrder:      5,
		MaxIter:       50,
		IC:            AICc,
		candidatesOut: &ranked,
	}
	if _, err := AutoArima(logAp, nil, opts); err != nil {
		t.Fatalf("AutoArima: %v", err)
	}
	if len(ranked) == 0 {
		t.Fatal("ranked list empty — search visited no candidates")
	}
	// Ascending by IC.
	for i := 1; i < len(ranked); i++ {
		if ranked[i].ic < ranked[i-1].ic {
			t.Errorf("ranked[%d].ic=%.4f < ranked[%d].ic=%.4f — not sorted ascending",
				i, ranked[i].ic, i-1, ranked[i-1].ic)
		}
	}
	// All candidates must share the same (d, D) — the unit-root tests
	// determine these once at the start of the search and stepwise only
	// varies (p, q, P, Q). On log-airpassengers under default SEAS the
	// verdict is d=0, D=1; on raw AirPassengers it'd be d=1, D=1. We
	// don't hardcode the value — just assert consistency.
	d0 := ranked[0].order.D
	cap0 := ranked[0].seasonal.D
	for i, c := range ranked {
		if c.order.D != d0 {
			t.Errorf("ranked[%d] d=%d, ranked[0] d=%d — d should be fixed",
				i, c.order.D, d0)
		}
		if c.seasonal.D != cap0 {
			t.Errorf("ranked[%d] D=%d, ranked[0] D=%d — D should be fixed",
				i, c.seasonal.D, cap0)
		}
		if c.seasonal.M != 12 && (c.seasonal.P+c.seasonal.D+c.seasonal.Q) > 0 {
			t.Errorf("ranked[%d] non-empty seasonal but m=%d (expected 12)", i, c.seasonal.M)
		}
	}
}

// PG-104: MultiStart fits the four H-K initial seeds, picks the
// lowest-IC one, and runs stepwise from there. As of 2026-05-10 it is
// **enabled by default** (MultiStart=nil) for R parity — R's
// stepwise=TRUE always runs the four seeds. This test verifies both
// the default-on path and the explicit-off override.
func TestAutoArima_MultiStartPG104(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	t.Run("default-on", func(t *testing.T) {
		m, err := AutoArima(logAp, nil, AutoArimaOpts{
			M: 12, MaxP: 3, MaxQ: 3, MaxCapP: 2, MaxCapQ: 2,
			MaxOrder: 5, MaxIter: 50, IC: AICc,
		})
		if err != nil {
			t.Fatal(err)
		}
		if m.LogLikelihood() == 0 {
			t.Error("model not fitted")
		}
		// On airpassengers default mode, multi-start should match or
		// beat the default-seed AICc (1017.76 in the canonical grid).
		// Allow small tolerance for BFGS convergence noise.
		if m.AICc() > 1018.5 {
			t.Errorf("multi-start AICc on log-airpassengers default: got %.4f, expected ≤ 1018.5",
				m.AICc())
		}
	})
	t.Run("explicit-off", func(t *testing.T) {
		m, err := AutoArima(logAp, nil, AutoArimaOpts{
			M: 12, MaxP: 3, MaxQ: 3, MaxCapP: 2, MaxCapQ: 2,
			MaxOrder:   5, MaxIter: 50, IC: AICc,
			MultiStart: BoolPtr(false),
		})
		if err != nil {
			t.Fatal(err)
		}
		if m.LogLikelihood() == 0 {
			t.Error("model not fitted")
		}
	})
}

// PG-100: When AllowDrift=true (and d+D > 0), the stepwise constant-
// toggle move is enabled. The picked model's intercept reflects
// whichever produced lower AICc — toggle MAY pick true OR false,
// whichever fits better. We only assert the search converges and
// the toggle option doesn't break it.
//
// Default behaviour (AllowDrift=nil, AllowMean=nil) is unchanged —
// no toggle, opt-in only. A separate sub-test verifies that.
func TestAutoArima_ConstantTogglePG100(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	t.Run("default-no-toggle", func(t *testing.T) {
		// AllowDrift=nil, AllowMean=nil → toggle gated off.
		m, err := AutoArima(logAp, nil, AutoArimaOpts{
			M: 12, MaxP: 2, MaxQ: 2, MaxCapP: 1, MaxCapQ: 1,
			MaxOrder: 4, MaxIter: 50, IC: AICc,
		})
		if err != nil {
			t.Fatal(err)
		}
		if m.LogLikelihood() == 0 {
			t.Error("model not fitted")
		}
	})
	t.Run("allow-drift-toggle-on", func(t *testing.T) {
		// AllowDrift=true → toggle enabled (since d+D > 0 for log-AP).
		// Search must converge and produce a valid model.
		m, err := AutoArima(logAp, nil, AutoArimaOpts{
			M: 12, MaxP: 2, MaxQ: 2, MaxCapP: 1, MaxCapQ: 1,
			MaxOrder:   4, MaxIter: 50, IC: AICc,
			AllowDrift: BoolPtr(true),
		})
		if err != nil {
			t.Fatal(err)
		}
		if m.LogLikelihood() == 0 {
			t.Error("model not fitted")
		}
	})
}

// PG-99: Approximation refit must succeed on standard cases (the
// fallback only fires on rare numerical edge cases). This test
// verifies a fast-mode AutoArima fit on log-airpassengers returns
// a usable model — the fallback path doesn't degrade the success
// case. The fallback is hard to trigger deterministically with real
// data; coverage here is the success path + the construction path
// of the candidate list.
func TestAutoArima_ApproximationRefitFallback_SuccessCase(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	m, err := AutoArima(logAp, nil, AutoArimaOpts{
		M: 12, MaxP: 3, MaxQ: 3, MaxCapP: 2, MaxCapQ: 2,
		MaxOrder:      5,
		MaxIter:       50,
		IC:            AICc,
		Approximation: true,
	})
	if err != nil {
		t.Fatalf("Approximation refit must succeed on log-airpassengers: %v", err)
	}
	if m.LogLikelihood() == 0 {
		t.Error("model not fitted (logL=0)")
	}
	// The picked model must be a valid SARIMA on m=12 data.
	if m.Seasonal.M != 12 {
		t.Errorf("expected seasonal m=12, got %d", m.Seasonal.M)
	}
}

// PG-4a: AutoArimaOpts.StepwiseDiagonals expands stepwise's neighbor
// set with the 2-axis (p±,q±) and (P±,Q±) diagonal moves R's
// auto.arima visits. With diagonals on, the search reaches more
// candidates per iteration; with off, the search uses only single-axis
// ±1 moves (legacy goarima behavior). Both must produce a usable
// model; default (off) must not differ from a baseline OFF run.
func TestAutoArima_StepwiseDiagonalsOptIn(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	base := AutoArimaOpts{
		M: 12, MaxP: 3, MaxQ: 3, MaxCapP: 2, MaxCapQ: 2,
		MaxOrder: 5,
		MaxIter:  50,
		IC:       AICc,
	}
	mOff, err := AutoArima(logAp, nil, base)
	if err != nil {
		t.Fatalf("OFF: %v", err)
	}
	if mOff.LogLikelihood() == 0 {
		t.Error("OFF model not fitted (logL=0)")
	}

	on := base
	on.StepwiseDiagonals = true
	mOn, err := AutoArima(logAp, nil, on)
	if err != nil {
		t.Fatalf("ON: %v", err)
	}
	if mOn.LogLikelihood() == 0 {
		t.Error("ON model not fitted (logL=0)")
	}
	// ON's pick should be at least as good as OFF (it explores a
	// superset of OFF's neighbors). Allow a small tolerance for
	// BFGS convergence noise.
	if mOn.AICc() > mOff.AICc()+0.5 {
		t.Errorf("ON AICc=%.4f should not exceed OFF AICc=%.4f by more than 0.5",
			mOn.AICc(), mOff.AICc())
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
//
// Uses M=0 (non-seasonal) so the picked model has D=0; drift is only
// meaningful when d+D=1 (matches R's auto.arima rule
// `if (d+D > 1) allowdrift <- FALSE`). With seasonal differencing
// active (M=12, D=1) and a non-stationary residual, AutoArima now
// matches R in picking (0,1,1)(0,1,1)[12] with d+D=2, where R also
// quietly disables drift even when allowdrift=TRUE.
func TestGAP4_AllowDriftOverride(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	m, err := AutoArima(logAp, nil, AutoArimaOpts{
		M: 0, MaxP: 2, MaxQ: 2,
		MaxOrder:   4,
		MaxIter:    50,
		IC:         AICc,
		AllowDrift: BoolPtr(true),
	})
	if err != nil {
		t.Fatal(err)
	}
	if d := m.Order.D; d == 0 {
		t.Fatalf("expected d>0 to test AllowDrift; got Order=%v", m.Order)
	}
	if !m.WithIntercept {
		t.Errorf("AllowDrift=true on d>0 model: WithIntercept must be true; got %v (Order=%v)",
			m.WithIntercept, m.Order)
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
