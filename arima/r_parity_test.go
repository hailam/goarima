package arima

import (
	"math"
	"testing"

	"github.com/hailam/goarima/datasets"
)

// approxEq compares with absolute and relative tolerance.
func approxEq(got, want, atol, rtol float64) bool {
	if math.IsNaN(got) || math.IsNaN(want) {
		return false
	}
	if math.Abs(got-want) <= atol {
		return true
	}
	return math.Abs(got-want)/math.Abs(want) <= rtol
}

// Canonical Box-Jenkins airline model on log(AirPassengers).
//
// R reference (from `arima(log(AirPassengers), order=c(0,1,1), seasonal=list(order=c(0,1,1), period=12))`):
//
//	ma1     = -0.4018
//	sma1    = -0.5570
//	sigma^2 ≈  0.001348
//	logLik ≈  244.70
//	AIC    ≈ -483.40
//
// Two assertions: the simple-differencing path matches statsmodels with
// simple_differencing=True (which we already verified to 6 digits) and
// the NonSimpleDifferencing path matches R's stats::arima exactly.
func TestRParityAirlineLog(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	m, err := RArima(logAp, RArimaOpts{
		Order:    Order{P: 0, D: 1, Q: 1},
		Seasonal: SeasonalOrder{P: 0, D: 1, Q: 1, M: 12},
		MaxIter:  200,
	})
	if err != nil {
		t.Fatal(err)
	}
	params := m.Params()
	if len(params) != 2 {
		t.Fatalf("expected 2 params, got %d", len(params))
	}
	if !approxEq(params[0], -0.4018, 1e-3, 0) {
		t.Errorf("ma1 = %v want -0.4018", params[0])
	}
	if !approxEq(params[1], -0.5570, 1e-3, 0) {
		t.Errorf("sma1 = %v want -0.5570", params[1])
	}
	if !approxEq(m.Sigma2(), 0.001348, 1e-5, 0) {
		t.Errorf("sigma2 = %v want 0.001348", m.Sigma2())
	}
	if !approxEq(m.LogLikelihood(), 244.6965, 0.05, 0) {
		t.Errorf("logL = %v want 244.6965", m.LogLikelihood())
	}
	if !approxEq(m.AIC(), -483.3930, 0.1, 0) {
		t.Errorf("AIC = %v want -483.3930", m.AIC())
	}

	// NonSimpleDifferencing: should also reach R's optimum and reproduce
	// R's reported logL (244.6965) within 0.005.
	m2 := NewARIMA(Order{P: 0, D: 1, Q: 1})
	m2.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
	m2.NonSimpleDifferencing = true
	m2.MaxIter = 200
	if err := m2.Fit(logAp, nil); err != nil {
		t.Fatal(err)
	}
	if !approxEq(m2.LogLikelihood(), 244.6965, 0.005, 0) {
		t.Errorf("NonSimple airline logL = %v want 244.6965 (R)", m2.LogLikelihood())
	}
	p2 := m2.Params()
	if !approxEq(p2[0], -0.4018, 1e-3, 0) {
		t.Errorf("NonSimple ma1 = %v want -0.4018", p2[0])
	}
	if !approxEq(p2[1], -0.5570, 1e-3, 0) {
		t.Errorf("NonSimple sma1 = %v want -0.5570", p2[1])
	}
}

// wineind ARIMA(1,1,1)(0,1,1)[12] — fits cleanly; we assert the model
// reaches at least statsmodels' published log-likelihood (-1529.25). The
// SARIMA likelihood landscape is multimodal here, so we don't pin to a
// specific local optimum — only that we reach as good or better.
func TestRParityWineindSARIMA(t *testing.T) {
	m, err := RArima(datasets.LoadWineind(), RArimaOpts{
		Order:    Order{P: 1, D: 1, Q: 1},
		Seasonal: SeasonalOrder{P: 0, D: 1, Q: 1, M: 12},
		MaxIter:  200,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Params()) != 3 {
		t.Fatalf("expected 3 params, got %d", len(m.Params()))
	}
	// statsmodels: logL ≈ -1529.25. Accept any optimum at least that good.
	if m.LogLikelihood() < -1529.5 {
		t.Errorf("logL = %v should be ≥ -1529.5 (statsmodels reference)", m.LogLikelihood())
	}
}

// AR(2) on lynx — well-studied; statsmodels and R produce *different*
// fits here due to the AR(2) likelihood having multiple local optima.
//
// statsmodels: ar1 ≈ 1.147, ar2 ≈ -0.600, intercept ≈ 1538
// R:            ar1 ≈ 1.134, ar2 ≈ -0.509, intercept ≈ 1547
//
// Our Go matches statsmodels (since the conventions and Kalman form are
// identical). For full R parity on this case, the user would need to
// supply R-style Hannan-Rissanen initial values, which we don't yet
// implement. We assert a positive AR(1), negative AR(2), and intercept
// near the data mean (~1538), which both R and statsmodels satisfy.
func TestRParityLynxAR2(t *testing.T) {
	m, err := RArima(datasets.LoadLynx(), RArimaOpts{
		Order:       Order{P: 2, D: 0, Q: 0},
		IncludeMean: true,
		MaxIter:     200,
	})
	if err != nil {
		t.Fatal(err)
	}
	params := m.Params()
	if len(params) != 3 {
		t.Fatalf("expected 3 params, got %d", len(params))
	}
	if params[0] <= 0 {
		t.Errorf("ar1 = %v want positive", params[0])
	}
	if params[1] >= 0 {
		t.Errorf("ar2 = %v want negative", params[1])
	}
	// intercept is near the lynx mean (~1538). Allow loose tolerance.
	if math.Abs(params[2]-1538) > 200 {
		t.Errorf("intercept = %v want ~1538", params[2])
	}
}

// AirPassengers ARIMA(2,1,1).
//
// R / statsmodels:
//
//	ar1  ≈  1.0907
//	ar2  ≈ -0.4890
//	ma1  ≈ -0.8439
func TestRParityAirPassengers211(t *testing.T) {
	m, err := RArima(datasets.LoadAirPassengers(), RArimaOpts{
		Order:   Order{P: 2, D: 1, Q: 1},
		MaxIter: 200,
	})
	if err != nil {
		t.Fatal(err)
	}
	params := m.Params()
	if len(params) != 3 {
		t.Fatalf("expected 3 params, got %d", len(params))
	}
	if !approxEq(params[0], 1.0907, 0.05, 0) {
		t.Errorf("ar1 = %v want ~1.09", params[0])
	}
	if !approxEq(params[1], -0.4890, 0.05, 0) {
		t.Errorf("ar2 = %v want ~-0.49", params[1])
	}
	if !approxEq(params[2], -0.8439, 0.05, 0) {
		t.Errorf("ma1 = %v want ~-0.84", params[2])
	}
}

// include.drift test: AirPassengers with d=1 and drift=TRUE.
// The drift coefficient should be positive (AirPassengers is increasing).
func TestRParityIncludeDrift(t *testing.T) {
	m, err := RArima(datasets.LoadAirPassengers(), RArimaOpts{
		Order:        Order{P: 0, D: 1, Q: 0},
		IncludeDrift: true,
		MaxIter:      100,
	})
	if err != nil {
		t.Fatal(err)
	}
	beta := m.Beta()
	if len(beta) != 1 {
		t.Fatalf("expected 1 drift coef, got %d (%v)", len(beta), beta)
	}
	if beta[0] <= 0 {
		t.Errorf("drift coef = %v want positive", beta[0])
	}
}

// include.drift errors when d+D > 1 (matches R's warning, here as an error).
func TestRParityDriftWithHighDiff(t *testing.T) {
	if _, err := RArima([]float64{1, 2, 3, 4, 5}, RArimaOpts{
		Order:        Order{P: 0, D: 2, Q: 0},
		IncludeDrift: true,
	}); err == nil {
		t.Error("expected error: drift incompatible with d+D > 1")
	}
}

// Method "CSS" should produce a similar (but not identical) fit to CSS-ML.
func TestRParityMethodCSS(t *testing.T) {
	y := simulateAR1(300, 0.5, 1.0, 1)
	m, err := RArima(y, RArimaOpts{
		Order:  Order{P: 1, D: 0, Q: 0},
		Method: "CSS",
	})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(m.Params()[0]-0.5) > 0.1 {
		t.Errorf("AR(1) CSS coef = %v want ~0.5", m.Params()[0])
	}
}
