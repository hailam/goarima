package arima

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/hailam/goarima/datasets"
)

func TestSimulate_UnfittedErrors(t *testing.T) {
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	if _, err := m.Simulate(10, SimulateOpts{Seed: 1}); err == nil {
		t.Error("expected error simulating from unfitted model")
	}
}

func TestSimulate_BadArgs(t *testing.T) {
	y := simulateAR1(200, 0.5, 1.0, 1)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Simulate(0, SimulateOpts{Seed: 1}); err == nil {
		t.Error("expected error for n=0")
	}
	if _, err := m.Simulate(10, SimulateOpts{BurnIn: -1, Seed: 1}); err == nil {
		t.Error("expected error for negative burnIn")
	}
	if _, err := m.Simulate(10, SimulateOpts{Seed: 1, FutureExog: [][]float64{{0}}}); err == nil {
		t.Error("expected error: model has no exog but futureExog passed")
	}
}

func TestSimulate_ShapeAndDeterminism(t *testing.T) {
	y := simulateAR1(200, 0.5, 1.0, 1)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	a, err := m.Simulate(50, SimulateOpts{Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != 50 {
		t.Fatalf("got %d samples, want 50", len(a))
	}
	// Same seed → bit-identical output.
	b, _ := m.Simulate(50, SimulateOpts{Seed: 42})
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("seed determinism broken at i=%d: %v vs %v", i, a[i], b[i])
		}
	}
	// Different seed → different output (almost surely).
	c, _ := m.Simulate(50, SimulateOpts{Seed: 43})
	allEqual := true
	for i := range a {
		if a[i] != c[i] {
			allEqual = false
			break
		}
	}
	if allEqual {
		t.Error("different seeds produced identical output (statistically impossible)")
	}
}

// For an AR(1) with phi=0.5, sigma=1, the stationary distribution has
// mean=0, var = sigma²/(1-phi²) = 1/0.75 ≈ 1.333. Simulate should match.
func TestSimulate_StatisticalProperties(t *testing.T) {
	// Construct a model with known params (skip Fit; set state directly).
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.phi = []float64{0.5}
	m.sigma2 = 1.0
	m.fitted = true

	const N = 100_000
	samples, err := m.Simulate(N, SimulateOpts{BurnIn: 200, Seed: 12345})
	if err != nil {
		t.Fatal(err)
	}

	mean := 0.0
	for _, v := range samples {
		mean += v
	}
	mean /= float64(N)
	if math.Abs(mean) > 0.05 {
		t.Errorf("simulated mean = %g, want ~0", mean)
	}

	variance := 0.0
	for _, v := range samples {
		d := v - mean
		variance += d * d
	}
	variance /= float64(N - 1)
	wantVar := 1.0 / (1.0 - 0.5*0.5) // = 1.333...
	if math.Abs(variance-wantVar) > 0.05*wantVar {
		t.Errorf("simulated variance = %g, want ~%g (within 5%%)", variance, wantVar)
	}
}

// Recover params: simulate from a known model, refit, compare params.
// Loose tolerance because finite-sample noise is real.
func TestSimulate_RecoverParams(t *testing.T) {
	gen := NewARIMA(Order{P: 1, D: 0, Q: 1})
	gen.phi = []float64{0.6}
	gen.theta = []float64{-0.3}
	gen.sigma2 = 1.0
	gen.fitted = true

	y, err := gen.Simulate(2000, SimulateOpts{BurnIn: 200, Seed: 9876})
	if err != nil {
		t.Fatal(err)
	}
	refit := NewARIMA(Order{P: 1, D: 0, Q: 1})
	refit.WithIntercept = false
	if err := refit.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	// Tolerance 0.1 — finite sample size + optimizer noise.
	if math.Abs(refit.phi[0]-0.6) > 0.1 {
		t.Errorf("recovered phi = %g, want ~0.6", refit.phi[0])
	}
	if math.Abs(refit.theta[0]-(-0.3)) > 0.1 {
		t.Errorf("recovered theta = %g, want ~-0.3", refit.theta[0])
	}
}

// SARIMA case: ARIMA(0,1,1)(0,1,1)[12] on log AirPassengers, then simulate.
// Just check shape, finite values, and reasonable magnitude.
func TestSimulate_SARIMAAirline(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	m := NewARIMA(Order{P: 0, D: 1, Q: 1})
	m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
	m.MaxIter = 100
	if err := m.Fit(logAp, nil); err != nil {
		t.Fatal(err)
	}
	samples, err := m.Simulate(48, SimulateOpts{Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 48 {
		t.Fatalf("got %d samples, want 48", len(samples))
	}
	for i, v := range samples {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("samples[%d] = %v (NaN/Inf)", i, v)
		}
	}
}

// Box-Cox + simulate: output should be inverse-transformed (positive when
// Lambda=0 inverse is exp).
func TestSimulate_BoxCox(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	lambda := 0.0
	m := NewARIMA(Order{P: 0, D: 1, Q: 1})
	m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
	m.Lambda = &lambda
	m.MaxIter = 100
	if err := m.Fit(ap, nil); err != nil {
		t.Fatal(err)
	}
	samples, err := m.Simulate(36, SimulateOpts{Seed: 11})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range samples {
		// Inverse Box-Cox of log → exp → must be positive.
		if v <= 0 {
			t.Errorf("Box-Cox-inverted samples[%d] = %g, want positive", i, v)
		}
	}
}

// With exog: pass a known X column, verify shape.
func TestSimulate_WithExog(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	n := 150
	y := make([]float64, n)
	x := make([][]float64, n)
	for i := 0; i < n; i++ {
		x[i] = []float64{rng.Float64()}
		if i == 0 {
			y[i] = 1.5*x[i][0] + rng.NormFloat64()
		} else {
			y[i] = 0.5*y[i-1] + 1.5*x[i][0] + rng.NormFloat64()
		}
	}
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.WithIntercept = true
	if err := m.Fit(y, x); err != nil {
		t.Fatal(err)
	}
	futureX := make([][]float64, 20)
	for i := range futureX {
		futureX[i] = []float64{rng.Float64()}
	}
	samples, err := m.Simulate(20, SimulateOpts{Seed: 5, FutureExog: futureX})
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 20 {
		t.Fatalf("got %d samples, want 20", len(samples))
	}
	// Wrong row count → error.
	if _, err := m.Simulate(20, SimulateOpts{Seed: 5, FutureExog: futureX[:10]}); err == nil {
		t.Error("expected error: futureExog rows mismatch n")
	}
}
