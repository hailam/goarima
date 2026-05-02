package arima

import (
	"math"
	"math/rand/v2"
	"testing"
)

// Linear regression with white-noise residuals: y = 2 + 3*x1 + 0.5*x2 + e.
// ARIMA(0,0,0) with exog should recover beta = [3, 0.5] (intercept = 2 captured by mean).
func TestARIMA_ExogLinearOnly(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	n := 300
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x1 := rng.NormFloat64()
		x2 := rng.NormFloat64()
		X[i] = []float64{x1, x2}
		y[i] = 2 + 3*x1 + 0.5*x2 + rng.NormFloat64()
	}
	m := NewARIMA(Order{P: 0, D: 0, Q: 0})
	m.WithIntercept = true
	m.MaxIter = 200
	if err := m.Fit(y, X); err != nil {
		t.Fatal(err)
	}
	beta := m.Beta()
	if len(beta) != 2 {
		t.Fatalf("beta len=%d", len(beta))
	}
	if math.Abs(beta[0]-3) > 0.1 {
		t.Errorf("beta[0]=%v want ~3", beta[0])
	}
	if math.Abs(beta[1]-0.5) > 0.1 {
		t.Errorf("beta[1]=%v want ~0.5", beta[1])
	}
}

// AR(1) errors with linear regression: y_t = 1 + 2*x_t + u_t, u_t = 0.6*u_{t-1} + e_t.
// Joint estimation should recover beta ≈ 2 and phi ≈ 0.6.
func TestARIMA_ExogWithARErrors(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	n := 500
	X := make([][]float64, n)
	y := make([]float64, n)
	u := 0.0
	for i := 0; i < n; i++ {
		x := rng.NormFloat64()
		u = 0.6*u + rng.NormFloat64()
		X[i] = []float64{x}
		y[i] = 1 + 2*x + u
	}
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.WithIntercept = true
	m.MaxIter = 300
	if err := m.Fit(y, X); err != nil {
		t.Fatal(err)
	}
	beta := m.Beta()
	if len(beta) != 1 {
		t.Fatalf("beta len=%d", len(beta))
	}
	if math.Abs(beta[0]-2) > 0.1 {
		t.Errorf("beta[0]=%v want ~2", beta[0])
	}
	params := m.Params()
	// Layout: [phi, intercept, beta]; phi is index 0.
	if math.Abs(params[0]-0.6) > 0.1 {
		t.Errorf("phi=%v want ~0.6", params[0])
	}
}

// Forecast with future exog must use the supplied X for predictions.
func TestARIMA_ExogForecast(t *testing.T) {
	rng := rand.New(rand.NewPCG(33, 34))
	n := 200
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x := float64(i) // linear trend regressor
		X[i] = []float64{x}
		y[i] = 5 + 0.1*x + rng.NormFloat64()*0.5
	}
	m := NewARIMA(Order{P: 0, D: 0, Q: 0})
	m.WithIntercept = true
	m.MaxIter = 200
	if err := m.Fit(y, X); err != nil {
		t.Fatal(err)
	}
	futureX := make([][]float64, 5)
	for i := 0; i < 5; i++ {
		futureX[i] = []float64{float64(n + i)}
	}
	fc, _, _, err := m.Predict(5, 0, futureX)
	if err != nil {
		t.Fatal(err)
	}
	if len(fc) != 5 {
		t.Fatalf("forecast len=%d", len(fc))
	}
	// fc should follow the trend: roughly 5 + 0.1*(200..204).
	for i, v := range fc {
		want := 5 + 0.1*float64(n+i)
		if math.Abs(v-want) > 1.5 {
			t.Errorf("fc[%d]=%v want ~%v", i, v, want)
		}
	}
}

// Predict without future exog must error if model used exog.
func TestARIMA_ExogPredictMissingX(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	n := 100
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		X[i] = []float64{rng.NormFloat64()}
		y[i] = X[i][0] + rng.NormFloat64()
	}
	m := NewARIMA(Order{P: 0, D: 0, Q: 0})
	m.WithIntercept = true
	m.MaxIter = 60
	if err := m.Fit(y, X); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := m.Predict(5, 0, nil); err == nil {
		t.Error("expected error: future exog required")
	}
}

// FittedValues / PredictInSample sanity: returns len(y) - d - D*m values, each finite.
func TestARIMA_FittedValues(t *testing.T) {
	y := simulateAR1(200, 0.5, 1.0, 50)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 80
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	fv := m.FittedValues()
	if len(fv) != len(y) {
		t.Errorf("FittedValues len=%d want %d", len(fv), len(y))
	}
	for i, v := range fv {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("fv[%d] not finite: %v", i, v)
		}
	}
}

// Update appends new observations and refits.
func TestARIMA_Update(t *testing.T) {
	y := simulateAR1(150, 0.5, 1.0, 1)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 60
	if err := m.Fit(y[:100], nil); err != nil {
		t.Fatal(err)
	}
	if err := m.Update(y[100:], nil); err != nil {
		t.Fatal(err)
	}
	if len(m.yTrain) != 150 {
		t.Errorf("yTrain after Update = %d, want 150", len(m.yTrain))
	}
}
