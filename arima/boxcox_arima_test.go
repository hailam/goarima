package arima

import (
	"math"
	"math/rand/v2"
	"testing"
)

// Box-Cox lambda inside ARIMA: log-transform stabilises the variance, then
// inverse-transform restores forecasts to the original scale.
func TestARIMA_BoxCoxLog(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	n := 200
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		// Exponentially increasing process; positive only.
		y[i] = math.Exp(0.5 + 0.01*float64(i) + 0.05*rng.NormFloat64())
	}
	zero := 0.0
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.Lambda = &zero // log transform
	m.WithIntercept = true
	m.MaxIter = 80
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	fc, _, _, err := m.Predict(5, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range fc {
		if v <= 0 {
			t.Errorf("forecast[%d]=%v should be positive (Box-Cox inverse)", i, v)
		}
	}
}

// Lambda=0.5 (square-root style) — round-trip on a known monotone series.
func TestARIMA_BoxCoxLambdaHalf(t *testing.T) {
	half := 0.5
	y := make([]float64, 100)
	for i := range y {
		y[i] = float64(i+1) * 1.5
	}
	m := NewARIMA(Order{P: 0, D: 1, Q: 0})
	m.Lambda = &half
	m.MaxIter = 50
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	fc, _, _, err := m.Predict(3, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Linear-trend extrapolation: next 3 values ≈ 101*1.5, 102*1.5, 103*1.5 = 151.5, 153, 154.5
	want := []float64{151.5, 153, 154.5}
	for i, v := range fc {
		if math.Abs(v-want[i])/want[i] > 0.05 {
			t.Errorf("fc[%d]=%v want ~%v", i, v, want[i])
		}
	}
}

// Negative values must raise during Box-Cox.
func TestARIMA_BoxCoxNegativeError(t *testing.T) {
	zero := 0.0
	m := NewARIMA(Order{P: 0, D: 0, Q: 0})
	m.Lambda = &zero
	if err := m.Fit([]float64{-1, 2, 3, 4, 5}, nil); err == nil {
		t.Error("expected Box-Cox error on negative y")
	}
}
