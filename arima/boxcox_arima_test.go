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

// BOXCOX-INV-1: boxCoxInvert must match R's forecast::InvBoxCox
// signed-power convention on the negative domain (xx = λ·x+1 < 0),
// which is reached by the lower prediction-interval bound of Box-Cox
// models. R reference values verified 2026-06-12 (forecast 9.0.2):
//
//	InvBoxCox(-3,  0.5)  == -0.25
//	InvBoxCox(-10, 0.5)  == -16
//	InvBoxCox(-3,  0.4)  == -0.017889... (sign(-0.2)·0.2^2.5)
//	InvBoxCox(1,  -0.5)  == 4
//	InvBoxCox(3,  -0.5)  == NA   (x > -1/λ; inverse undefined)
func TestBoxCoxInvert_RSignedPowerParity(t *testing.T) {
	cases := []struct {
		x, lam, want float64
		wantNaN      bool
	}{
		{-3, 0.5, -0.25, false},
		{-10, 0.5, -16, false},
		{-3, 0.4, -0.0178885438, false},
		{1, -0.5, 4, false},
		{3, -0.5, 0, true},
		{2, 0.5, 4, false},     // positive domain unchanged
		{0, 0.5, 1, false},     // xx=1 → 1
		{-2, 0.5, 0, false},    // xx=0 → 0
	}
	for _, c := range cases {
		got := boxCoxInvert([]float64{c.x}, c.lam, 0)[0]
		if c.wantNaN {
			if !math.IsNaN(got) {
				t.Errorf("InvBoxCox(%v, λ=%v) = %v, want NaN", c.x, c.lam, got)
			}
			continue
		}
		if math.Abs(got-c.want) > 1e-9 {
			t.Errorf("InvBoxCox(%v, λ=%v) = %v, want %v", c.x, c.lam, got, c.want)
		}
	}
}

// BOXCOX-INV-1 integration: Predict with Lambda set must return
// finite, ordered intervals (lo ≤ mean ≤ hi) even when the lower
// bound maps through the negative Box-Cox domain.
func TestARIMA_BoxCoxPredictIntervalsFinite(t *testing.T) {
	// Small positive series with high noise so the transformed lower
	// bound goes well below zero at longer horizons.
	rng := rand.New(rand.NewPCG(7, 9))
	y := make([]float64, 80)
	for i := range y {
		y[i] = 5 + 4*rng.Float64()
	}
	lam := 0.5
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.WithIntercept = true
	m.Lambda = &lam
	m.MaxIter = 100
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	mean, lo, hi, err := m.Predict(24, 0.05, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range mean {
		if math.IsNaN(mean[i]) || math.IsNaN(lo[i]) || math.IsNaN(hi[i]) {
			t.Fatalf("step %d: NaN in forecast (mean=%v lo=%v hi=%v)", i+1, mean[i], lo[i], hi[i])
		}
		if !(lo[i] <= mean[i] && mean[i] <= hi[i]) {
			t.Errorf("step %d: interval not ordered: lo=%v mean=%v hi=%v", i+1, lo[i], mean[i], hi[i])
		}
	}
}
