package arima

import (
	"math"
	"math/rand/v2"
	"testing"
)

// simulateAR1 generates an AR(1) series with given phi and noise sigma.
func simulateAR1(n int, phi, sigma float64, seed uint64) []float64 {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		var prev float64
		if i > 0 {
			prev = y[i-1]
		}
		y[i] = phi*prev + sigma*rng.NormFloat64()
	}
	return y
}

func simulateMA1(n int, theta, sigma float64, seed uint64) []float64 {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	y := make([]float64, n)
	prevE := 0.0
	for i := 0; i < n; i++ {
		e := sigma * rng.NormFloat64()
		y[i] = e + theta*prevE
		prevE = e
	}
	return y
}

// Verifies the fitted phi coefficient is close to the true value.
func TestARIMA_AR1_Fit(t *testing.T) {
	y := simulateAR1(500, 0.7, 1.0, 42)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 200
	if err := m.Fit(y); err != nil {
		t.Fatal(err)
	}
	params := m.Params()
	if len(params) != 1 {
		t.Fatalf("expected 1 param, got %d", len(params))
	}
	if math.Abs(params[0]-0.7) > 0.07 {
		t.Errorf("fit phi=%.3f want ~0.7", params[0])
	}
}

func TestARIMA_MA1_Fit(t *testing.T) {
	y := simulateMA1(500, 0.5, 1.0, 7)
	m := NewARIMA(Order{P: 0, D: 0, Q: 1})
	m.MaxIter = 200
	if err := m.Fit(y); err != nil {
		t.Fatal(err)
	}
	params := m.Params()
	if len(params) != 1 {
		t.Fatalf("expected 1 param, got %d", len(params))
	}
	if math.Abs(params[0]-0.5) > 0.1 {
		t.Errorf("fit theta=%.3f want ~0.5", params[0])
	}
}

func TestARIMA_RandomWalk_010(t *testing.T) {
	// Random walk: ARIMA(0,1,0). After differencing, white noise.
	rng := rand.New(rand.NewPCG(99, 100))
	y := make([]float64, 200)
	for i := 1; i < len(y); i++ {
		y[i] = y[i-1] + rng.NormFloat64()
	}
	m := NewARIMA(Order{P: 0, D: 1, Q: 0})
	if err := m.Fit(y); err != nil {
		t.Fatal(err)
	}
	if len(m.Params()) != 0 {
		t.Errorf("(0,1,0) should have no params, got %v", m.Params())
	}
	// Forecast 5 ahead — for random walk, the forecast is the last value.
	fc, _, _, err := m.Predict(5, 0)
	if err != nil {
		t.Fatal(err)
	}
	last := y[len(y)-1]
	for i, v := range fc {
		if math.Abs(v-last) > 1e-6 {
			t.Errorf("rw forecast[%d]=%v want last=%v", i, v, last)
		}
	}
}

func TestARIMA_PredictLength(t *testing.T) {
	y := simulateAR1(200, 0.6, 1.0, 1)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 100
	if err := m.Fit(y); err != nil {
		t.Fatal(err)
	}
	fc, lo, hi, err := m.Predict(7, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	if len(fc) != 7 || len(lo) != 7 || len(hi) != 7 {
		t.Errorf("forecast lengths: %d %d %d", len(fc), len(lo), len(hi))
	}
	for i := range fc {
		if !(lo[i] <= fc[i] && fc[i] <= hi[i]) {
			t.Errorf("CI ordering violated at %d: lo=%v fc=%v hi=%v", i, lo[i], fc[i], hi[i])
		}
		if i > 0 {
			// CI width should grow with horizon (or stay same)
			w0 := hi[0] - lo[0]
			wi := hi[i] - lo[i]
			if wi < w0-1e-9 {
				t.Errorf("CI width decreased at h=%d: %.4f < %.4f", i, wi, w0)
			}
		}
	}
}

// White-noise sanity check for the normal PPF.
func TestNormPPF(t *testing.T) {
	cases := map[float64]float64{
		0.5:    0.0,
		0.975:  1.959964,
		0.025:  -1.959964,
		0.99:   2.326348,
		0.01:   -2.326348,
		0.6915: 0.5, // approx
	}
	for p, want := range cases {
		got := normPPF(p)
		if math.Abs(got-want) > 1e-3 {
			t.Errorf("normPPF(%v)=%v want %v", p, got, want)
		}
	}
}

func TestARTransparams(t *testing.T) {
	// arTransparams should map 0 → 0.
	got := arTransparams([]float64{0, 0, 0})
	for i, v := range got {
		if math.Abs(v) > 1e-12 {
			t.Errorf("arTransparams(0)[%d]=%v want 0", i, v)
		}
	}
	// Non-zero input should produce coefficients with absolute partial-autocorrelations < 1
	// and characteristic polynomial roots outside the unit circle (stationarity).
	got = arTransparams([]float64{1.0, -0.5})
	if len(got) != 2 {
		t.Fatalf("len = %d", len(got))
	}
	// |phi_p| (the last) must be |tanh(last param)| < 1, by construction
	if math.Abs(got[1]) >= 1 {
		t.Errorf("|phi_2| = %v not < 1", got[1])
	}
}
