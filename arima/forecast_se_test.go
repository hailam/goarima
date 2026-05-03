package arima

import (
	"math"
	"testing"
)

// For ARIMA(0,1,0) (random walk), the integrated MA(∞) representation is
// y_t = sum_{j=0..t} e_{t-j}, so psi_h = 1 for all h. Forecast variance at
// horizon h+1 is sigma² · sum_{j=0..h} 1² = sigma² · (h+1), giving
// SE = sigma · sqrt(h+1).
//
// This is the canonical test for integrated-process forecast intervals.
// Pre-fix, computePsi only used the ARMA part (psi=[1,0,0,…]), so var2
// stayed at 1 every horizon → SE was a flat sigma regardless of h.
// Post-fix, psi=[1,1,1,…] and SE scales as expected.
func TestForecastSE_RandomWalk_GrowsLikeSqrtH(t *testing.T) {
	// Construct a known random-walk model directly (skip Fit).
	m := NewARIMA(Order{P: 0, D: 1, Q: 0})
	m.sigma2 = 4.0 // sigma = 2
	m.fitted = true
	m.yTrain = []float64{0, 1, 0, 1, 2, 3, 4} // arbitrary; only used for level
	m.yMSCache = append([]float64(nil), m.yTrain...)
	m.wsCenteredCache = []float64{1, -1, 1, 1, 1, 1}
	m.resids = []float64{1, -1, 1, 1, 1, 1}
	m.nobs = 6
	m.computePsi()

	const horizons = 12
	_, lower, upper, err := m.Predict(horizons, 0.05, nil)
	if err != nil {
		t.Fatal(err)
	}
	z := normPPF(0.975) // 1.959963…

	// Expected SE at horizon h (1-indexed): sigma * sqrt(h).
	// (Predict's internal index h0 starts at 0 and var2 += psi[h0]² before SE,
	// so first forecast has var2=1, second has var2=2, …)
	sigma := math.Sqrt(m.sigma2)
	for h := 0; h < horizons; h++ {
		seGot := (upper[h] - lower[h]) / (2 * z)
		seWant := sigma * math.Sqrt(float64(h+1))
		if math.Abs(seGot-seWant)/seWant > 1e-9 {
			t.Errorf("horizon h=%d: SE got %g, want %g (rel err %g)",
				h+1, seGot, seWant, math.Abs(seGot-seWant)/seWant)
		}
	}
}

// ARIMA(0,2,0): double-integrated white noise. psi=[1,1,1,…] after one
// cumsum becomes [1,2,3,4,…] after the second. Variance at horizon h+1 is
// sigma² · sum_{j=0..h} (j+1)² = sigma² · (h+1)(h+2)(2h+3)/6.
func TestForecastSE_DoubleIntegrated(t *testing.T) {
	m := NewARIMA(Order{P: 0, D: 2, Q: 0})
	m.sigma2 = 1.0
	m.fitted = true
	m.yTrain = []float64{0, 1, 3, 6, 10, 15, 21}
	m.yMSCache = append([]float64(nil), m.yTrain...)
	m.wsCenteredCache = []float64{1, 0, 0, 0, 0}
	m.resids = []float64{1, 0, 0, 0, 0}
	m.nobs = 5
	m.computePsi()

	const horizons = 6
	_, lower, upper, err := m.Predict(horizons, 0.05, nil)
	if err != nil {
		t.Fatal(err)
	}
	z := normPPF(0.975)

	for h := 0; h < horizons; h++ {
		seGot := (upper[h] - lower[h]) / (2 * z)
		// var = sum_{j=0..h} (j+1)² = (h+1)(h+2)(2h+3)/6
		k := float64(h + 1)
		varWant := k * (k + 1) * (2*k + 1) / 6.0
		seWant := math.Sqrt(varWant)
		if math.Abs(seGot-seWant)/seWant > 1e-9 {
			t.Errorf("horizon h=%d: SE got %g, want %g", h+1, seGot, seWant)
		}
	}
}

// SARIMA(0,1,0)(0,1,0)[12]: airline-style integration. After non-seasonal
// cumsum, psi=[1,1,1,…]. After seasonal cumsum at stride 12:
//
//	psi[0..11]  = 1
//	psi[12..23] = 2
//	psi[24..35] = 3, etc.
//
// At horizon 24, var2 = sum_{h=0..23} psi[h]² = 12·1 + 12·4 = 60.
// At horizon 25 (h=24): var2 += psi[24]² = 60 + 9 = 69.
func TestForecastSE_AirlineStyleIntegration(t *testing.T) {
	m := NewARIMA(Order{P: 0, D: 1, Q: 0})
	m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 0, M: 12}
	m.sigma2 = 1.0
	m.fitted = true
	// Need at least D + D*M + 1 = 1 + 12 + 1 = 14 yTrain values for nobs >=1.
	m.yTrain = make([]float64, 30)
	for i := range m.yTrain {
		m.yTrain[i] = float64(i)
	}
	m.yMSCache = append([]float64(nil), m.yTrain...)
	m.wsCenteredCache = make([]float64, len(m.yTrain)-13)
	m.resids = make([]float64, len(m.yTrain)-13)
	m.nobs = len(m.yTrain) - 13
	m.computePsi()

	// Spot-check psi values directly (clearer than working through Predict).
	wantPsi := []float64{
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // h=0..11
		2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // h=12..23
		3, 3, 3, 3, // h=24..27
	}
	for i, w := range wantPsi {
		if m.psiInf[i] != w {
			t.Errorf("psi[%d] = %g, want %g", i, m.psiInf[i], w)
		}
	}
}

// Stationary AR(1): no integration, so psi recursion alone is correct.
// The fix should NOT change behavior for d=D=0 models.
func TestForecastSE_AR1_Unchanged(t *testing.T) {
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.phi = []float64{0.5}
	m.sigma2 = 1.0
	m.fitted = true
	m.yTrain = []float64{0, 0.5, 0.25, 0.125, 0.0625}
	m.yMSCache = append([]float64(nil), m.yTrain...)
	m.wsCenteredCache = []float64{0, 0.5, 0.25, 0.125, 0.0625}
	m.resids = []float64{0, 0.5, 0.0, 0.0, 0.0}
	m.nobs = 5
	m.computePsi()

	// AR(1) psi: psi[h] = 0.5^h.
	for h := 0; h < 10; h++ {
		want := math.Pow(0.5, float64(h))
		if math.Abs(m.psiInf[h]-want) > 1e-12 {
			t.Errorf("psi[%d] = %g, want %g (AR(1) should be unaffected by fix)",
				h, m.psiInf[h], want)
		}
	}
}
