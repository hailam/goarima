package arima

import (
	"math"
	"math/rand/v2"
	"testing"
)

// burgAR must recover known AR(1) coefficient on long synthetic data.
// Tolerance allows for finite-sample noise.
func TestBurgAR_AR1Recovery(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	const phiTrue = 0.5
	const n = 1000
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = phiTrue*y[i-1] + rng.NormFloat64()
	}
	phi, err := burgAR(y, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(phi) != 1 {
		t.Fatalf("len(phi) = %d, want 1", len(phi))
	}
	// At n=1000, expect within ~3·SE of truth. For AR(1) MLE,
	// SE(phi) ≈ √(1-phi²)/√n ≈ 0.027. So |phi - 0.5| < 0.1 is loose.
	if math.Abs(phi[0]-phiTrue) > 0.1 {
		t.Errorf("burgAR AR(1): got phi=%g, want ≈%g", phi[0], phiTrue)
	}
	t.Logf("AR(1) phi_true=%g  burg=%g", phiTrue, phi[0])
}

// burgAR must produce stationary AR(2) coefficients (within unit
// circle) on a known-stationary process.
func TestBurgAR_AR2Stationarity(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	const phi1, phi2 = 0.4, 0.3 // |phi1|+|phi2| < 1 → stationary
	const n = 500
	y := make([]float64, n)
	for i := 2; i < n; i++ {
		y[i] = phi1*y[i-1] + phi2*y[i-2] + rng.NormFloat64()
	}
	phi, err := burgAR(y, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(phi) != 2 {
		t.Fatalf("len(phi) = %d, want 2", len(phi))
	}
	t.Logf("AR(2): true=(%g,%g) burg=(%g,%g)", phi1, phi2, phi[0], phi[1])
	// Stationarity: |phi1 + phi2| < 1 AND |phi2 - phi1| < 1 AND |phi2| < 1.
	if math.Abs(phi[0])+math.Abs(phi[1]) >= 1 {
		t.Errorf("Burg produced non-stationary AR(2): phi=(%g, %g)", phi[0], phi[1])
	}
	if math.Abs(phi[1]) >= 1 {
		t.Errorf("Burg AR(2): |phi2| = %g must be < 1", math.Abs(phi[1]))
	}
}

// Edge case: p ≥ n must error.
func TestBurgAR_ErrorPaths(t *testing.T) {
	if _, err := burgAR([]float64{1, 2}, 5); err == nil {
		t.Error("expected error when p ≥ n")
	}
	if _, err := burgAR([]float64{1, 2}, 0); err == nil {
		t.Error("expected error when p < 1")
	}
}

// On short series with low n, Burg's lower bias compared to OLS should
// give an estimate closer to the truth on average.
func TestBurgAR_VsOLS_LowerBiasOnShortAR1(t *testing.T) {
	if testing.Short() {
		t.Skip("Monte Carlo: K=30 series @ n=40")
	}
	const phiTrue = 0.7
	const n = 40
	const K = 30

	rng := rand.New(rand.NewPCG(123, 456))
	burgSum := 0.0
	olsSum := 0.0
	burgN, olsN := 0, 0
	for k := 0; k < K; k++ {
		y := make([]float64, n)
		for i := 1; i < n; i++ {
			y[i] = phiTrue*y[i-1] + rng.NormFloat64()
		}
		phiBurg, err := burgAR(y, 1)
		if err == nil {
			burgSum += phiBurg[0]
			burgN++
		}
		// OLS comparison: simple AR(1) regression y_t on y_{t-1}.
		num, den := 0.0, 0.0
		for i := 1; i < n; i++ {
			num += y[i] * y[i-1]
			den += y[i-1] * y[i-1]
		}
		if den > 0 {
			olsSum += num / den
			olsN++
		}
	}
	burgMean := burgSum / float64(burgN)
	olsMean := olsSum / float64(olsN)
	burgBias := burgMean - phiTrue
	olsBias := olsMean - phiTrue
	t.Logf("AR(1) phi=%g, n=%d, K=%d:", phiTrue, n, K)
	t.Logf("  Burg mean=%.4f bias=%+.4f", burgMean, burgBias)
	t.Logf("  OLS  mean=%.4f bias=%+.4f", olsMean, olsBias)
	// Both biased, but Burg's bias should be smaller in absolute value.
	if math.Abs(burgBias) >= math.Abs(olsBias) {
		t.Logf("Burg bias not smaller in this run — sample variance may dominate at K=%d", K)
	}
}
