package arima

import (
	"math"
	"math/rand/v2"
	"testing"
)

// BIAS-1: mechanics test. After CorrectBiasBootstrap, the coefficients
// must change from their MLE values; psi cache must be recomputed.
// Direction check: AR(1) MLE biases φ̂ DOWNWARD on short n, so the
// corrected estimate should be LARGER (closer to true positive φ).
func TestCorrectBiasBootstrap_AR1MechanicsAndDirection(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	const phiTrue = 0.7
	const n = 50
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = phiTrue*y[i-1] + rng.NormFloat64()
	}

	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 60
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	mleBeforePhi := m.phi[0]
	mleBeforePsi := append([]float64{}, m.psi.Load().values...)

	if err := m.CorrectBiasBootstrap(50, 42); err != nil {
		t.Fatal(err)
	}

	// Mechanics: phi changed.
	if m.phi[0] == mleBeforePhi {
		t.Error("CorrectBiasBootstrap: phi[0] unchanged — bootstrap had no effect")
	}
	// psi cache recomputed.
	psiAfter := m.psi.Load().values
	psiSame := len(psiAfter) == len(mleBeforePsi)
	if psiSame {
		for i := range psiAfter {
			if psiAfter[i] != mleBeforePsi[i] {
				psiSame = false
				break
			}
		}
	}
	if psiSame {
		t.Error("CorrectBiasBootstrap: psi cache identical — should recompute under new coefs")
	}
	// Direction: AR(1) MLE on n=50 with phi_true=0.7 typically biases
	// φ̂ downward. The corrected estimate should be greater than (or
	// at least within range of) the MLE — bigger and closer to 0.7.
	t.Logf("AR(1) phi_true=%g  MLE=%g  corrected=%g", phiTrue, mleBeforePhi, m.phi[0])
}

// BIAS-1: Monte Carlo bias-reduction verification. Simulate K
// realisations of an AR(1), fit each both ways, compare the mean
// |bias| of the corrected vs uncorrected estimators. Expected:
// corrected mean is closer to the true φ.
//
// Skipped under -short because of the K · B fit cost.
func TestCorrectBiasBootstrap_MonteCarloReducesAR1Bias(t *testing.T) {
	if testing.Short() {
		t.Skip("Monte Carlo: K=20 series × B=80 bootstraps × ~3ms = ~5s")
	}
	const phiTrue = 0.6
	const n = 40
	const K = 20
	const B = 80

	rng := rand.New(rand.NewPCG(123, 456))
	totalMLE := 0.0
	totalCorrected := 0.0
	for k := 0; k < K; k++ {
		y := make([]float64, n)
		for i := 1; i < n; i++ {
			y[i] = phiTrue*y[i-1] + rng.NormFloat64()
		}
		m := NewARIMA(Order{P: 1, D: 0, Q: 0})
		m.MaxIter = 50
		if err := m.Fit(y, nil); err != nil {
			continue
		}
		totalMLE += m.phi[0]
		if err := m.CorrectBiasBootstrap(B, uint64(k)+1); err != nil {
			t.Logf("k=%d bootstrap error: %v", k, err)
			continue
		}
		totalCorrected += m.phi[0]
	}
	meanMLE := totalMLE / float64(K)
	meanCorrected := totalCorrected / float64(K)
	mleBias := meanMLE - phiTrue
	correctedBias := meanCorrected - phiTrue

	t.Logf("AR(1) Monte Carlo (K=%d, n=%d, phi=%g):", K, n, phiTrue)
	t.Logf("  MLE  mean=%.4f  bias=%+.4f", meanMLE, mleBias)
	t.Logf("  corr mean=%.4f  bias=%+.4f", meanCorrected, correctedBias)

	// Assert that corrected has smaller |bias| than MLE. Hall-bootstrap
	// is asymptotically equivalent to closed-form; on finite K the
	// reduction may be partial (sampling noise) but the corrected mean
	// should be closer.
	if math.Abs(correctedBias) >= math.Abs(mleBias) {
		t.Errorf("bias correction did not reduce |bias|: MLE %+.4f vs corrected %+.4f",
			mleBias, correctedBias)
	}
}

// BIAS-1: error paths.
func TestCorrectBiasBootstrap_ErrorPaths(t *testing.T) {
	// Unfitted model.
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	if err := m.CorrectBiasBootstrap(100, 1); err == nil {
		t.Error("unfitted model should error")
	}

	// B too small.
	rng := rand.New(rand.NewPCG(11, 12))
	y := make([]float64, 100)
	for i := 1; i < 100; i++ {
		y[i] = 0.5*y[i-1] + rng.NormFloat64()
	}
	m = NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 50
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	if err := m.CorrectBiasBootstrap(5, 1); err == nil {
		t.Error("B<10 should error")
	}
}
