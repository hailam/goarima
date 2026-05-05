package arima

import (
	"errors"
	"fmt"
)

// CorrectBiasBootstrap applies Hall (1992) parametric-bootstrap bias
// correction to the fitted ARIMA parameters, in place. After this
// call, m.phi / m.theta / m.Phi / m.Theta / m.c / m.beta hold
// bias-corrected estimates; m.psi cache is recomputed under the new
// coefficients.
//
// Algorithm
//
//   1. Snapshot the fitted coefficients θ̂.
//   2. For b = 1..B:
//      - Simulate a series of length n from the current fit (parametric
//        bootstrap; reuses m.Simulate).
//      - Re-fit a fresh ARIMA of the same shape on the simulated series.
//      - Record the bootstrapped coefficients θ_b.
//   3. Compute the bias estimate: bias_hat = mean(θ_b) − θ̂.
//   4. Apply correction: θ_corrected = θ̂ − bias_hat = 2·θ̂ − mean(θ_b).
//   5. Clip to stationarity / invertibility (|coef| ≤ 0.99) for safety.
//   6. Recompute m.psi cache.
//
// Under the bootstrap principle, the bootstrap world's bias (θ_b vs θ̂)
// estimates the real-world bias (θ̂ vs θ_true). Subtracting that
// estimate from θ̂ removes the leading-order finite-sample bias, which
// MLE famously suffers from on short series (e.g. AR(1) MLE biases φ̂
// downward by ~−2/n).
//
// Caveats
//
//   - m.LogLikelihood() and m.Sigma2() are NOT recomputed. They reflect
//     the original MLE fit, since bias-corrected estimates aren't ML
//     estimates anymore. AIC / AICc / BIC use the original logL.
//   - Only the coefficient vector is corrected. σ² has its own
//     small-n bias (Bartlett-type, factor n/(n−k)) that's not
//     addressed here.
//   - Cost: B fresh Fit calls. With B=200 and our post-G-NEW-2 /
//     KAL-WORKSPACE optimizations (~3 ms per default Fit), ~600 ms
//     total. Use sparingly; opt-in by design.
//   - Correction may push coefficients toward the boundary; clipping
//     to ±0.99 is the safe default. If the correction would push past
//     +1, the clip leaves the coefficient at +0.99, which is still an
//     improvement over the un-corrected MLE estimate.
//
// Closes BIAS-1.
func (m *ARIMA) CorrectBiasBootstrap(B int, seed uint64) error {
	if !m.fitted {
		return errors.New("arima: model not fitted; call Fit before CorrectBiasBootstrap")
	}
	if B < 10 {
		return errors.New("arima: B must be ≥ 10 (typical: 200-1000)")
	}
	n := len(m.yTrain)
	if n == 0 {
		return errors.New("arima: yTrain is empty")
	}

	// Snapshot the original fitted coefficients (the θ̂ we're correcting).
	origPhi := append([]float64{}, m.phi...)
	origTheta := append([]float64{}, m.theta...)
	origSPhi := append([]float64{}, m.Phi...)
	origSTheta := append([]float64{}, m.Theta...)
	origC := m.c
	origBeta := append([]float64{}, m.beta...)

	// Accumulators for the mean-of-bootstraps.
	sumPhi := make([]float64, len(origPhi))
	sumTheta := make([]float64, len(origTheta))
	sumSPhi := make([]float64, len(origSPhi))
	sumSTheta := make([]float64, len(origSTheta))
	sumC := 0.0
	sumBeta := make([]float64, len(origBeta))

	// Use the same exog as training so each bootstrap fit operates on
	// the same regressor structure. nil for non-exog models.
	var simExog [][]float64
	if m.nExog > 0 {
		simExog = m.xTrain
	}

	nUsed := 0
	for b := 0; b < B; b++ {
		simY, err := m.Simulate(n, SimulateOpts{
			Seed:       seed + uint64(b) + 1,
			FutureExog: simExog,
		})
		if err != nil {
			continue
		}
		candidate := NewARIMA(m.Order)
		candidate.Seasonal = m.Seasonal
		candidate.WithIntercept = m.WithIntercept
		candidate.Method = m.Method
		candidate.MaxIter = m.MaxIter
		candidate.NonSimpleDifferencing = m.NonSimpleDifferencing
		candidate.DiffuseConvention = m.DiffuseConvention
		if m.Lambda != nil {
			lam := *m.Lambda
			candidate.Lambda = &lam
			candidate.Lambda2 = m.Lambda2
		}
		candidate.RidgePenalty = m.RidgePenalty

		fitExog := m.xTrain
		if m.nExog == 0 {
			fitExog = nil
		}
		if err := candidate.Fit(simY, fitExog); err != nil {
			continue
		}

		for i := range candidate.phi {
			sumPhi[i] += candidate.phi[i]
		}
		for i := range candidate.theta {
			sumTheta[i] += candidate.theta[i]
		}
		for i := range candidate.Phi {
			sumSPhi[i] += candidate.Phi[i]
		}
		for i := range candidate.Theta {
			sumSTheta[i] += candidate.Theta[i]
		}
		sumC += candidate.c
		for i := range candidate.beta {
			sumBeta[i] += candidate.beta[i]
		}
		nUsed++
	}

	if nUsed < B/2 {
		return fmt.Errorf("arima: only %d/%d bootstrap fits succeeded; bias correction unreliable", nUsed, B)
	}
	nf := float64(nUsed)

	clip := func(v, lo, hi float64) float64 {
		if v < lo {
			return lo
		}
		if v > hi {
			return hi
		}
		return v
	}

	// θ_corrected = 2·θ̂ − mean(θ_b). AR/MA coefs clipped to ±0.99 for
	// stationarity / invertibility safety. Intercept and exog β aren't
	// clipped — they have no boundary constraint.
	for i := range m.phi {
		m.phi[i] = clip(2*origPhi[i]-sumPhi[i]/nf, -0.99, 0.99)
	}
	for i := range m.theta {
		m.theta[i] = clip(2*origTheta[i]-sumTheta[i]/nf, -0.99, 0.99)
	}
	for i := range m.Phi {
		m.Phi[i] = clip(2*origSPhi[i]-sumSPhi[i]/nf, -0.99, 0.99)
	}
	for i := range m.Theta {
		m.Theta[i] = clip(2*origSTheta[i]-sumSTheta[i]/nf, -0.99, 0.99)
	}
	m.c = 2*origC - sumC/nf
	for i := range m.beta {
		m.beta[i] = 2*origBeta[i] - sumBeta[i]/nf
	}

	// Recompute psi cache under the new coefficients so Predict's CI
	// bands and m.PredictVar reflect the corrected coefs.
	m.computePsi()
	return nil
}
