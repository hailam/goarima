package arima

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"sort"
)

// BootResult holds nSim simulated forecast paths plus bootstrap quantile CIs.
type BootResult struct {
	Mean  []float64   // mean of simulated paths
	Lower []float64   // alpha/2 quantile per horizon
	Upper []float64   // 1-alpha/2 quantile per horizon
	Paths [][]float64 // nSim x nPeriods raw simulated paths
}

// PredictBoot produces bootstrap forecast intervals via residual resampling.
//
// Approach: for each simulation, recursively forecast nPeriods ahead while
// drawing innovations uniformly with replacement from the in-sample residuals.
// Returns the per-step empirical mean and (alpha/2, 1-alpha/2) quantiles
// across nSim paths. Mirrors pmdarima.ARIMA.predict(return_conf_int=True,
// bootstrap=True).
//
// Useful when the residuals are non-Gaussian and the standard parametric
// CIs (Predict's built-in alpha) misstate uncertainty.
func (m *ARIMA) PredictBoot(nPeriods int, alpha float64, nSim int, seed uint64, futureExog [][]float64) (*BootResult, error) {
	if !m.fitted {
		return nil, errors.New("model not fitted")
	}
	if nPeriods <= 0 {
		return &BootResult{}, nil
	}
	if nSim <= 0 {
		nSim = 1000
	}
	if alpha <= 0 || alpha >= 1 {
		return nil, errors.New("alpha must be in (0,1)")
	}
	if m.nExog > 0 {
		if futureExog == nil || len(futureExog) != nPeriods {
			return nil, fmt.Errorf("future exog rows (%d) must match nPeriods (%d)", len(futureExog), nPeriods)
		}
	}

	// Pull residuals from the differenced fit, drop the leading zeros that
	// the CSS recursion left in place for warm-up indices.
	residPool := make([]float64, 0, len(m.resids))
	for _, r := range m.resids {
		if r != 0 {
			residPool = append(residPool, r)
		}
	}
	if len(residPool) == 0 {
		residPool = m.resids
	}
	if len(residPool) == 0 {
		return nil, errors.New("no residuals to bootstrap from")
	}

	// Reconstruct differenced (centered/intercept-adjusted) training series.
	fullPhi := expandSARMA(m.phi, m.Phi, m.Seasonal.M)
	fullTheta := expandSMA(m.theta, m.Theta, m.Seasonal.M)
	ws := append([]float64{}, m.yTrain...)
	if m.Order.D > 0 {
		ws = applyDiff(ws, 1, m.Order.D)
	}
	if m.Seasonal.Active() && m.Seasonal.D > 0 {
		ws = applyDiff(ws, m.Seasonal.M, m.Seasonal.D)
	}
	var wX [][]float64
	if m.xTrain != nil {
		wX = cloneMat(m.xTrain)
		if m.Order.D > 0 {
			wX = applyMatDiff(wX, 1, m.Order.D)
		}
		if m.Seasonal.Active() && m.Seasonal.D > 0 {
			wX = applyMatDiff(wX, m.Seasonal.M, m.Seasonal.D)
		}
	}
	wsCentered := make([]float64, len(ws))
	for i, v := range ws {
		r := v - m.mean - m.c
		if wX != nil {
			for j, b := range m.beta {
				r -= b * wX[i][j]
			}
		}
		wsCentered[i] = r
	}
	_, _, baseRes := armaCSS(wsCentered, fullPhi, fullTheta)

	// Difference combined exog.
	var futureWX [][]float64
	if futureExog != nil {
		combined := append([][]float64{}, m.xTrain...)
		combined = append(combined, futureExog...)
		diffed := combined
		if m.Order.D > 0 {
			diffed = applyMatDiff(diffed, 1, m.Order.D)
		}
		if m.Seasonal.Active() && m.Seasonal.D > 0 {
			diffed = applyMatDiff(diffed, m.Seasonal.M, m.Seasonal.D)
		}
		futureWX = diffed[len(diffed)-nPeriods:]
	}

	rng := rand.New(rand.NewPCG(seed, seed+1))
	paths := make([][]float64, nSim)
	for s := 0; s < nSim; s++ {
		// Working copy of the differenced state for this simulation.
		hist := append([]float64{}, wsCentered...)
		residHist := append([]float64{}, baseRes...)
		simDiffed := make([]float64, nPeriods)
		for h := 0; h < nPeriods; h++ {
			yh := 0.0
			for i, ph := range fullPhi {
				idx := len(hist) - 1 - i
				if idx >= 0 {
					yh += ph * hist[idx]
				}
			}
			for j, th := range fullTheta {
				idx := len(residHist) - 1 - j
				if idx >= 0 && idx < len(residHist) {
					yh += th * residHist[idx]
				}
			}
			// Inject a bootstrapped innovation.
			eps := residPool[rng.IntN(len(residPool))]
			yh += eps
			simDiffed[h] = yh
			hist = append(hist, yh)
			residHist = append(residHist, eps)
		}
		// Re-add intercept, mean, and exog contribution on differenced scale.
		for i := range simDiffed {
			simDiffed[i] += m.mean + m.c
			if futureWX != nil {
				for j, b := range m.beta {
					simDiffed[i] += b * futureWX[i][j]
				}
			}
		}
		// Integrate back through differencing.
		out := simDiffed
		if m.Seasonal.Active() && m.Seasonal.D > 0 {
			head := lastN(diffStream(m.yTrain, 1, m.Order.D), m.Seasonal.D*m.Seasonal.M)
			full := integrateBack(out, head, m.Seasonal.M, m.Seasonal.D)
			out = full[len(head):]
		}
		if m.Order.D > 0 {
			head := lastN(m.yTrain, m.Order.D)
			full := integrateBack(out, head, 1, m.Order.D)
			out = full[len(head):]
		}
		paths[s] = out
	}

	// Per-step mean and quantiles.
	mean := make([]float64, nPeriods)
	lower := make([]float64, nPeriods)
	upper := make([]float64, nPeriods)
	col := make([]float64, nSim)
	loQ := alpha / 2
	hiQ := 1 - alpha/2
	for h := 0; h < nPeriods; h++ {
		for s := 0; s < nSim; s++ {
			col[s] = paths[s][h]
		}
		sum := 0.0
		for _, v := range col {
			sum += v
		}
		mean[h] = sum / float64(nSim)
		sorted := append([]float64{}, col...)
		sort.Float64s(sorted)
		lower[h] = quantile(sorted, loQ)
		upper[h] = quantile(sorted, hiQ)
	}
	return &BootResult{Mean: mean, Lower: lower, Upper: upper, Paths: paths}, nil
}

// quantile linear-interpolates the q-th quantile of a sorted slice.
func quantile(sorted []float64, q float64) float64 {
	if q <= 0 {
		return sorted[0]
	}
	if q >= 1 {
		return sorted[len(sorted)-1]
	}
	idx := q * float64(len(sorted)-1)
	lo := int(idx)
	frac := idx - float64(lo)
	if lo+1 >= len(sorted) {
		return sorted[lo]
	}
	return sorted[lo]*(1-frac) + sorted[lo+1]*frac
}
