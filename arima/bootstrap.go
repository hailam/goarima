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
	// Drift auto-extension (mirrors Predict) — prepend the linear time index
	// before validating widths.
	var driftErr error
	futureExog, driftErr = m.extendDriftIfNeeded(futureExog, nPeriods)
	if driftErr != nil {
		return nil, driftErr
	}
	if m.nExog > 0 {
		if futureExog == nil || len(futureExog) != nPeriods {
			return nil, fmt.Errorf("future exog rows (%d) must match nPeriods (%d)", len(futureExog), nPeriods)
		}
		for i, row := range futureExog {
			if len(row) != m.nExog {
				return nil, fmt.Errorf("future exog row %d cols (%d) must match training (%d)",
					i, len(row), m.nExog)
			}
		}
	} else if futureExog != nil {
		return nil, errors.New("model was fitted without exog; do not pass futureExog")
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

	fullPhi := expandSARMA(m.phi, m.Phi, m.Seasonal.M)
	fullTheta := expandSMA(m.theta, m.Theta, m.Seasonal.M)

	// Pull cached differenced+centered training series + residuals from Fit
	// (set by m.wsCenteredCache / m.resids). Avoids re-running applyDiff +
	// armaCSS on every PredictBoot call — important when nSim is large.
	wsCentered := m.wsCenteredCache
	baseRes := m.resids

	// Difference combined exog (one-time cost; not in the per-sim loop).
	// Trim m.xTrain to the last `dHead = D + D*M` rows of historical context
	// — that's all the differencing operator needs. Same fix as #C17 for
	// Predict; pre-fix bootstrap copied the full training xTrain on every
	// PredictBoot call, which adds up fast on long-history workloads.
	var futureWX [][]float64
	if futureExog != nil {
		dHead := m.Order.D
		if m.Seasonal.Active() {
			dHead += m.Seasonal.D * m.Seasonal.M
		}
		if dHead == 0 {
			futureWX = futureExog
		} else {
			trimStart := len(m.xTrain) - dHead
			if trimStart < 0 {
				trimStart = 0
			}
			combined := make([][]float64, 0, dHead+nPeriods)
			combined = append(combined, m.xTrain[trimStart:]...)
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
	}

	// Pre-compute integration heads once — they're identical for every sim.
	// Use the *model-scale* yTrain (Box-Cox-applied if Lambda is set) so the
	// integration is unit-consistent with the simulated series, which lives
	// on the model scale. Output paths are inverse-Box-Cox'd at the end of
	// each simulation so users see original units.
	yMS := m.yMSCache
	if yMS == nil {
		yMS = m.yTrain // older snapshots without cache; Lambda must be nil
	}
	var seasHead, nonSeasHead []float64
	if m.Seasonal.Active() && m.Seasonal.D > 0 {
		seasHead = lastN(diffStream(yMS, 1, m.Order.D), m.Seasonal.D*m.Seasonal.M)
	}
	if m.Order.D > 0 {
		nonSeasHead = lastN(yMS, m.Order.D)
	}

	// Per-sim, the AR/MA forecast loop only reads the last len(fullPhi) /
	// len(fullTheta) elements of the history. Copying the entire training
	// series per simulation (the old code) was wasted work — extract the
	// lag windows once and clone just those.
	pLag := len(fullPhi)
	qLag := len(fullTheta)
	phiWin := lastN(wsCentered, pLag)
	thetaWin := lastN(baseRes, qLag)

	rng := rand.New(rand.NewPCG(seed, seed+1))
	paths := make([][]float64, nSim)
	// Reusable per-sim buffers (allocated once, overwritten each iteration).
	hist := make([]float64, 0, pLag+nPeriods)
	residHist := make([]float64, 0, qLag+nPeriods)
	for s := 0; s < nSim; s++ {
		hist = append(hist[:0], phiWin...)
		residHist = append(residHist[:0], thetaWin...)
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
		// Integrate back through differencing using the pre-computed heads.
		out := simDiffed
		if seasHead != nil {
			full := integrateBack(out, seasHead, m.Seasonal.M, m.Seasonal.D)
			out = full[len(seasHead):]
		}
		if nonSeasHead != nil {
			full := integrateBack(out, nonSeasHead, 1, m.Order.D)
			out = full[len(nonSeasHead):]
		}
		// Inverse Box-Cox so paths land in the user's original units. Mirrors
		// the same step at the end of Predict.
		if m.Lambda != nil {
			out = boxCoxInvert(out, *m.Lambda, m.Lambda2)
		}
		paths[s] = out
	}

	// Per-step mean and quantiles. Reuse the col buffer in place: each
	// horizon overwrites it from paths[s][h], computes the mean, then
	// sorts col in-place (we don't need the unsorted order again — the
	// next horizon overwrites col entirely). Saves an nSim-float alloc
	// per horizon vs `sorted := append([]float64{}, col...)`.
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
		sort.Float64s(col)
		lower[h] = quantile(col, loQ)
		upper[h] = quantile(col, hiQ)
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
