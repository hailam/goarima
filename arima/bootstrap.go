package arima

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"sort"
	"sync"
)

// BootResult holds nSim simulated forecast paths plus bootstrap quantile CIs.
type BootResult struct {
	Mean     []float64   // mean of simulated paths per horizon
	Variance []float64   // empirical variance of paths per horizon (PRED-VAR)
	Lower    []float64   // alpha/2 quantile per horizon
	Upper    []float64   // 1-alpha/2 quantile per horizon
	Paths    [][]float64 // nSim x nPeriods raw simulated paths
}

// PredictBootOpts configures PredictBootWithOpts. Default zero-value
// matches the legacy PredictBoot signature: non-parametric residual
// resampling, alpha=0.05 (when explicitly set), nSim=1000.
type PredictBootOpts struct {
	// Alpha is the two-sided coverage error rate. Lower/Upper bounds
	// are the alpha/2 and 1-alpha/2 empirical quantiles. Required:
	// must be in (0, 1).
	Alpha float64

	// NSim is the number of bootstrap paths to generate. ≤0 → 1000.
	NSim int

	// Seed for the per-path PCG RNG. Different seeds give different
	// path realisations; same seed gives bit-identical results across
	// runs and across worker counts (per-path PCG re-seed via golden-
	// ratio salt).
	Seed uint64

	// FutureExog is an [nPeriods][k] matrix of regressors aligned with
	// the forecast horizon. Required when the model was fitted with
	// exog, must be nil when it wasn't. Drift-augmented automatically
	// when m.DriftIncluded.
	FutureExog [][]float64

	// Parametric, when true, draws innovations from N(0, σ²) instead
	// of resampling empirical residuals. Useful when:
	//   - residuals are approximately Gaussian (the common case when
	//     AR/MA structure was correctly specified)
	//   - the residual sample is small (n<50) — empirical sampling is
	//     unstable when the resample pool is tiny
	//   - users want symmetric CIs that don't inherit residual
	//     skewness or thin tails
	//
	// Default false → empirical residual resampling (legacy behaviour,
	// matches pmdarima.ARIMA.predict(bootstrap=True)).
	Parametric bool
}

// PredictBoot is the legacy positional-arg entry point. Equivalent to
// PredictBootWithOpts with non-parametric (residual-resampling) mode.
// New code should prefer PredictBootWithOpts which supports the
// parametric option (innovations drawn from N(0, σ²)) and is easier
// to extend.
//
// Approach: for each simulation, recursively forecast nPeriods ahead
// while drawing innovations uniformly with replacement from the in-
// sample residuals. Returns the per-step empirical mean and (alpha/2,
// 1-alpha/2) quantiles across nSim paths. Mirrors
// pmdarima.ARIMA.predict(return_conf_int=True, bootstrap=True).
//
// Useful when the residuals are non-Gaussian and the standard
// parametric CIs (Predict's built-in alpha) misstate uncertainty.
func (m *ARIMA) PredictBoot(nPeriods int, alpha float64, nSim int, seed uint64, futureExog [][]float64) (*BootResult, error) {
	return m.PredictBootWithOpts(nPeriods, PredictBootOpts{
		Alpha:      alpha,
		NSim:       nSim,
		Seed:       seed,
		FutureExog: futureExog,
		Parametric: false,
	})
}

// PredictBootWithOpts produces bootstrap forecast intervals. See
// PredictBootOpts for configuration. Set opts.Parametric=true to draw
// innovations from N(0, σ²) instead of resampling empirical residuals
// — useful on short series and when residuals are approximately
// Gaussian.
func (m *ARIMA) PredictBootWithOpts(nPeriods int, opts PredictBootOpts) (*BootResult, error) {
	alpha := opts.Alpha
	nSim := opts.NSim
	seed := opts.Seed
	futureExog := opts.FutureExog
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
	// the CSS recursion left in place for warm-up indices. Skipped in the
	// parametric branch where we draw from N(0, σ²) directly.
	var residPool []float64
	if !opts.Parametric {
		residPool = make([]float64, 0, len(m.resids))
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
	}
	// Pre-compute σ for the parametric branch (innovations ~ N(0, σ²)).
	sigma := 0.0
	if opts.Parametric {
		if m.sigma2 <= 0 {
			return nil, errors.New("model sigma2 ≤ 0; cannot draw parametric innovations")
		}
		sigma = math.Sqrt(m.sigma2)
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

	// CDX-4: flat backing storage for paths. paths[s] aliases a row of
	// `pathsFlat`, which is one contiguous allocation instead of nSim
	// fresh per-path slices. Combines naturally with CDX-2's in-place
	// integrateBackTail and the new in-place box-cox invert below.
	pathsFlat := make([]float64, nSim*nPeriods)
	paths := make([][]float64, nSim)
	for s := 0; s < nSim; s++ {
		paths[s] = pathsFlat[s*nPeriods : (s+1)*nPeriods]
	}

	// simulateOneInto writes simulation s's output path directly into
	// `dst` (length nPeriods, sliced from pathsFlat). PCG seeded from
	// (seed, s) so paths[s] is deterministic regardless of worker
	// assignment.
	const seedSalt = uint64(0x9E3779B97F4A7C15) // golden-ratio salt
	simulateOneInto := func(s int, pcg *rand.PCG, hist, residHist, dst []float64) {
		pcg.Seed(seed^seedSalt, uint64(s)+1)
		rng := rand.New(pcg)
		hist = append(hist[:0], phiWin...)
		residHist = append(residHist[:0], thetaWin...)
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
			var eps float64
			if opts.Parametric {
				eps = rng.NormFloat64() * sigma
			} else {
				eps = residPool[rng.IntN(len(residPool))]
			}
			yh += eps
			dst[h] = yh
			hist = append(hist, yh)
			residHist = append(residHist, eps)
		}
		// Re-add intercept, mean, and exog contribution on differenced scale.
		for i := range dst {
			dst[i] += m.mean + m.c
			if futureWX != nil {
				for j, b := range m.beta {
					dst[i] += b * futureWX[i][j]
				}
			}
		}
		// CDX-2: integrateBackTail writes the forecast region directly
		// into dst (no alloc). Aliasing-safe.
		if seasHead != nil {
			integrateBackTail(dst, dst, seasHead, m.Seasonal.M, m.Seasonal.D)
		}
		if nonSeasHead != nil {
			integrateBackTail(dst, dst, nonSeasHead, 1, m.Order.D)
		}
		// CDX-4: in-place box-cox invert (no alloc).
		if m.Lambda != nil {
			boxCoxInvertInto(dst, dst, *m.Lambda, m.Lambda2)
		}
	}

	// Worker count: dispatch to goroutines once nSim is big enough that the
	// per-path simulation work amortises goroutine setup. Below the threshold
	// we stay single-threaded — overhead would dominate.
	nWorkers := runtime.GOMAXPROCS(0)
	if nWorkers > nSim {
		nWorkers = nSim
	}
	const parallelThreshold = 64
	if nSim < parallelThreshold || nWorkers < 2 {
		// Serial path — one PCG, one set of scratch buffers, processes all paths.
		pcg := rand.NewPCG(0, 0)
		hist := make([]float64, 0, pLag+nPeriods)
		residHist := make([]float64, 0, qLag+nPeriods)
		for s := 0; s < nSim; s++ {
			simulateOneInto(s, pcg, hist, residHist, paths[s])
		}
	} else {
		// Parallel path — each worker owns its scratch buffers and PCG;
		// paths are partitioned by index modulo nWorkers. Each goroutine
		// writes only to its own paths[s] rows (sliced from pathsFlat),
		// so no lock is needed.
		var wg sync.WaitGroup
		for w := 0; w < nWorkers; w++ {
			w := w
			wg.Add(1)
			go func() {
				defer wg.Done()
				pcg := rand.NewPCG(0, 0)
				hist := make([]float64, 0, pLag+nPeriods)
				residHist := make([]float64, 0, qLag+nPeriods)
				for s := w; s < nSim; s += nWorkers {
					simulateOneInto(s, pcg, hist, residHist, paths[s])
				}
			}()
		}
		wg.Wait()
	}

	// Per-step mean and quantiles. Reuse the col buffer in place: each
	// horizon overwrites it from paths[s][h], computes the mean, then
	// sorts col in-place (we don't need the unsorted order again — the
	// next horizon overwrites col entirely). Saves an nSim-float alloc
	// per horizon vs `sorted := append([]float64{}, col...)`.
	mean := make([]float64, nPeriods)
	variance := make([]float64, nPeriods)
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
		// Empirical variance (sample, divisor n−1). PRED-VAR exposes
		// this so callers don't have to recompute from Paths.
		ssq := 0.0
		for _, v := range col {
			d := v - mean[h]
			ssq += d * d
		}
		if nSim > 1 {
			variance[h] = ssq / float64(nSim-1)
		}
		sort.Float64s(col)
		lower[h] = quantile(col, loQ)
		upper[h] = quantile(col, hiQ)
	}
	return &BootResult{
		Mean:     mean,
		Variance: variance,
		Lower:    lower,
		Upper:    upper,
		Paths:    paths,
	}, nil
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
