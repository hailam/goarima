package arima

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"time"
)

// Simulate returns n samples drawn from the fitted ARIMA process.
//
// burnIn discards transient initial samples to escape the zero-state startup
// effect; pass 0 to use the default of 100. seed pins the RNG (math/rand/v2
// PCG); pass 0 for a time-based seed (non-deterministic).
//
// futureExog must have exactly n rows if the model was fit with exogenous
// regressors. Rows must each have the same width as during Fit.
//
// Output is in the model's original units. If Box-Cox was applied during
// Fit (Lambda != nil), the simulated series on the model scale is
// inverse-transformed back.
//
// Mirrors statsmodels' SARIMAX.simulate (default settings) and R's
// arima.sim. The simulation uses the fitted phi, theta, sigma², c, beta,
// and any seasonal counterparts. Differencing is integrated back with a
// zero historical head so the simulation represents a "fresh start" from
// the stationary distribution rather than a continuation of yTrain.
func (m *ARIMA) Simulate(n, burnIn int, seed uint64, futureExog [][]float64) ([]float64, error) {
	if !m.fitted {
		return nil, errors.New("arima: cannot simulate from unfitted model — call Fit first")
	}
	if n <= 0 {
		return nil, errors.New("arima: n must be positive")
	}
	if burnIn < 0 {
		return nil, errors.New("arima: burnIn must be non-negative")
	}
	if burnIn == 0 {
		burnIn = 100
	}
	// Drift auto-extension — same as Predict/PredictBoot.
	futureExog = m.extendDriftIfNeeded(futureExog, n)
	if m.nExog > 0 {
		if futureExog == nil {
			return nil, errors.New("arima: futureExog required (model was fit with exog)")
		}
		if len(futureExog) != n {
			return nil, fmt.Errorf("arima: futureExog rows (%d) must equal n (%d)", len(futureExog), n)
		}
		for i, row := range futureExog {
			if len(row) != m.nExog {
				return nil, fmt.Errorf("arima: futureExog[%d] cols (%d) must match model nExog (%d)",
					i, len(row), m.nExog)
			}
		}
	} else if futureExog != nil {
		return nil, errors.New("arima: model was fit without exog; do not pass futureExog")
	}

	if seed == 0 {
		seed = uint64(time.Now().UnixNano())
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9E3779B97F4A7C15))

	fullPhi := expandSARMA(m.phi, m.Phi, m.Seasonal.M)
	fullTheta := expandSMA(m.theta, m.Theta, m.Seasonal.M)
	sigma := safeSqrt(m.sigma2)

	// ARMA recursion on the differenced+centered scale. We need (n + burnIn)
	// samples plus enough history for AR/MA lags.
	total := n + burnIn
	maxLag := len(fullPhi)
	if len(fullTheta) > maxLag {
		maxLag = len(fullTheta)
	}
	hist := make([]float64, total+maxLag)
	innov := make([]float64, total+maxLag)

	for t := maxLag; t < total+maxLag; t++ {
		e := sigma * rng.NormFloat64()
		innov[t] = e
		v := e
		for i, ph := range fullPhi {
			v += ph * hist[t-1-i]
		}
		for j, th := range fullTheta {
			v += th * innov[t-1-j]
		}
		hist[t] = v
	}
	// Drop burn-in and pre-history.
	wsSim := hist[maxLag+burnIn:]
	// wsSim now has exactly n samples on the differenced+centered scale.

	// Re-add intercept and exog contribution (on the differenced scale).
	for i := range wsSim {
		wsSim[i] += m.mean + m.c
		if m.nExog > 0 {
			for j, b := range m.beta {
				wsSim[i] += b * futureExog[i][j]
			}
		}
	}

	// Integrate back any differencing — start from a zero historical head
	// so the simulation represents a fresh draw, not a continuation of yTrain.
	out := wsSim
	if m.Seasonal.Active() && m.Seasonal.D > 0 {
		head := make([]float64, m.Seasonal.D*m.Seasonal.M)
		full := integrateBack(out, head, m.Seasonal.M, m.Seasonal.D)
		out = full[len(head):]
	}
	if m.Order.D > 0 {
		head := make([]float64, m.Order.D)
		full := integrateBack(out, head, 1, m.Order.D)
		out = full[len(head):]
	}

	// Invert Box-Cox if applied during Fit.
	if m.Lambda != nil {
		out = boxCoxInvert(out, *m.Lambda, m.Lambda2)
	}
	return out, nil
}

// safeSqrt returns sqrt(x), clamping tiny negative values from numerical
// roundoff to zero. σ² should never be materially negative; if it is
// (presumably a bug elsewhere) we still return 0 rather than NaN-poisoning
// every simulated sample.
func safeSqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return math.Sqrt(x)
}
