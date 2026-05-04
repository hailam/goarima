package arima

import (
	"errors"
	"fmt"
	"math"

	"gonum.org/v1/gonum/stat/distuv"
)

// LjungBox runs the Ljung–Box portmanteau test for serial correlation in
// the residual series. Returns Q (the test statistic) and p (the p-value).
//
//	Q = n (n + 2) · Σ_{k=1..h} ρ_k² / (n − k)
//	df = h − modelOrder    (clamped to a minimum of 1)
//	p = 1 − F_χ²(Q; df)
//
// where ρ_k is the sample autocorrelation at lag k. Mirrors R's
// `Box.test(x, lag=h, type="Ljung-Box", fitdf=modelOrder)` and FPP3 §7.3.
//
// `modelOrder` should be the total ARMA degrees of freedom of the fitted
// model: p + q + P + Q (sum of non-seasonal and seasonal AR/MA orders).
// Passing 0 makes the test ignore the model-fit adjustment, suitable for
// raw-data tests.
//
// Input residuals are filtered for NaN / Inf — goarima's Resid() returns
// NaN-padded output for the differencing-warmup region, so callers can
// pass m.Resid() directly without preprocessing.
//
// Returns an error when fewer than 4 finite residuals remain after filtering
// or when h ≥ len(filtered).
func LjungBox(resid []float64, h, modelOrder int) (Q, p float64, err error) {
	r, n, err := residualACFCommon(resid, h, modelOrder)
	if err != nil {
		return 0, 0, err
	}
	Q = 0
	for k := 1; k <= h; k++ {
		Q += r[k] * r[k] / float64(n-k)
	}
	Q *= float64(n) * float64(n+2)
	p = chiSqUpperTail(Q, h, modelOrder)
	return Q, p, nil
}

// BoxPierce runs the Box–Pierce portmanteau test, an older simpler variant
// of Ljung–Box. Same null hypothesis (residuals are white noise) but uses
// the unweighted sum of squared autocorrelations:
//
//	Q = n · Σ_{k=1..h} ρ_k²
//
// Ljung–Box generally has better small-sample behavior; Box–Pierce is
// kept for parity with R's `Box.test(..., type="Box-Pierce")` and older
// references that report it.
func BoxPierce(resid []float64, h, modelOrder int) (Q, p float64, err error) {
	r, n, err := residualACFCommon(resid, h, modelOrder)
	if err != nil {
		return 0, 0, err
	}
	Q = 0
	for k := 1; k <= h; k++ {
		Q += r[k] * r[k]
	}
	Q *= float64(n)
	p = chiSqUpperTail(Q, h, modelOrder)
	return Q, p, nil
}

// LjungBox runs the Ljung–Box test on this fitted model's residuals. The
// effective `modelOrder` is auto-derived from the fitted ARMA orders
// (p + q + P + Q). Convenience wrapper around the package-level function.
func (m *ARIMA) LjungBox(h int) (Q, p float64, err error) {
	if !m.fitted {
		return 0, 0, errors.New("arima: model not fitted")
	}
	return LjungBox(m.Resid(), h, m.armaDoF())
}

// BoxPierce runs the Box–Pierce test on this fitted model's residuals.
func (m *ARIMA) BoxPierce(h int) (Q, p float64, err error) {
	if !m.fitted {
		return 0, 0, errors.New("arima: model not fitted")
	}
	return BoxPierce(m.Resid(), h, m.armaDoF())
}

// armaDoF returns p + q + P + Q — the count of fitted ARMA parameters
// used to adjust df in the portmanteau test.
func (m *ARIMA) armaDoF() int {
	d := m.Order.P + m.Order.Q
	if m.Seasonal.Active() {
		d += m.Seasonal.P + m.Seasonal.Q
	}
	return d
}

// residualACFCommon validates the input, drops NaN/Inf, and returns the
// sample autocorrelations up to lag h plus the cleaned series length.
// Centralizes the prep step shared by LjungBox and BoxPierce.
func residualACFCommon(resid []float64, h, modelOrder int) ([]float64, int, error) {
	if h < 1 {
		return nil, 0, errors.New("arima: h must be >= 1")
	}
	if modelOrder < 0 {
		return nil, 0, errors.New("arima: modelOrder must be >= 0")
	}
	clean := make([]float64, 0, len(resid))
	for _, v := range resid {
		if !math.IsNaN(v) && !math.IsInf(v, 0) {
			clean = append(clean, v)
		}
	}
	n := len(clean)
	if n < 4 {
		return nil, 0, fmt.Errorf("arima: need at least 4 finite residuals, got %d", n)
	}
	if h >= n {
		return nil, 0, fmt.Errorf("arima: h (%d) must be < len(residuals) (%d)", h, n)
	}

	// Sample autocorrelation at lag k = sum_{t=k+1..n} (r_t - mean)(r_{t-k} - mean) / sum_t (r_t - mean)^2
	mean := 0.0
	for _, v := range clean {
		mean += v
	}
	mean /= float64(n)
	dev := make([]float64, n)
	denom := 0.0
	for i, v := range clean {
		dev[i] = v - mean
		denom += dev[i] * dev[i]
	}
	if denom == 0 {
		return nil, 0, errors.New("arima: residual variance is zero")
	}
	rho := make([]float64, h+1)
	for k := 1; k <= h; k++ {
		num := 0.0
		for t := k; t < n; t++ {
			num += dev[t] * dev[t-k]
		}
		rho[k] = num / denom
	}
	return rho, n, nil
}

// chiSqUpperTail returns 1 - F_χ²(Q; df) where df = max(1, h - modelOrder).
func chiSqUpperTail(Q float64, h, modelOrder int) float64 {
	df := h - modelOrder
	if df < 1 {
		df = 1
	}
	chi := distuv.ChiSquared{K: float64(df)}
	return 1 - chi.CDF(Q)
}
