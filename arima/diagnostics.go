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

// LjungBoxWithDF runs the Ljung-Box test with an explicit fitdf override
// (model degrees-of-freedom adjustment). The default `m.LjungBox(h)`
// auto-derives fitdf from `p+q+P+Q`; this variant lets callers match
// R's `Box.test(resid, lag=h, fitdf=...)` when they need to override
// (e.g. when comparing residuals from a model fit elsewhere or when
// reproducing a published test where fitdf was set differently).
//
// Closes GAP-5.
func (m *ARIMA) LjungBoxWithDF(h, fitdf int) (Q, p float64, err error) {
	if !m.fitted {
		return 0, 0, errors.New("arima: model not fitted")
	}
	return LjungBox(m.Resid(), h, fitdf)
}

// BoxPierceWithDF is the explicit-fitdf variant of m.BoxPierce(h).
// See LjungBoxWithDF for rationale.
func (m *ARIMA) BoxPierceWithDF(h, fitdf int) (Q, p float64, err error) {
	if !m.fitted {
		return 0, 0, errors.New("arima: model not fitted")
	}
	return BoxPierce(m.Resid(), h, fitdf)
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
	clean := dropNonFinite(resid)
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

// JarqueBera tests the null hypothesis that the residual series is drawn
// from a normal distribution. Returns the JB statistic and its p-value.
//
//	JB = (n / 6) · (S² + (K − 3)² / 4)
//
// where S is sample skewness and K is sample kurtosis (raw — not the
// "excess" kurtosis that stats books sometimes use). Under H0 (normality),
// JB ~ χ²(2). A small p-value rejects normality.
//
// NaN/Inf inputs are filtered (consistent with LjungBox / BoxPierce so
// callers can pass `m.Resid()` directly). Returns an error if fewer than
// 4 finite residuals remain or if all residuals are identical.
//
// Mirrors `statsmodels.stats.stattools.jarque_bera`.
func JarqueBera(resid []float64) (jb, pValue, skewness, kurtosis float64, err error) {
	clean := dropNonFinite(resid)
	n := len(clean)
	if n < 4 {
		return 0, 0, 0, 0, fmt.Errorf("arima: need at least 4 finite residuals, got %d", n)
	}
	mean := 0.0
	for _, v := range clean {
		mean += v
	}
	mean /= float64(n)
	var m2, m3, m4 float64
	for _, v := range clean {
		d := v - mean
		d2 := d * d
		m2 += d2
		m3 += d2 * d
		m4 += d2 * d2
	}
	m2 /= float64(n)
	m3 /= float64(n)
	m4 /= float64(n)
	if m2 <= 0 {
		return 0, 0, 0, 0, errors.New("arima: residuals have zero variance")
	}
	skewness = m3 / math.Pow(m2, 1.5)
	kurtosis = m4 / (m2 * m2)
	jb = float64(n) / 6.0 * (skewness*skewness + (kurtosis-3)*(kurtosis-3)/4.0)
	chi := distuv.ChiSquared{K: 2}
	pValue = 1 - chi.CDF(jb)
	return jb, pValue, skewness, kurtosis, nil
}

// JarqueBera runs the Jarque-Bera normality test on this fitted model's
// residuals. Convenience wrapper around the package-level function.
func (m *ARIMA) JarqueBera() (jb, pValue, skewness, kurtosis float64, err error) {
	if !m.fitted {
		return 0, 0, 0, 0, errors.New("arima: model not fitted")
	}
	return JarqueBera(m.Resid())
}

// ArchLM runs Engle's Lagrange-multiplier test for ARCH (autoregressive
// conditional heteroskedasticity) effects in the residual series.
// Returns the LM statistic and its p-value.
//
// Procedure: regress e²_t on a constant + e²_{t-1}, …, e²_{t-q} via OLS.
// LM = (n - q) · R². Under H0 (no ARCH at lags 1..q), LM ~ χ²(q). A small
// p-value indicates volatility clustering — squared residuals are
// autocorrelated, suggesting GARCH-style modelling could capture more
// of the dynamics.
//
// `q` is the number of lags to include (typical: 12 for monthly data,
// 5 for daily, depending on context).
//
// NaN/Inf inputs are filtered. Returns an error if `q < 1`, fewer than
// 2q+2 finite residuals remain, or the regression is rank-deficient.
//
// Mirrors `statsmodels.stats.diagnostic.het_arch`.
func ArchLM(resid []float64, q int) (lm, pValue float64, err error) {
	if q < 1 {
		return 0, 0, errors.New("arima: q must be >= 1")
	}
	clean := dropNonFinite(resid)
	n := len(clean)
	if n < 2*q+2 {
		return 0, 0, fmt.Errorf("arima: need at least %d finite residuals for q=%d, got %d", 2*q+2, q, n)
	}
	// Build squared-residual series and regress e²_t ~ const + e²_{t-1..t-q}.
	sq := make([]float64, n)
	for i, v := range clean {
		sq[i] = v * v
	}
	rows := n - q
	y := make([]float64, rows)
	X := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		t := q + i
		y[i] = sq[t]
		row := make([]float64, q)
		for k := 1; k <= q; k++ {
			row[k-1] = sq[t-k]
		}
		X[i] = row
	}
	beta, err := olsFit(X, y, true) // include intercept
	if err != nil {
		return 0, 0, fmt.Errorf("arima: ARCH-LM regression failed: %w", err)
	}
	// Compute R² = 1 - SSres/SStot.
	var ymean, ssTot, ssRes float64
	for _, v := range y {
		ymean += v
	}
	ymean /= float64(rows)
	for i, v := range y {
		// Predicted = beta[0] + Σ beta[k+1] * X[i][k] (since olsFit prepended intercept column).
		pred := beta[0]
		for k := 0; k < q; k++ {
			pred += beta[k+1] * X[i][k]
		}
		ssRes += (v - pred) * (v - pred)
		d := v - ymean
		ssTot += d * d
	}
	if ssTot == 0 {
		return 0, 0, errors.New("arima: squared residuals have zero variance")
	}
	r2 := 1 - ssRes/ssTot
	lm = float64(rows) * r2
	chi := distuv.ChiSquared{K: float64(q)}
	pValue = 1 - chi.CDF(lm)
	return lm, pValue, nil
}

// ArchLM runs Engle's ARCH-LM test on this fitted model's residuals.
func (m *ARIMA) ArchLM(q int) (lm, pValue float64, err error) {
	if !m.fitted {
		return 0, 0, errors.New("arima: model not fitted")
	}
	return ArchLM(m.Resid(), q)
}

// dropNonFinite returns a new slice with NaN/Inf entries removed,
// preserving order. Callers can pass goarima's NaN-padded Resid() output
// directly.
func dropNonFinite(xs []float64) []float64 {
	out := make([]float64, 0, len(xs))
	for _, v := range xs {
		if !math.IsNaN(v) && !math.IsInf(v, 0) {
			out = append(out, v)
		}
	}
	return out
}
