package utils

import (
	"errors"
	"math"
)

// ACF returns the sample autocorrelation function up to nLags (inclusive).
// Output length = nLags + 1; out[0] = 1.0 by definition.
//
// Mirrors statsmodels.tsa.stattools.acf with default unbiased=False.
//
//	acf(k) = sum_{t=k+1..n} (y_t - mean)(y_{t-k} - mean) / sum_{t=1..n} (y_t - mean)^2
func ACF(y []float64, nLags int) ([]float64, error) {
	n := len(y)
	if nLags < 0 {
		return nil, errors.New("nLags must be >= 0")
	}
	if n == 0 {
		return nil, errors.New("y must be non-empty")
	}
	if nLags >= n {
		return nil, errors.New("nLags must be < len(y)")
	}
	mean := 0.0
	for _, v := range y {
		mean += v
	}
	mean /= float64(n)
	c0 := 0.0
	for _, v := range y {
		d := v - mean
		c0 += d * d
	}
	if c0 == 0 {
		return nil, errors.New("zero-variance series")
	}
	out := make([]float64, nLags+1)
	out[0] = 1
	for k := 1; k <= nLags; k++ {
		s := 0.0
		for t := k; t < n; t++ {
			s += (y[t] - mean) * (y[t-k] - mean)
		}
		out[k] = s / c0
	}
	return out, nil
}

// PACF returns the sample partial autocorrelation function up to nLags via
// the Levinson-Durbin recursion (Yule-Walker estimates of AR coefficients).
//
// Mirrors statsmodels.tsa.stattools.pacf with method="ywunbiased"/"yw" (default).
// Output length = nLags + 1; out[0] = 1.0.
func PACF(y []float64, nLags int) ([]float64, error) {
	r, err := ACF(y, nLags)
	if err != nil {
		return nil, err
	}
	pacf := make([]float64, nLags+1)
	pacf[0] = 1
	if nLags == 0 {
		return pacf, nil
	}
	// Levinson-Durbin recursion.
	phi := make([][]float64, nLags+1)
	for i := range phi {
		phi[i] = make([]float64, nLags+1)
	}
	v := make([]float64, nLags+1)
	v[0] = r[0]
	for k := 1; k <= nLags; k++ {
		s := 0.0
		for j := 1; j < k; j++ {
			s += phi[k-1][j] * r[k-j]
		}
		// Zero-variance guard: if the partial-autocorrelation residual variance
		// has collapsed to ~machine epsilon, the next coefficient is undefined
		// (the Levinson-Durbin recursion divides by it). The threshold is set
		// well above f64 denormal range so we abort cleanly on near-singular
		// covariance instead of producing huge spurious pacf values.
		if math.Abs(v[k-1]) < 1e-12 {
			pacf[k] = 0
			break
		}
		phi[k][k] = (r[k] - s) / v[k-1]
		for j := 1; j < k; j++ {
			phi[k][j] = phi[k-1][j] - phi[k][k]*phi[k-1][k-j]
		}
		v[k] = v[k-1] * (1 - phi[k][k]*phi[k][k])
		pacf[k] = phi[k][k]
	}
	return pacf, nil
}
