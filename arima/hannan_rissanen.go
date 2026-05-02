package arima

import (
	"math"
)

// hannanRissanen returns initial ARMA(p, q) parameter estimates from a
// stationary series via the two-stage regression of Hannan & Rissanen (1982).
//
// Stage 1: fit a long autoregression AR(m) with m >> p+q via OLS, compute
// residuals e_hat.
//
// Stage 2: regress y_t on (y_{t-1}..y_{t-p}, e_hat_{t-1}..e_hat_{t-q}) for
// t > m. The first p OLS coefficients are the AR initial estimate; the next
// q are the MA initial estimate.
//
// Returns ([]nil, []nil) when the series is too short for the chosen p+q.
func hannanRissanen(y []float64, p, q int) (phi, theta []float64) {
	n := len(y)
	if p == 0 && q == 0 {
		return nil, nil
	}
	// Stage-1 AR order. Box & Jenkins suggest a value larger than p+q,
	// up to roughly log(n). We use max(p+q+1, ceil(log(n))) but cap by n/4.
	m := p + q + 1
	if lg := int(math.Ceil(math.Log(float64(n)))); lg > m {
		m = lg
	}
	if m > n/4 {
		m = n / 4
	}
	if m <= 0 || m+1 >= n {
		return nil, nil
	}

	// Stage 1: AR(m) fit on y_t = sum_{k=1..m} a_k * y_{t-k} + e_t.
	rows := n - m
	X := make([][]float64, rows)
	yt := make([]float64, rows)
	for i := 0; i < rows; i++ {
		t := m + i
		yt[i] = y[t]
		row := make([]float64, m)
		for k := 1; k <= m; k++ {
			row[k-1] = y[t-k]
		}
		X[i] = row
	}
	a, err := olsFit(X, yt, false)
	if err != nil {
		return nil, nil
	}
	// Residuals.
	e := make([]float64, n)
	for i := 0; i < rows; i++ {
		s := 0.0
		for k := 0; k < m; k++ {
			s += a[k] * X[i][k]
		}
		e[m+i] = yt[i] - s
	}
	// Earlier residuals are unknown; leave at zero.

	// Stage 2: y_t ~ phi_1 y_{t-1} + ... + theta_1 e_{t-1} + ...
	start := m + max3(p, q, 1)
	if start >= n {
		return nil, nil
	}
	rows2 := n - start
	if rows2 < p+q+1 {
		return nil, nil
	}
	cols := p + q
	if cols == 0 {
		return nil, nil
	}
	X2 := make([][]float64, rows2)
	y2 := make([]float64, rows2)
	for i := 0; i < rows2; i++ {
		t := start + i
		y2[i] = y[t]
		row := make([]float64, cols)
		for k := 1; k <= p; k++ {
			row[k-1] = y[t-k]
		}
		for k := 1; k <= q; k++ {
			row[p+k-1] = e[t-k]
		}
		X2[i] = row
	}
	beta, err := olsFit(X2, y2, false)
	if err != nil {
		return nil, nil
	}
	if p > 0 {
		phi = append([]float64{}, beta[:p]...)
	}
	if q > 0 {
		theta = append([]float64{}, beta[p:p+q]...)
	}
	return phi, theta
}

func max3(a, b, c int) int {
	m := a
	if b > m {
		m = b
	}
	if c > m {
		m = c
	}
	return m
}

// invertARTransform computes the inverse of arTransparams: given a vector of
// AR coefficients (phi_1..phi_p), recover the unconstrained reals whose tanh
// transform yields those phi.
//
// The transformation is computed by inverting the partial-autocorrelation
// recursion (Jones 1980). The recursion goes:
//
//	phi^(p)_p = a_p, where a_p = tanh(param_p)
//	phi^(p-1)_k = (phi^(p)_k + phi^(p)_p * phi^(p)_{p-k}) / (1 - phi^(p)_p^2)
//
// We invert this step-by-step.
func invertARTransform(phi []float64) []float64 {
	p := len(phi)
	if p == 0 {
		return nil
	}
	// Work top-down: at each level k, the last coefficient is the partial
	// autocorrelation a_k; remaining coefficients are unwound.
	cur := append([]float64{}, phi...)
	pacf := make([]float64, p)
	for k := p; k >= 1; k-- {
		ak := cur[k-1]
		// Clamp to (-1, 1) so atanh is finite.
		if ak >= 1 {
			ak = 1 - 1e-9
		}
		if ak <= -1 {
			ak = -1 + 1e-9
		}
		pacf[k-1] = ak
		if k == 1 {
			break
		}
		// Unwind one level: prev[i] = (cur[i] + ak * cur[k-1-i]) / (1 - ak^2)
		// for i = 0..k-2.
		denom := 1 - ak*ak
		if denom == 0 {
			denom = 1e-12
		}
		next := make([]float64, k-1)
		for i := 0; i < k-1; i++ {
			next[i] = (cur[i] + ak*cur[k-2-i]) / denom
		}
		cur = next
	}
	out := make([]float64, p)
	for i, a := range pacf {
		out[i] = math.Atanh(a)
	}
	return out
}

// invertMATransform inverts the MA partial-autocorrelation transformation.
func invertMATransform(theta []float64) []float64 {
	q := len(theta)
	if q == 0 {
		return nil
	}
	cur := append([]float64{}, theta...)
	pacf := make([]float64, q)
	for k := q; k >= 1; k-- {
		ak := cur[k-1]
		if ak >= 1 {
			ak = 1 - 1e-9
		}
		if ak <= -1 {
			ak = -1 + 1e-9
		}
		pacf[k-1] = ak
		if k == 1 {
			break
		}
		denom := 1 - ak*ak
		if denom == 0 {
			denom = 1e-12
		}
		next := make([]float64, k-1)
		for i := 0; i < k-1; i++ {
			// MA recurrence has the opposite sign of AR.
			next[i] = (cur[i] - ak*cur[k-2-i]) / denom
		}
		cur = next
	}
	out := make([]float64, q)
	for i, a := range pacf {
		out[i] = math.Atanh(a)
	}
	return out
}
