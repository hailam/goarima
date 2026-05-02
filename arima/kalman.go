package arima

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// expandSARMA combines (1 - phi B)(1 - Phi B^m) → AR polynomial of total order p + P*m.
// Returns AR coefficients indexed 1..p+P*m (no intercept term).
func expandSARMA(phi, Phi []float64, m int) []float64 {
	p := len(phi)
	P := len(Phi)
	if P == 0 || m <= 1 {
		out := make([]float64, p)
		copy(out, phi)
		return out
	}
	// Build polynomial coefficients including B^0.
	// nonSeasonal poly = 1 - phi_1 B - phi_2 B^2 ... = [1, -phi_1, -phi_2, ...]
	a := make([]float64, p+1)
	a[0] = 1
	for i, v := range phi {
		a[i+1] = -v
	}
	// seasonal poly = 1 - Phi_1 B^m - Phi_2 B^{2m} - ...
	b := make([]float64, P*m+1)
	b[0] = 1
	for i, v := range Phi {
		b[(i+1)*m] = -v
	}
	// multiply
	c := make([]float64, len(a)+len(b)-1)
	for i, ai := range a {
		for j, bj := range b {
			c[i+j] += ai * bj
		}
	}
	// expand back to phi: -c[1:]
	out := make([]float64, len(c)-1)
	for i := 1; i < len(c); i++ {
		out[i-1] = -c[i]
	}
	return out
}

// expandSMA combines (1 + theta B)(1 + Theta B^m).
// Returns MA coefficients indexed 1..q+Q*m.
func expandSMA(theta, Theta []float64, m int) []float64 {
	q := len(theta)
	Q := len(Theta)
	if Q == 0 || m <= 1 {
		out := make([]float64, q)
		copy(out, theta)
		return out
	}
	a := make([]float64, q+1)
	a[0] = 1
	for i, v := range theta {
		a[i+1] = v
	}
	b := make([]float64, Q*m+1)
	b[0] = 1
	for i, v := range Theta {
		b[(i+1)*m] = v
	}
	c := make([]float64, len(a)+len(b)-1)
	for i, ai := range a {
		for j, bj := range b {
			c[i+j] += ai * bj
		}
	}
	out := make([]float64, len(c)-1)
	for i := 1; i < len(c); i++ {
		out[i-1] = c[i]
	}
	return out
}

// armaCSS evaluates the Conditional Sum of Squares for an ARMA(p,q) model
// over a (centered) series y. Returns negLogLik (proportional to log SSE) and
// the estimated sigma^2.
//
// Recursion (with e_t = 0 for t < max(p,q)):
//
//	e_t = y_t - sum_i phi_i y_{t-i} - sum_j theta_j e_{t-j}
//
// Loss:  L = (n/2)*log(SSE/n)  (concentrated profile likelihood, additive constant dropped)
func armaCSS(y, phi, theta []float64) (negLogLik, sigma2 float64, residuals []float64) {
	n := len(y)
	p := len(phi)
	q := len(theta)
	start := p
	if q > start {
		start = q
	}
	if start >= n {
		return math.Inf(1), 0, nil
	}
	res := make([]float64, n)
	sse := 0.0
	count := 0
	for t := start; t < n; t++ {
		pred := 0.0
		for i := 0; i < p; i++ {
			pred += phi[i] * y[t-1-i]
		}
		for j := 0; j < q; j++ {
			pred += theta[j] * res[t-1-j]
		}
		e := y[t] - pred
		res[t] = e
		sse += e * e
		count++
	}
	if count == 0 || sse <= 0 {
		return math.Inf(1), 0, res
	}
	s2 := sse / float64(count)
	// concentrated CSS: (n/2)*log(sigma^2)
	return float64(count) / 2 * math.Log(s2), s2, res
}

// kalmanARMALikelihood computes the exact Gaussian log-likelihood of an
// ARMA(p,q) model on a centered series via Kalman filter on the Hamilton
// state-space form with stationary state initialization.
//
// Returns -log L (positive value to minimize), sigma^2, and innovations.
func kalmanARMALikelihood(y, phi, theta []float64) (negLogLik, sigma2 float64, innovations []float64) {
	n := len(y)
	p := len(phi)
	q := len(theta)
	r := p
	if q+1 > r {
		r = q + 1
	}
	if r == 0 {
		// pure white-noise model
		sse := 0.0
		for _, v := range y {
			sse += v * v
		}
		s2 := sse / float64(n)
		if s2 <= 0 {
			return math.Inf(1), 0, nil
		}
		return float64(n) / 2 * (math.Log(2*math.Pi*s2) + 1), s2, nil
	}

	// Build T (companion form), Z = e_1, R column.
	T := mat.NewDense(r, r, nil)
	for i := 0; i < r; i++ {
		if i < p {
			T.Set(i, 0, phi[i])
		}
		if i+1 < r {
			T.Set(i, i+1, 1)
		}
	}
	R := mat.NewVecDense(r, nil)
	R.SetVec(0, 1)
	for j := 0; j < q; j++ {
		if j+1 < r {
			R.SetVec(j+1, theta[j])
		}
	}
	// RR^T (sigma^2 will scale at the end)
	RRt := mat.NewDense(r, r, nil)
	RRt.Mul(R, R.T())

	// Stationary initial covariance: vec(P0) = (I - T⊗T)^{-1} vec(RR^T)
	P0, ok := stationaryCov(T, RRt, r)
	if !ok {
		// fallback to large diffuse
		P0 = mat.NewDense(r, r, nil)
		for i := 0; i < r; i++ {
			P0.Set(i, i, 1e6)
		}
	}

	a := mat.NewVecDense(r, nil) // initial state mean = 0
	P := mat.DenseCopyOf(P0)

	innov := make([]float64, n)
	logF := 0.0
	sumVF := 0.0

	for t := 0; t < n; t++ {
		// y_t - Z*a
		v := y[t] - a.AtVec(0)
		// F_t = Z*P*Z^T = P[0,0]
		F := P.At(0, 0)
		if F <= 0 || math.IsNaN(F) {
			return math.Inf(1), 0, innov
		}
		// K_t = P*Z^T / F = P[:,0] / F
		K := mat.NewVecDense(r, nil)
		for i := 0; i < r; i++ {
			K.SetVec(i, P.At(i, 0)/F)
		}
		// update a: a + K*v
		for i := 0; i < r; i++ {
			a.SetVec(i, a.AtVec(i)+K.AtVec(i)*v)
		}
		// update P: P - K*Z*P (rank-1 update). Snapshot row 0 first
		// because in-place writes to row 0 must not contaminate later rows.
		row0 := make([]float64, r)
		for j := 0; j < r; j++ {
			row0[j] = P.At(0, j)
		}
		for i := 0; i < r; i++ {
			ki := K.AtVec(i)
			for j := 0; j < r; j++ {
				P.Set(i, j, P.At(i, j)-ki*row0[j])
			}
		}

		innov[t] = v
		logF += math.Log(F)
		sumVF += v * v / F

		// predict: a = T*a, P = T*P*T^T + RR^T
		newA := mat.NewVecDense(r, nil)
		newA.MulVec(T, a)
		a = newA
		var TP mat.Dense
		TP.Mul(T, P)
		var TPTt mat.Dense
		TPTt.Mul(&TP, T.T())
		var newP mat.Dense
		newP.Add(&TPTt, RRt)
		P = mat.DenseCopyOf(&newP)
	}

	// Concentrated MLE: sigma^2 = sumVF / n
	if sumVF <= 0 {
		return math.Inf(1), 0, innov
	}
	s2 := sumVF / float64(n)
	// Profile log-lik: -0.5*(n*log(2*pi*sigma^2) + n + logF)  (sumVF/sigma2 = n)
	negLL := 0.5 * (float64(n)*(math.Log(2*math.Pi*s2)+1) + logF)
	return negLL, s2, innov
}

// stationaryCov solves (I - T⊗T) vec(P) = vec(Q) → P.
// Returns ok=false if singular.
func stationaryCov(T, Q *mat.Dense, r int) (*mat.Dense, bool) {
	n := r * r
	A := mat.NewDense(n, n, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < r; j++ {
			row := i*r + j
			A.Set(row, row, 1)
			for k := 0; k < r; k++ {
				for l := 0; l < r; l++ {
					col := k*r + l
					A.Set(row, col, A.At(row, col)-T.At(i, k)*T.At(j, l))
				}
			}
		}
	}
	q := mat.NewVecDense(n, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < r; j++ {
			q.SetVec(i*r+j, Q.At(i, j))
		}
	}
	var p mat.VecDense
	if err := p.SolveVec(A, q); err != nil {
		return nil, false
	}
	out := mat.NewDense(r, r, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < r; j++ {
			out.Set(i, j, p.AtVec(i*r+j))
		}
	}
	return out, true
}
