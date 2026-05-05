package arima

import "math"

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

// PublicKalmanLL exposes kalmanARMALikelihood for cross-implementation parity
// checks. Returns (negLogLik, sigma2_concentrated, innovations).
func PublicKalmanLL(y, phi, theta []float64) (float64, float64, []float64) {
	return kalmanARMALikelihood(y, phi, theta)
}

// kalmanARMALikelihoodInto is the buffer-reuse variant of
// kalmanARMALikelihood used by Fit's hot path. Identical algorithm; the
// 9 r-sized / r²-sized / 2r-sized scratch slices are taken from `ws`
// instead of being allocated per call.
//
// Returns negLogLik and sigma². The innovations are NOT returned —
// Fit's compute closure discards them, and skipping their allocation is
// half the point of this entry point. Callers that need innovations
// (Summary's Hessian probe, integration tests) keep using
// kalmanARMALikelihood.
func kalmanARMALikelihoodInto(y, phi, theta []float64, ws *kalmanWorkspace) (negLogLik, sigma2 float64) {
	n := len(y)
	p := len(phi)
	q := len(theta)
	r := p
	if q+1 > r {
		r = q + 1
	}
	if r == 0 {
		sse := 0.0
		for _, v := range y {
			sse += v * v
		}
		s2 := sse / float64(n)
		if s2 <= 0 {
			return math.Inf(1), 0
		}
		return float64(n) / 2 * (math.Log(2*math.Pi*s2) + 1), s2
	}

	// nzT — capacity 2r, length grown to actual nz count below.
	if cap(ws.nzT) < 2*r {
		ws.nzT = make([]tNZ, 0, 2*r)
	} else {
		ws.nzT = ws.nzT[:0]
	}
	for i := 0; i < r; i++ {
		if i < p && phi[i] != 0 {
			ws.nzT = append(ws.nzT, tNZ{i, 0, phi[i]})
		}
		if i+1 < r {
			ws.nzT = append(ws.nzT, tNZ{i, i + 1, 1})
		}
	}
	nzT := ws.nzT

	// Rvec — zeroed because we only set the [0..q] entries explicitly.
	ws.Rvec = ensureLenZ(ws.Rvec, r)
	Rvec := ws.Rvec
	Rvec[0] = 1
	for j := 0; j < q; j++ {
		if j+1 < r {
			Rvec[j+1] = theta[j]
		}
	}

	// RRt — zeroed because the inner loop only writes when ri != 0.
	ws.RRt = ensureLenZ(ws.RRt, r*r)
	RRt := ws.RRt
	for i := 0; i < r; i++ {
		ri := Rvec[i]
		if ri == 0 {
			continue
		}
		off := i * r
		for j := 0; j < r; j++ {
			RRt[off+j] = ri * Rvec[j]
		}
	}

	P, _ := stationaryCovGardner(phi, theta)

	// a — must be zeroed (state mean starts at 0).
	ws.a = ensureLenZ(ws.a, r)
	a := ws.a
	ws.K = ensureLen(ws.K, r)
	K := ws.K
	ws.row0 = ensureLen(ws.row0, r)
	row0 := ws.row0
	ws.newA = ensureLen(ws.newA, r)
	newA := ws.newA
	ws.TP = ensureLen(ws.TP, r*r)
	TP := ws.TP
	ws.newP = ensureLen(ws.newP, r*r)
	newP := ws.newP

	logF := 0.0
	sumVF := 0.0

	for t := 0; t < n; t++ {
		v := y[t] - a[0]
		F := P[0]
		if F <= 0 || math.IsNaN(F) {
			return math.Inf(1), 0
		}
		invF := 1.0 / F
		for i := 0; i < r; i++ {
			K[i] = P[i*r] * invF
			a[i] += K[i] * v
		}
		copy(row0, P[:r])
		for i := 0; i < r; i++ {
			ki := K[i]
			r0i := row0[i]
			off := i * r
			for j := 0; j < r; j++ {
				kj := K[j]
				P[off+j] += -ki*row0[j] - kj*r0i + ki*kj*F
			}
		}
		logF += math.Log(F)
		sumVF += v * v * invF

		for i := 0; i < r; i++ {
			newA[i] = 0
		}
		for _, e := range nzT {
			newA[e.i] += e.v * a[e.j]
		}
		copy(a, newA)

		for k := range TP {
			TP[k] = 0
		}
		for _, e := range nzT {
			ti := e.i * r
			tj := e.j * r
			tv := e.v
			for j := 0; j < r; j++ {
				TP[ti+j] += tv * P[tj+j]
			}
		}
		copy(newP, RRt)
		for i := 0; i < r; i++ {
			row := i * r
			for _, e := range nzT {
				newP[row+e.i] += TP[row+e.j] * e.v
			}
		}
		P, newP = newP, P
	}

	if sumVF <= 0 {
		return math.Inf(1), 0
	}
	s2 := sumVF / float64(n)
	negLL := 0.5 * (float64(n)*(math.Log(2*math.Pi*s2)+1) + logF)
	return negLL, s2
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
// Hot loop uses plain float64 slices (no mat.Dense overhead) — typical r
// is 1..5 so explicit indexing is faster than BLAS dispatch.
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

	// T (companion form) as sparse triples: T[i,0] = phi[i] for i<p,
	// T[i,i+1] = 1 for i+1<r. At most p + (r-1) ≤ 2r-1 nonzero entries.
	nzT := make([]tNZ, 0, 2*r)
	for i := 0; i < r; i++ {
		if i < p && phi[i] != 0 {
			nzT = append(nzT, tNZ{i, 0, phi[i]})
		}
		if i+1 < r {
			nzT = append(nzT, tNZ{i, i + 1, 1})
		}
	}
	// R selection: (1, theta_1, ..., theta_{r-1}, 0...). Build RR' once.
	Rvec := make([]float64, r)
	Rvec[0] = 1
	for j := 0; j < q; j++ {
		if j+1 < r {
			Rvec[j+1] = theta[j]
		}
	}
	RRt := make([]float64, r*r)
	for i := 0; i < r; i++ {
		ri := Rvec[i]
		if ri == 0 {
			continue
		}
		off := i * r
		for j := 0; j < r; j++ {
			RRt[off+j] = ri * Rvec[j]
		}
	}

	// Initial stationary covariance via Gardner-Harvey-Phillips O(r³) algorithm.
	// (Replaces the previous O(r⁶) Sylvester-style dense solve. For r=14 this
	// is ~2700× fewer flops per kalman call.) Output is already flat row-major
	// of the correct size — alias it directly as P.
	P, _ := stationaryCovGardner(phi, theta)

	// Pre-allocated working buffers (reused every step).
	a := make([]float64, r)
	K := make([]float64, r)
	row0 := make([]float64, r)
	newA := make([]float64, r)
	TP := make([]float64, r*r)
	newP := make([]float64, r*r)

	innov := make([]float64, n)
	logF := 0.0
	sumVF := 0.0

	for t := 0; t < n; t++ {
		v := y[t] - a[0]
		F := P[0]
		if F <= 0 || math.IsNaN(F) {
			return math.Inf(1), 0, innov
		}
		invF := 1.0 / F
		for i := 0; i < r; i++ {
			K[i] = P[i*r] * invF
			a[i] += K[i] * v
		}
		// Joseph-form covariance update: P = (I - K·H)·P·(I - K·H)' + K·R·K'.
		// With ARMA companion form H = [1,0,…,0] and observation noise R = 0:
		//   P_new[i,j] = P[i,j] - K[i]·row0[j] - K[j]·row0[i] + K[i]·K[j]·F
		// (row0 is a snapshot of P's first row taken BEFORE the update.)
		// Symmetric by construction; preserves PSD against rounding error.
		// Replaces the older rank-1 form `P -= K·row0` which is mathematically
		// equivalent only in exact arithmetic — numerically it loses
		// symmetry/PSD over many steps once BFGS pushes φ near the unit
		// circle, causing F = P[0,0] to drift wildly. See KAL-1.
		copy(row0, P[:r])
		for i := 0; i < r; i++ {
			ki := K[i]
			r0i := row0[i]
			off := i * r
			for j := 0; j < r; j++ {
				kj := K[j]
				P[off+j] += -ki*row0[j] - kj*r0i + ki*kj*F
			}
		}

		innov[t] = v
		logF += math.Log(F)
		sumVF += v * v * invF

		// Predict via sparse T:
		//   a' = T*a
		//   P' = T*P*T' + RR'
		for i := 0; i < r; i++ {
			newA[i] = 0
		}
		for _, e := range nzT {
			newA[e.i] += e.v * a[e.j]
		}
		copy(a, newA)

		// TP = T @ P (sparse-T, dense-P).
		for k := range TP {
			TP[k] = 0
		}
		for _, e := range nzT {
			ti := e.i * r
			tj := e.j * r
			tv := e.v
			for j := 0; j < r; j++ {
				TP[ti+j] += tv * P[tj+j]
			}
		}
		// P' = TP @ T' + RR'.
		copy(newP, RRt)
		for i := 0; i < r; i++ {
			row := i * r
			for _, e := range nzT {
				newP[row+e.i] += TP[row+e.j] * e.v
			}
		}
		P, newP = newP, P
	}

	if sumVF <= 0 {
		return math.Inf(1), 0, innov
	}
	s2 := sumVF / float64(n)
	negLL := 0.5 * (float64(n)*(math.Log(2*math.Pi*s2)+1) + logF)
	return negLL, s2, innov
}

