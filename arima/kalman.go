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

	// Pooled Gardner workspace from the kalmanWorkspace.
	P, _ := stationaryCovGardnerInto(&ws.gardner, phi, theta)

	// CKMS-1 (2026-06-12): Chandrasekhar–Morf–Kailath fast filter.
	// For a time-invariant univariate state-space started at the exact
	// stationary covariance, ΔP_t = P_{t+1} − P_t is rank-1 and stays
	// rank-1 under the Riccati recursion, so the filter can propagate
	// one r-vector (W) plus scalars (F, β) instead of the r×r matrix:
	// O(r) per step instead of O(r²). Recursions (Z = e₁, M ≔ K·F):
	//   w  = W[0]
	//   F⁺ = F − β·w²
	//   M⁺ = M − β·w·W
	//   W⁺ = T·(W − (w/F)·M)
	//   β⁺ = β·F/F⁺
	// derived by closing ΔP⁺ = −β⁺W⁺W⁺ᵀ exactly (validated to 1e-12
	// against the Riccati loop over 60 random models incl. near-unit
	// roots; ~12× per-eval at r=26, n=652). On any numerical-guard
	// trip (F⁺ ≤ 0/NaN — possible for extreme near-non-invertible MA)
	// we fall back to the Joseph-form Riccati loop below, which keeps
	// its KAL-1 stability properties and the SS-KALMAN-1 freeze.
	if negLL, s2, ok := ckmsARMAInto(y, phi, theta, P, r, ws); ok {
		return negLL, s2
	}

	// a — must be zeroed (state mean starts at 0).
	ws.a = ensureLenZ(ws.a, r)
	a := ws.a
	ws.K = ensureLen(ws.K, r)
	K := ws.K
	ws.row0 = ensureLen(ws.row0, r)
	row0 := ws.row0
	ws.newA = ensureLen(ws.newA, r)
	newA := ws.newA
	ws.newP = ensureLen(ws.newP, r*r)
	newP := ws.newP

	logF := 0.0
	sumVF := 0.0

	// PG-113: P is symmetric, so we only maintain its upper triangle in the
	// hot loop. The lower triangle of {P, newP} is not read; first-step
	// reads come from Gardner's full output (which is symmetric anyway).
	// Reads of "first column" P[i, 0] are remapped to P[0, i] by symmetry.

	// SS-KALMAN-1 (2026-06-12): steady-state freeze. For a stationary,
	// invertible ARMA (transparams guarantees both), the predicted
	// covariance P converges to the fixed point of the Riccati
	// recursion; from then on F = P[0,0] and K are constants and the
	// O(r²) Joseph-update + P-predict work is pure waste — only the
	// O(r) state-mean recursion still depends on t. We detect
	// convergence with a bit-tight relative test (max |ΔP| over the
	// upper triangle ≤ 1e-14·(1+max|P|) on two consecutive steps —
	// statsmodels uses the same construct with tolerance=1e-19) and
	// then freeze K, log F, and 1/F. Likelihood impact is bounded by
	// the tolerance (≪ 1e-9 cumulative over n in the worst case);
	// near-unit-root candidates simply never satisfy the test and run
	// the full recursion as before. Mirrors statsmodels'
	// KalmanFilter.converged shortcut (Harvey 1989 §3.3.3).
	// Freeze requires EXACT bitwise stability of the predicted
	// covariance across consecutive steps (ssRelTol = 0): at a bitwise
	// fixed point the unfrozen recursion would reproduce P, K, F, and
	// the per-step logF/sumVF contributions exactly, so the frozen
	// shortcut is bit-identical to the full loop (the Into-vs-legacy
	// equivalence test stays bit-exact). A 1e-14 relative tolerance
	// was tried first: it froze earlier but leaked ~1e-13 likelihood
	// drift, which both broke bit-exactness and wiggled BFGS
	// trajectories (±5 AICc on m5-family picks). Contracting Riccati
	// maps in float64 typically reach an exact fixed point; if one
	// settles into a 1-2 ulp limit cycle instead, the freeze simply
	// never fires and the full recursion runs as before.
	const ssRelTol = 0.0
	frozen := false
	ssArmed := false // prev full upper-triangle snapshot is valid
	ssStable := 0
	var ssInvF, ssLogF float64
	ssPrevTop := make([]float64, r)  // previous PREDICTED top row
	ssPrevP := []float64(nil)        // previous PREDICTED upper triangle (lazily allocated)
	copy(ssPrevTop, P[:r])

	for t := 0; t < n; t++ {
		if frozen {
			v := y[t] - a[0]
			for i := 0; i < r; i++ {
				a[i] += K[i] * v
			}
			// Predict step for the state mean only (companion shift +
			// phi broadcast) — same algebra as the full path below.
			a0 := a[0]
			for i := 0; i+1 < r; i++ {
				newA[i] = a[i+1]
			}
			newA[r-1] = 0
			for i := 0; i < p; i++ {
				if pi := phi[i]; pi != 0 {
					newA[i] += pi * a0
				}
			}
			copy(a, newA)
			logF += ssLogF
			sumVF += v * v * ssInvF
			continue
		}
		v := y[t] - a[0]
		F := P[0]
		if F <= 0 || math.IsNaN(F) {
			return math.Inf(1), 0
		}
		invF := 1.0 / F
		// K[i] = P[i,0]/F = P[0,i]/F by symmetry — read upper.
		for i := 0; i < r; i++ {
			K[i] = P[i] * invF
			a[i] += K[i] * v
		}
		// Joseph P-update (upper triangle only).
		// P[i,j] += -K[i]·row0[j] + (K[i]·F - row0[i])·K[j]
		// Hoisted coef + 2-AXPY shape; restricted to j ≥ i since P is
		// symmetric. The Joseph form preserves PSD against rounding error
		// (KAL-1) — cannot be replaced by the rank-1 `P -= K·row0` form.
		copy(row0, P[:r])
		for i := 0; i < r; i++ {
			ki := K[i]
			coef := ki*F - row0[i]
			off := i * r
			for j := i; j < r; j++ {
				P[off+j] += coef*K[j] - ki*row0[j]
			}
		}
		logF += math.Log(F)
		sumVF += v * v * invF

		// Predict step (PG-113). T is the ARMA companion matrix with
		// T[i, 0] = phi[i] (i<p) and T[i, i+1] = 1 (i+1<r).
		//
		// Fused TP+newP: newP[i,k] = (T·P·T')[i,k] + RRt[i,k] decomposes as
		//   • diagonal-shift bulk:    P[i+1, k+1]               (i+1<r, k+1<r)
		//   • phi-row broadcast:      phi[i] · P[0, k+1]        (i<p, k+1<r)
		//   • phi-col broadcast:      phi[k] · P[0, i+1]        (k<p, i+1<r)  [via symmetry]
		//   • phi-phi corner:         phi[i] · phi[k] · P[0,0]  (i<p, k<p)
		// We compute only the upper triangle (j ≥ i).

		a0 := a[0]
		for i := 0; i+1 < r; i++ {
			newA[i] = a[i+1]
		}
		newA[r-1] = 0
		for i := 0; i < p; i++ {
			if pi := phi[i]; pi != 0 {
				newA[i] += pi * a0
			}
		}
		copy(a, newA)

		// Diagonal-shift bulk + RRt (upper only).
		for i := 0; i+1 < r; i++ {
			rowOut := i * r
			rowIn := (i + 1) * r
			for k := i; k+1 < r; k++ {
				newP[rowOut+k] = P[rowIn+k+1] + RRt[rowOut+k]
			}
			newP[rowOut+r-1] = RRt[rowOut+r-1]
		}
		newP[(r-1)*r+r-1] = RRt[(r-1)*r+r-1]

		// phi-row: newP[i, k] += phi[i] · P[0, k+1] for i<p, i ≤ k, k+1<r.
		for i := 0; i < p; i++ {
			pi := phi[i]
			if pi == 0 {
				continue
			}
			rowOut := i * r
			for k := i; k+1 < r; k++ {
				newP[rowOut+k] += pi * P[k+1]
			}
		}
		// phi-col + phi-corner contribute only to the small (0..p-1, 0..p-1)
		// upper-triangular block. P[i+1, 0] = P[0, i+1] by symmetry.
		P0 := P[0]
		for k := 0; k < p; k++ {
			pk := phi[k]
			if pk == 0 {
				continue
			}
			for i := 0; i <= k; i++ {
				if i+1 < r {
					newP[i*r+k] += pk * P[i+1]
				}
				if pi := phi[i]; pi != 0 {
					newP[i*r+k] += pi * pk * P0
				}
			}
		}
		P, newP = newP, P

		// SS-KALMAN-1 convergence test, two-tier so the never-freezing
		// case (near-unit-root candidates — common among auto.arima
		// survivors) pays ~nothing. Compares the PREDICTED covariance
		// entering step t+1 against the predicted covariance that
		// entered step t (NOT against `newP`, which after the in-place
		// Joseph update + swap holds this step's FILTERED matrix — a
		// different fixed point; a v1 that compared against it never
		// fired). Tier 1: O(r) top-row gate vs a retained copy. Tier 2:
		// once the gate passes, snapshot the full upper triangle and
		// require two consecutive full-matrix matches.
		gate := true
		for i := 0; i < r; i++ {
			d := P[i] - ssPrevTop[i]
			if d < 0 {
				d = -d
			}
			av := P[i]
			if av < 0 {
				av = -av
			}
			if d > ssRelTol*(1+av) {
				gate = false
				break
			}
		}
		copy(ssPrevTop, P[:r])
		if !gate {
			ssArmed = false
			ssStable = 0
			continue
		}
		if ssPrevP == nil {
			ssPrevP = make([]float64, r*r)
		}
		if !ssArmed {
			for i := 0; i < r; i++ {
				off := i * r
				copy(ssPrevP[off+i:off+r], P[off+i:off+r])
			}
			ssArmed = true
			continue
		}
		full := true
		for i := 0; i < r && full; i++ {
			off := i * r
			for j := i; j < r; j++ {
				d := P[off+j] - ssPrevP[off+j]
				if d < 0 {
					d = -d
				}
				av := P[off+j]
				if av < 0 {
					av = -av
				}
				if d > ssRelTol*(1+av) {
					full = false
					break
				}
			}
		}
		for i := 0; i < r; i++ {
			off := i * r
			copy(ssPrevP[off+i:off+r], P[off+i:off+r])
		}
		if !full {
			ssStable = 0
			continue
		}
		ssStable++
		if ssStable >= 2 {
			F := P[0]
			if F > 0 && !math.IsNaN(F) {
				ssInvF = 1.0 / F
				ssLogF = math.Log(F)
				for i := 0; i < r; i++ {
					K[i] = P[i] * ssInvF
				}
				frozen = true
			}
		}
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
// ckmsARMAInto runs the CKMS fast filter (see CKMS-1 comment at the
// call site). P0 is the exact stationary covariance (only row 0 /
// column 0 and the diagonal head are read). Returns ok=false when the
// numerical guard trips; the caller falls back to the Riccati loop.
func ckmsARMAInto(y, phi, theta, P0 []float64, r int, ws *kalmanWorkspace) (negLogLik, sigma2 float64, ok bool) {
	n := len(y)
	p := len(phi)

	F := P0[0]
	if F <= 0 || math.IsNaN(F) {
		return 0, 0, false
	}

	ws.a = ensureLenZ(ws.a, r)
	a := ws.a
	ws.K = ensureLen(ws.K, r)
	M := ws.K // M = K·F = P e₁ (column 0)
	ws.row0 = ensureLen(ws.row0, r)
	W := ws.row0
	ws.newA = ensureLen(ws.newA, r)
	tmp := ws.newA

	// Column 0 of P0: PG-113 keeps only the upper triangle current in
	// the hot loop, but P0 here is Gardner's output, which is full and
	// symmetric — read row 0 (== column 0).
	copy(M, P0[:r])
	// W₀ = T·K₀ = T·(M/F); β₀ = F.
	invF0 := 1.0 / F
	for i := 0; i < r; i++ {
		tmp[i] = M[i] * invF0
	}
	applyCompanionT(W, tmp, phi, p, r)
	beta := F

	logF := 0.0
	sumVF := 0.0
	// SS freeze inside CKMS: once a covariance-factor update is a
	// bitwise no-op (F⁺ == F and every M[i] unchanged) on two
	// consecutive steps, it stays a no-op forever — β is frozen the
	// moment F is (β⁺ = β·F/F⁺), and the closed loop contracts W, so
	// the skipped updates could never become visible again. The frozen
	// step then only advances the state mean, identical to the
	// SS-KALMAN-1 frozen Riccati step, with cached log F and 1/F.
	frozen := false
	ssStable := 0
	var fLogF, fInvF float64
	for t := 0; t < n; t++ {
		if frozen {
			v := y[t] - a[0]
			c := v * fInvF
			for i := 0; i < r; i++ {
				tmp[i] = a[i] + M[i]*c
			}
			applyCompanionT(a, tmp, phi, p, r)
			logF += fLogF
			sumVF += v * v * fInvF
			continue
		}
		if F <= 0 || math.IsNaN(F) {
			return 0, 0, false
		}
		invF := 1.0 / F
		v := y[t] - a[0]
		logF += math.Log(F)
		sumVF += v * v * invF

		// a⁺ = T·(a + K·v), K = M/F.
		c := v * invF
		for i := 0; i < r; i++ {
			tmp[i] = a[i] + M[i]*c
		}
		applyCompanionT(a, tmp, phi, p, r)

		// Covariance-factor updates.
		w := W[0]
		Fn := F - beta*w*w
		if Fn <= 0 || math.IsNaN(Fn) {
			return 0, 0, false
		}
		bw := beta * w
		wf := w * invF
		noop := Fn == F
		for i := 0; i < r; i++ {
			mi := M[i]
			mn := mi - bw*W[i]
			if mn != mi {
				noop = false
			}
			M[i] = mn
			tmp[i] = W[i] - wf*mi
		}
		applyCompanionT(W, tmp, phi, p, r)
		beta = beta * F / Fn
		F = Fn
		if noop {
			ssStable++
			if ssStable >= 2 {
				fInvF = 1.0 / F
				fLogF = math.Log(F)
				frozen = true
			}
		} else {
			ssStable = 0
		}
	}
	if sumVF <= 0 {
		return 0, 0, false
	}
	s2 := sumVF / float64(n)
	return 0.5 * (float64(n)*(math.Log(2*math.Pi*s2)+1) + logF), s2, true
}

// applyCompanionT computes dst = T·x for the ARMA companion matrix
// (T[i,0] = phi[i] for i<p; T[i,i+1] = 1). dst and x must not alias.
func applyCompanionT(dst, x []float64, phi []float64, p, r int) {
	x0 := x[0]
	for i := 0; i+1 < r; i++ {
		dst[i] = x[i+1]
	}
	dst[r-1] = 0
	for i := 0; i < p; i++ {
		if pi := phi[i]; pi != 0 {
			dst[i] += pi * x0
		}
	}
}

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
	newP := make([]float64, r*r)

	innov := make([]float64, n)
	logF := 0.0
	sumVF := 0.0

	// PG-113: upper-triangle-only — see kalmanARMALikelihoodInto for the
	// detailed case decomposition.

	for t := 0; t < n; t++ {
		v := y[t] - a[0]
		F := P[0]
		if F <= 0 || math.IsNaN(F) {
			return math.Inf(1), 0, innov
		}
		invF := 1.0 / F
		for i := 0; i < r; i++ {
			K[i] = P[i] * invF
			a[i] += K[i] * v
		}
		copy(row0, P[:r])
		for i := 0; i < r; i++ {
			ki := K[i]
			coef := ki*F - row0[i]
			off := i * r
			for j := i; j < r; j++ {
				P[off+j] += coef*K[j] - ki*row0[j]
			}
		}

		innov[t] = v
		logF += math.Log(F)
		sumVF += v * v * invF

		a0 := a[0]
		for i := 0; i+1 < r; i++ {
			newA[i] = a[i+1]
		}
		newA[r-1] = 0
		for i := 0; i < p; i++ {
			if pi := phi[i]; pi != 0 {
				newA[i] += pi * a0
			}
		}
		copy(a, newA)

		for i := 0; i+1 < r; i++ {
			rowOut := i * r
			rowIn := (i + 1) * r
			for k := i; k+1 < r; k++ {
				newP[rowOut+k] = P[rowIn+k+1] + RRt[rowOut+k]
			}
			newP[rowOut+r-1] = RRt[rowOut+r-1]
		}
		newP[(r-1)*r+r-1] = RRt[(r-1)*r+r-1]

		for i := 0; i < p; i++ {
			pi := phi[i]
			if pi == 0 {
				continue
			}
			rowOut := i * r
			for k := i; k+1 < r; k++ {
				newP[rowOut+k] += pi * P[k+1]
			}
		}
		P0 := P[0]
		for k := 0; k < p; k++ {
			pk := phi[k]
			if pk == 0 {
				continue
			}
			for i := 0; i <= k; i++ {
				if i+1 < r {
					newP[i*r+k] += pk * P[i+1]
				}
				if pi := phi[i]; pi != 0 {
					newP[i*r+k] += pi * pk * P0
				}
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

