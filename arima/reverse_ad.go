package arima

import "math"

// REVERSE-AD-2: reverse-mode AD for the ARMA Kalman likelihood
// gradient. Replaces 2k forward Kalman calls (central-difference
// per-parameter) with one forward + one backward pass, giving
// k-independent per-gradient cost.
//
// Prototype (REVERSE-AD-1, /tmp/rev_ad/main.go) measured 1.84-6.93×
// speedup vs parallel 16-worker central-difference at typical ARIMA
// shapes (Apple M4, k ∈ {1..15}, r ∈ {1..15}). Speedup grows with k
// and r — at production-scale m4-fast shapes (r=51, k≈7) the win is
// 5-8× per gradient call.
//
// Implementation choices:
//   - Rank-1 P-update (not Joseph). Algebraically equivalent in exact
//     arithmetic; production fits use Joseph for stability against
//     unit-circle saturation but the AD-path objective stays
//     internally consistent (Func and Grad both use rank-1).
//   - Forward-pass checkpoints (a_t, P_t, K_t, v_t, F_t) for all t
//     are stored on the workspace `adTape`. For (n=700, r=51) the
//     tape is ~14 MB; pooled across BFGS iterations within one Fit.
//   - P_0 is taken from stationaryCovGardnerInto and treated as a
//     CONSTANT for AD purposes. Differentiating through Gardner is
//     skipped in v1 — small approximation; if it causes basin shifts
//     vs numerical-gradient fits, address in a follow-up via implicit
//     function theorem (Lyapunov derivative).
//   - The cotangent dP would naturally be symmetric (P is symmetric),
//     but the row-0 backprop step writes only to (0, j) entries.
//     We use the explicit `(dP + dP') · X` form everywhere this
//     matters — avoids a factor-of-2 bug found during prototyping.
//
// Returns ok=false on numerical failure (F ≤ 0 or NaN). Caller
// should fall back to the central-difference path in that case.

// kalmanARMALikelihoodGradAD computes negLogLik, sigma², and the
// gradient (∂negLL/∂phi, ∂negLL/∂theta) via reverse-mode AD.
//
// Mirrors kalmanARMALikelihoodInto's contract (same inputs, same
// negLL/sigma² semantics) but additionally returns the gradients.
//
// `p0Override` lets callers supply a fixed P_0 (skip Gardner) — used
// by tests to isolate the Kalman-recursion AD from Gardner's
// contribution. Pass nil to use Gardner output (production path).
//
// gradPhi has length len(phi); gradTheta has length len(theta).
//
// IMPORTANT: P_0 is treated as CONSTANT for AD purposes. The gradient
// returned ignores ∂P_0/∂(phi, theta) — i.e., how the initial
// stationary covariance depends on the AR/MA coefficients via
// Gardner. For production BFGS, this is a small-but-nonzero
// approximation. Differentiating through Gardner properly would
// require the implicit-function theorem on the Lyapunov equation;
// deferred to a future enhancement.
func kalmanARMALikelihoodGradAD(y, phi, theta, p0Override []float64, ws *kalmanWorkspace) (negLogLik, sigma2 float64, gradPhi, gradTheta []float64, ok bool) {
	n := len(y)
	p := len(phi)
	q := len(theta)
	r := p
	if q+1 > r {
		r = q + 1
	}
	if r == 0 {
		// Trivial — likelihood is the variance of y. Gradient is zero.
		sse := 0.0
		for _, v := range y {
			sse += v * v
		}
		s2 := sse / float64(n)
		if s2 <= 0 {
			return math.Inf(1), 0, nil, nil, false
		}
		return float64(n) / 2 * (math.Log(2*math.Pi*s2) + 1), s2, nil, nil, true
	}

	// Build T (companion form) and R (selection vector).
	tTape := ensureLenZ(make([]float64, r*r), r*r)
	for i := 0; i < r; i++ {
		if i < p {
			tTape[i*r] = phi[i]
		}
		if i+1 < r {
			tTape[i*r+i+1] = 1
		}
	}
	R := make([]float64, r)
	R[0] = 1
	for j := 0; j < q; j++ {
		if j+1 < r {
			R[j+1] = theta[j]
		}
	}

	// Initial P: prefer caller-supplied (used by tests to hold P_0 fixed
	// across perturbations), otherwise compute from Gardner. Either way
	// it is treated as a CONSTANT for AD purposes.
	var P0 []float64
	if p0Override != nil {
		P0 = p0Override
	} else {
		P0, _ = stationaryCovGardnerInto(&ws.gardner, phi, theta)
	}

	// Allocate / size the tape.
	tape := &ws.adTape
	rr := r * r
	tape.aHist = ensureLen(tape.aHist, (n+1)*r)
	tape.pHist = ensureLen(tape.pHist, (n+1)*rr)
	tape.kHist = ensureLen(tape.kHist, n*r)
	tape.vHist = ensureLen(tape.vHist, n)
	tape.fHist = ensureLen(tape.fHist, n)

	// Working state (current a, P).
	a := ensureLenZ(make([]float64, r), r)
	P := make([]float64, rr)
	copy(P, P0)

	// Save initial checkpoint.
	for i := 0; i < r; i++ {
		tape.aHist[i] = 0
	}
	copy(tape.pHist[:rr], P)

	logF := 0.0
	sumVF := 0.0

	// === FORWARD PASS with checkpointing ===
	for t := 0; t < n; t++ {
		v := y[t] - a[0]
		F := P[0]
		if F <= 0 || math.IsNaN(F) {
			return math.Inf(1), 0, nil, nil, false
		}
		invF := 1.0 / F
		// K[i] = P[i, 0] / F (column 0).
		K := tape.kHist[t*r : (t+1)*r]
		for i := 0; i < r; i++ {
			K[i] = P[i*r] * invF
		}
		tape.vHist[t] = v
		tape.fHist[t] = F

		// a' = a + K*v
		ap := make([]float64, r)
		for i := 0; i < r; i++ {
			ap[i] = a[i] + K[i]*v
		}
		// P' = P - K * row0  (rank-1)
		Pp := make([]float64, rr)
		for i := 0; i < r; i++ {
			row := i * r
			ki := K[i]
			for j := 0; j < r; j++ {
				Pp[row+j] = P[row+j] - ki*P[j] // P[j] = row0[j]
			}
		}
		// a_{t+1} = T · a'
		aNew := make([]float64, r)
		for i := 0; i < r; i++ {
			s := 0.0
			row := i * r
			for j := 0; j < r; j++ {
				s += tTape[row+j] * ap[j]
			}
			aNew[i] = s
		}
		// P_{t+1} = T · Pp · T' + R R'
		// First TP = T · Pp
		TP := make([]float64, rr)
		for i := 0; i < r; i++ {
			rowOut := i * r
			for j := 0; j < r; j++ {
				s := 0.0
				for k := 0; k < r; k++ {
					s += tTape[rowOut+k] * Pp[k*r+j]
				}
				TP[rowOut+j] = s
			}
		}
		// PNew = TP · T' + R R'
		PNew := make([]float64, rr)
		for i := 0; i < r; i++ {
			rowOut := i * r
			for j := 0; j < r; j++ {
				s := 0.0
				for k := 0; k < r; k++ {
					s += TP[rowOut+k] * tTape[j*r+k] // T'[k,j] = T[j,k]
				}
				PNew[rowOut+j] = s + R[i]*R[j]
			}
		}

		// Save next checkpoint.
		copy(tape.aHist[(t+1)*r:(t+2)*r], aNew)
		copy(tape.pHist[(t+1)*rr:(t+2)*rr], PNew)

		copy(a, aNew)
		copy(P, PNew)

		logF += math.Log(F)
		sumVF += v * v * invF
	}

	if sumVF <= 0 {
		return math.Inf(1), 0, nil, nil, false
	}
	s2 := sumVF / float64(n)
	negLogLik = 0.5 * (float64(n)*(math.Log(2*math.Pi*s2)+1) + logF)

	// === BACKWARD PASS ===
	// Initialise output cotangents.
	tape.dT = ensureLenZ(tape.dT, rr)
	tape.dR = ensureLenZ(tape.dR, r)
	tape.dA = ensureLenZ(tape.dA, r)
	tape.dP = ensureLenZ(tape.dP, rr)
	tape.dAin = ensureLen(tape.dAin, r)
	tape.dPin = ensureLen(tape.dPin, rr)
	tape.dap = ensureLen(tape.dap, r)
	tape.dK = ensureLen(tape.dK, r)
	tape.dPp = ensureLen(tape.dPp, rr)
	tape.TtdP = ensureLen(tape.TtdP, rr)
	tape.TPp = ensureLen(tape.TPp, rr)
	tape.apScratch = ensureLen(tape.apScratch, r)
	tape.PpScratch = ensureLen(tape.PpScratch, rr)

	dT := tape.dT
	dR := tape.dR
	dA := tape.dA
	dP := tape.dP

	invS2 := 1.0 / s2

	for t := n - 1; t >= 0; t-- {
		v := tape.vHist[t]
		F := tape.fHist[t]
		invF := 1.0 / F
		K := tape.kHist[t*r : (t+1)*r]
		Pt := tape.pHist[t*rr : (t+1)*rr]
		at := tape.aHist[t*r : (t+1)*r]

		// Seed cotangents from the likelihood at this t.
		// dL/dF_t = (1/2)/F - (1/(2 s²)) v²/F²
		// dL/dv_t = v / (s² F)
		dF := 0.5*invF - 0.5*invS2*v*v*invF*invF
		dV := invS2 * v * invF

		// Reconstruct ap_t and Pp_t for backward use.
		ap := tape.apScratch
		for i := 0; i < r; i++ {
			ap[i] = at[i] + K[i]*v
		}
		Pp := tape.PpScratch
		for i := 0; i < r; i++ {
			row := i * r
			ki := K[i]
			for j := 0; j < r; j++ {
				Pp[row+j] = Pt[row+j] - ki*Pt[j]
			}
		}

		// --- Backprop a_{t+1} = T · ap ---
		// dap = T' · dA;  dT += dA ⊗ ap
		dap := tape.dap
		for i := 0; i < r; i++ {
			dap[i] = 0
		}
		for i := 0; i < r; i++ {
			for j := 0; j < r; j++ {
				dap[j] += tTape[i*r+j] * dA[i]
				dT[i*r+j] += dA[i] * ap[j]
			}
		}

		// --- Backprop P_{t+1} = T · Pp · T' + R R' ---
		// dPp = T' · dP · T  (sandwich)
		TtdP := tape.TtdP
		for a := 0; a < r; a++ {
			for j := 0; j < r; j++ {
				s := 0.0
				for i := 0; i < r; i++ {
					s += tTape[i*r+a] * dP[i*r+j]
				}
				TtdP[a*r+j] = s
			}
		}
		dPp := tape.dPp
		for a := 0; a < r; a++ {
			for b := 0; b < r; b++ {
				s := 0.0
				for j := 0; j < r; j++ {
					s += TtdP[a*r+j] * tTape[j*r+b]
				}
				dPp[a*r+b] = s
			}
		}
		// dT += (dP + dP') · T · Pp  — explicit transpose handles asymmetric dP.
		// (Row-0 backprop below makes dP asymmetric; the 2×dP form was
		// the factor-of-2 bug found during prototyping.)
		TPp := tape.TPp
		for i := 0; i < r; i++ {
			for j := 0; j < r; j++ {
				s := 0.0
				for k := 0; k < r; k++ {
					s += tTape[i*r+k] * Pp[k*r+j]
				}
				TPp[i*r+j] = s
			}
		}
		for i := 0; i < r; i++ {
			for j := 0; j < r; j++ {
				s := 0.0
				for k := 0; k < r; k++ {
					s += (dP[i*r+k] + dP[k*r+i]) * TPp[k*r+j]
				}
				dT[i*r+j] += s
			}
		}
		// dR += (dP + dP') · R
		for i := 0; i < r; i++ {
			for j := 0; j < r; j++ {
				dR[i] += (dP[i*r+j] + dP[j*r+i]) * R[j]
			}
		}

		// --- Backprop ap = a + K·v ---
		dAin := tape.dAin
		dK := tape.dK
		for i := 0; i < r; i++ {
			dAin[i] = dap[i]
			dK[i] = dap[i] * v
			dV += dap[i] * K[i]
		}

		// --- Backprop Pp[i,j] = P[i,j] - K[i] · row0(P)[j] ---
		// dP_in[i,j] += dPp[i,j]
		// dK[i] -= sum_j dPp[i,j] · row0[j]
		// dP_in[0,j] -= sum_i dPp[i,j] · K[i]
		dPin := tape.dPin
		for i := 0; i < rr; i++ {
			dPin[i] = dPp[i]
		}
		for i := 0; i < r; i++ {
			for j := 0; j < r; j++ {
				dK[i] -= dPp[i*r+j] * Pt[j] // row0[j] = P[j]
			}
		}
		for j := 0; j < r; j++ {
			for i := 0; i < r; i++ {
				dPin[j] -= dPp[i*r+j] * K[i]
			}
		}

		// --- Backprop K[i] = P[i,0] / F ---
		// dP_in[i,0] += dK[i] / F
		// dF -= sum_i dK[i] · K[i] / F
		for i := 0; i < r; i++ {
			dPin[i*r] += dK[i] * invF
			dF -= dK[i] * K[i] * invF
		}

		// --- Backprop F = P[0,0] ---
		dPin[0] += dF

		// --- Backprop v = y_t - a[0] ---
		dAin[0] -= dV

		// Move to previous step. Swap dA/dAin and dP/dPin via copy.
		copy(dA, dAin)
		copy(dP, dPin)
	}

	// Extract gradients.
	// dT[i, 0] for i<p contains dL/dphi[i].
	gradPhi = make([]float64, p)
	for i := 0; i < p; i++ {
		gradPhi[i] = dT[i*r]
	}
	// dR[j+1] for j<q contains dL/dtheta[j].
	gradTheta = make([]float64, q)
	for j := 0; j < q; j++ {
		if j+1 < r {
			gradTheta[j] = dR[j+1]
		}
	}

	return negLogLik, s2, gradPhi, gradTheta, true
}
