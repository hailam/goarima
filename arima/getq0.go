package arima

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// PublicGardner exposes Gardner's stationary cov for parity diagnostics.
func PublicGardner(phi, theta []float64) [][]float64 {
	flat, r := stationaryCovGardner(phi, theta)
	out := make([][]float64, r)
	for i := 0; i < r; i++ {
		row := make([]float64, r)
		copy(row, flat[i*r:(i+1)*r])
		out[i] = row
	}
	return out
}

// PublicInclu2 exposes Gardner's inclu2 path (no Smith dispatch) for
// parity diagnostics. Used by the GARD-OPT-1 cross-check tests.
func PublicInclu2(phi, theta []float64) [][]float64 {
	flat, r := stationaryCovInclu2OnlyInto(&gardnerWorkspace{}, phi, theta)
	if r == 0 {
		return nil
	}
	out := make([][]float64, r)
	for i := 0; i < r; i++ {
		row := make([]float64, r)
		copy(row, flat[i*r:(i+1)*r])
		out[i] = row
	}
	return out
}

// PublicSmith exposes Smith's doubling Lyapunov solver for parity
// diagnostics (GARD-OPT-1). Returns nil if Smith fails to converge
// (the production dispatch falls back to inclu2 in that case).
func PublicSmith(phi, theta []float64) [][]float64 {
	p := len(phi)
	q := len(theta)
	r := p
	if q+1 > r {
		r = q + 1
	}
	if r == 0 {
		return nil
	}
	var ws gardnerWorkspace
	flat, ok := stationaryCovSmithInto(&ws, phi, theta, r)
	if !ok {
		return nil
	}
	out := make([][]float64, r)
	for i := 0; i < r; i++ {
		row := make([]float64, r)
		copy(row, flat[i*r:(i+1)*r])
		out[i] = row
	}
	return out
}

// stationaryCovGardner computes the stationary covariance of the ARMA(p,q)
// state-space form using Gardner's algorithm — a numerically-stable
// alternative to direct discrete-Lyapunov solution.
//
// Mirrors R's `stats::C_getQ0` (in src/library/stats/src/arima.c). Used during
// the exact diffuse Kalman filter to seed the ARMA block of P_*.
//
// phi: non-seasonal AR coefficients (length p, fully expanded if seasonal)
// theta: non-seasonal MA coefficients (length q, fully expanded if seasonal)
// Returns the flat row-major r×r covariance and r itself, where r = max(p, q+1).
// The returned slice has length r*r; the caller can copy or alias as needed.
func stationaryCovGardner(phi, theta []float64) ([]float64, int) {
	var ws gardnerWorkspace
	return stationaryCovGardnerInto(&ws, phi, theta)
}

// stationaryCovGardnerInto is the buffer-reuse variant of
// stationaryCovGardner. All ~7 internal buffers come from `ws` instead
// of being allocated per call. The returned []float64 aliases ws.P,
// valid until the next call on the same workspace. Pre-pool, this
// function was the single largest allocator in AutoArima
// (~36% of total per-Fit allocations).
//
// GARD-OPT-1 dispatch: at r ≥ 20 with p > 0 (AR-containing model), the
// inclu2 Givens-rotation path costs O(r⁴) and dominates wallclock for
// high-period seasonal models (m=24, 52, etc.). Smith's doubling
// iteration is O(r³) with small constants and is dispatched
// automatically below. For p = 0 (pure MA) the existing closed-form
// backsubstitution is already optimal and stays. For small r (≤ 19)
// inclu2's tight inner loop wins via cache locality.
const gardSmithThresholdR = 20

func stationaryCovGardnerInto(ws *gardnerWorkspace, phi, theta []float64) ([]float64, int) {
	p := len(phi)
	q := len(theta)
	r := p
	if q+1 > r {
		r = q + 1
	}
	if r == 0 {
		return nil, 0
	}
	if p > 0 && r >= gardSmithThresholdR {
		if out, ok := stationaryCovSmithInto(ws, phi, theta, r); ok {
			return out, r
		}
		// Fall through to inclu2 on Smith failure (e.g., non-stable T).
	}
	return stationaryCovInclu2OnlyInto(ws, phi, theta)
}

// stationaryCovInclu2OnlyInto runs the original Gardner inclu2 path
// without the Smith dispatch — used by GARD-OPT-1's parity tests as
// the reference implementation, and as the fallback for non-stable
// fits where Smith's doubling iteration cannot converge.
func stationaryCovInclu2OnlyInto(ws *gardnerWorkspace, phi, theta []float64) ([]float64, int) {
	p := len(phi)
	q := len(theta)
	r := p
	if q+1 > r {
		r = q + 1
	}
	if r == 0 {
		return nil, 0
	}
	np := r * (r + 1) / 2
	nrbar := np * (np - 1) / 2
	if nrbar < 0 {
		nrbar = 0
	}

	// V[ind] = vi * vj where vk = (k == 0 ? 1 : (k-1 < q ? theta[k-1] : 0))
	ws.V = ensureLen(ws.V, np)
	V := ws.V
	{
		ind := 0
		for j := 0; j < r; j++ {
			vj := 0.0
			if j == 0 {
				vj = 1
			} else if j-1 < q {
				vj = theta[j-1]
			}
			for i := j; i < r; i++ {
				vi := 0.0
				if i == 0 {
					vi = 1
				} else if i-1 < q {
					vi = theta[i-1]
				}
				V[ind] = vi * vj
				ind++
			}
		}
	}

	ws.P = ensureLenZ(ws.P, r*r) // zeroed because we write only some cells
	P := ws.P                    // final r×r matrix, flat column-major like R

	if r == 1 {
		if p == 0 {
			P[0] = 1
		} else {
			P[0] = 1 / (1 - phi[0]*phi[0])
		}
		return P, 1
	}

	if p > 0 {
		// Givens-rotation based solver via inclu2.
		ws.xnext = ensureLenZ(ws.xnext, np)
		ws.xrow = ensureLenZ(ws.xrow, np)
		ws.rbar = ensureLenZ(ws.rbar, nrbar)
		ws.thetab = ensureLenZ(ws.thetab, np)
		ws.Pbuf = ensureLenZ(ws.Pbuf, np)
		xnext := ws.xnext
		xrow := ws.xrow
		rbar := ws.rbar
		thetab := ws.thetab
		Pbuf := ws.Pbuf

		ind := 0
		ind1 := -1
		npr := np - r
		npr1 := npr + 1
		indj := npr
		ind2 := npr - 1

		for j := 0; j < r; j++ {
			phij := 0.0
			if j < p {
				phij = phi[j]
			}
			xnext[indj] = 0
			indj++
			indi := npr1 + j
			for i := j; i < r; i++ {
				ynext := V[ind]
				ind++
				phii := 0.0
				if i < p {
					phii = phi[i]
				}
				if j != r-1 {
					xnext[indj] = -phii
					if i != r-1 {
						xnext[indi] -= phij
						ind1++
						xnext[ind1] = -1
					}
				}
				xnext[npr] = -phii * phij
				ind2++
				if ind2 >= np {
					ind2 = 0
				}
				xnext[ind2] += 1
				inclu2(np, xnext, xrow, ynext, Pbuf, rbar, thetab)
				xnext[ind2] = 0
				if i != r-1 {
					xnext[indi] = 0
					indi++
					xnext[ind1] = 0
				}
			}
		}

		// Backsubstitution
		ithisr := nrbar - 1
		im := np - 1
		for i := 0; i < np; i++ {
			bi := thetab[im]
			jm := np - 1
			for j := 0; j < i; j++ {
				bi -= rbar[ithisr] * Pbuf[jm]
				ithisr--
				jm--
			}
			Pbuf[im] = bi
			im--
		}

		// Re-order P (R's "now re-order p")
		// Take last r entries of Pbuf and store in xnext
		ind = npr
		for i := 0; i < r; i++ {
			xnext[i] = Pbuf[ind]
			ind++
		}
		// Shift first npr entries of Pbuf to the end
		ind = np - 1
		ind1 = npr - 1
		for i := 0; i < npr; i++ {
			Pbuf[ind] = Pbuf[ind1]
			ind--
			ind1--
		}
		for i := 0; i < r; i++ {
			Pbuf[i] = xnext[i]
		}

		// Unpack Pbuf into P (full matrix). R's unpack:
		//   for(i = r-1, ind = np; i > 0; i--)
		//       for(j = r-1; j >= i; j--) P[r*i + j] = P[--ind];
		//   for(i = 0; i < r-1; i++) for(j = i+1; j < r; j++)
		//       P[i + r*j] = P[j + r*i];
		// Here P is treated as column-major flat: P[r*i + j] = P[j, i].
		// We'll work in the same indexing then return as row-major.
		ind = np
		for i := r - 1; i > 0; i-- {
			for j := r - 1; j >= i; j-- {
				ind--
				P[r*i+j] = Pbuf[ind]
			}
		}
		// First column (i=0) stays as Pbuf[0..r-1]
		for j := 0; j < r; j++ {
			P[r*0+j] = Pbuf[j]
		}
		// Symmetrize: P[i, j] = P[j, i] for i < j (column-major: P[i+r*j] = P[j+r*i])
		for i := 0; i < r-1; i++ {
			for j := i + 1; j < r; j++ {
				P[i+r*j] = P[j+r*i]
			}
		}
	} else {
		// Pure MA: backsubstitution with V
		ws.Pbuf = ensureLenZ(ws.Pbuf, np)
		Pbuf := ws.Pbuf
		indn := np
		ind := np
		for i := 0; i < r; i++ {
			for j := 0; j <= i; j++ {
				ind--
				Pbuf[ind] = V[ind]
				if j != 0 {
					indn--
					Pbuf[ind] += Pbuf[indn]
				}
			}
		}
		// Unpack
		ind = np
		for i := r - 1; i > 0; i-- {
			for j := r - 1; j >= i; j-- {
				ind--
				P[r*i+j] = Pbuf[ind]
			}
		}
		for j := 0; j < r; j++ {
			P[r*0+j] = Pbuf[j]
		}
		for i := 0; i < r-1; i++ {
			for j := i + 1; j < r; j++ {
				P[i+r*j] = P[j+r*i]
			}
		}
	}

	// CDX-R1: P is symmetric after the symmetrization above, so its
	// column-major and row-major linear layouts are bit-identical
	// (P[i*r+j] equals M[i,j] in both, since M[i,j] = M[j,i]). Skip
	// the redundant conversion that previously allocated `out`.
	return P, r
}

// stationaryCovSmithInto solves the discrete Lyapunov equation
//
//	P = T·P·Tᵀ + Q
//
// for the ARMA companion form (T as built by Gardner, Q = R·Rᵀ where
// R = (1, θ_1, …, θ_{r-1})ᵀ) via Smith's (1968) doubling iteration:
//
//	P₀ = Q,  T₀ = T
//	Pₖ₊₁ = Pₖ + Tₖ · Pₖ · Tₖᵀ,   Tₖ₊₁ = Tₖ²
//
// Converges geometrically when ρ(T) < 1. Each step is two r×r matmuls
// = O(r³), so total cost is O(r³ · log₂(precision/ρ)). For ARMA stable
// fits ~6 iterations suffice. Returns (P_aliased_to_ws.P, true) on
// success; (nil, false) if the iteration fails to converge in
// smithMaxIter (typically because |ρ(T)| ≥ 1, BFGS pushed off
// stationarity).
//
// GARD-OPT-1: dispatched from stationaryCovGardnerInto when p > 0 and
// r ≥ gardSmithThresholdR — beats inclu2's O(r⁴) by 1.4× at r=14 to
// 20× at r=101.
func stationaryCovSmithInto(ws *gardnerWorkspace, phi, theta []float64, r int) ([]float64, bool) {
	const (
		smithMaxIter = 60
		smithRelTol  = 1e-15
	)
	rr := r * r
	ws.P = ensureLenZ(ws.P, rr)
	ws.smithT = ensureLenZ(ws.smithT, rr)
	ws.smithT2 = ensureLenZ(ws.smithT2, rr)
	ws.smithTP = ensureLenZ(ws.smithTP, rr)
	ws.smithDP = ensureLenZ(ws.smithDP, rr)
	P := ws.P
	T := ws.smithT
	T2 := ws.smithT2
	TP := ws.smithTP
	dP := ws.smithDP

	// Initialize T as the ARMA companion: T[i,0] = phi[i] (i<p), T[i,i+1] = 1.
	p := len(phi)
	for i := 0; i < r; i++ {
		if i < p {
			T[i*r] = phi[i]
		}
		if i+1 < r {
			T[i*r+i+1] = 1
		}
	}
	// Initialize P = R·Rᵀ where R = (1, θ_1, …, θ_{q}). Sparse outer product.
	q := len(theta)
	rvec := make([]float64, r)
	rvec[0] = 1
	for j := 0; j < q && j+1 < r; j++ {
		rvec[j+1] = theta[j]
	}
	for i := 0; i < r; i++ {
		ri := rvec[i]
		if ri == 0 {
			continue
		}
		off := i * r
		for j := 0; j < r; j++ {
			P[off+j] = ri * rvec[j]
		}
	}

	// ‖T₀‖_F² baseline for the T-shrinkage convergence test.
	// VEC-AUDIT (2026-05-10): floats.Dot tested but REGRESSED these
	// inner-loop benches by 17-22% — function-call overhead breaks
	// inlining and dominates at r² ≤ 2601. Hand-rolled stays.
	tNorm0 := 0.0
	for k := 0; k < rr; k++ {
		tNorm0 += T[k] * T[k]
	}
	if tNorm0 == 0 {
		// T = 0 means y_t = R·ε_t (no AR feedback) — P = R Rᵀ exactly.
		return P, true
	}

	for iter := 0; iter < smithMaxIter; iter++ {
		// TP = T · P
		matmulRR(TP, T, P, r)
		// dP = TP · Tᵀ  (i.e., the increment T_k · P_k · T_kᵀ)
		matmulRRt(dP, TP, T, r)

		// Apply increment.
		dNorm, pNorm := 0.0, 0.0
		for k := 0; k < rr; k++ {
			P[k] += dP[k]
			pNorm += P[k] * P[k]
			dNorm += dP[k] * dP[k]
		}

		// T = T²
		matmulRR(T2, T, T, r)
		copy(T, T2)
		tNorm := 0.0
		for k := 0; k < rr; k++ {
			tNorm += T[k] * T[k]
		}

		// Two convergence checks must BOTH hold to terminate:
		//   (a) ‖T_k‖_F / ‖T₀‖_F < tol — future increments are at most
		//       O(‖T_k‖² · ‖P‖) so once T has decayed, the truncation
		//       error is bounded.
		//   (b) ‖dP‖_F / ‖P‖_F < tol — the latest increment is small
		//       relative to current P.
		// Requiring both avoids the pure-AR pitfall where the very
		// first increment can be small (sparse Q only fills column 0)
		// even though the iteration hasn't propagated through T's
		// remaining 2^k powers.
		if tNorm <= smithRelTol*smithRelTol*tNorm0 {
			if pNorm == 0 || dNorm <= smithRelTol*smithRelTol*pNorm {
				return P, true
			}
		}
		// Diverging: ‖T_k‖_F grows by more than 10^6 over baseline →
		// T has ρ ≥ 1 (BFGS pushed off stationarity). Bail.
		if tNorm > 1e12*tNorm0 {
			return nil, false
		}
	}
	return nil, false
}

// matmulRR computes C = A · B for r×r row-major matrices via gonum's
// BLAS DGEMM. SIMD-MATMUL-1 (2026-05-10): the previous hand-rolled
// ikj loop was 1.26-2.68× slower than gonum's hand-tuned asm across
// r ∈ {14, 27, 51, 101}. Gain compounds inside Smith doubling
// (GARD-OPT-1) which calls this 2× per iteration × ~10 iterations.
//
// Per-call overhead at r=14 is sub-µs; the dispatch cost is
// dominated by the actual matmul work even at small r, so no
// threshold is needed.
func matmulRR(C, A, B []float64, r int) {
	gA := blas64.General{Rows: r, Cols: r, Stride: r, Data: A}
	gB := blas64.General{Rows: r, Cols: r, Stride: r, Data: B}
	gC := blas64.General{Rows: r, Cols: r, Stride: r, Data: C}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1.0, gA, gB, 0.0, gC)
}

// matmulRRt computes C = A · Bᵀ for r×r row-major matrices.
func matmulRRt(C, A, B []float64, r int) {
	gA := blas64.General{Rows: r, Cols: r, Stride: r, Data: A}
	gB := blas64.General{Rows: r, Cols: r, Stride: r, Data: B}
	gC := blas64.General{Rows: r, Cols: r, Stride: r, Data: C}
	blas64.Gemm(blas.NoTrans, blas.Trans, 1.0, gA, gB, 0.0, gC)
}

// inclu2 is the Givens-rotation update used by stationaryCovGardner.
//
// Mirrors stats::inclu2 in arima.c.
func inclu2(np int, xnext, xrow []float64, ynext float64, d, rbar, thetab []float64) {
	for i := 0; i < np; i++ {
		xrow[i] = xnext[i]
	}
	ithisr := 0
	for i := 0; i < np; i++ {
		if xrow[i] != 0 {
			xi := xrow[i]
			di := d[i]
			dpi := di + xi*xi
			d[i] = dpi
			cbar := di / dpi
			sbar := xi / dpi
			for k := i + 1; k < np; k++ {
				xk := xrow[k]
				rbthis := rbar[ithisr]
				xrow[k] = xk - xi*rbthis
				rbar[ithisr] = cbar*rbthis + sbar*xk
				ithisr++
			}
			xk := ynext
			ynext = xk - xi*thetab[i]
			thetab[i] = cbar*thetab[i] + sbar*xk
			if di == 0 {
				return
			}
		} else {
			ithisr += np - i - 1
		}
	}
}
