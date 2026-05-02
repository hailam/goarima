package arima

import (
	"math"
)

// kalmanARIMAFullConv runs the exact-diffuse Kalman filter with a selectable
// likelihood convention.
func kalmanARIMAFullConv(
	y []float64, d, m, D int,
	phi, theta, sPhi, sTheta []float64,
	kappa float64, conv DiffuseConv,
) (negLogLik, sigma2 float64, innovations []float64) {
	return kalmanARIMAFullImpl(y, d, m, D, phi, theta, sPhi, sTheta, kappa, conv)
}

// kalmanARIMAFull computes the Gaussian log-likelihood for ARIMA(p,d,q)(P,D,Q,m)
// using the full integrated state-space form (no pre-differencing).
//
// Mirrors `stats::makeARIMA` + `C_ARIMA_Like` from R: state dimension is
// r + d + D*m where r = max(p, q+1). Integrated states get a diffuse prior
// (rank dInt identity in P_inf); ARMA states get the Gardner-Harvey-Phillips
// stationary covariance in P_*. The exact diffuse Kalman filter
// (Durbin-Koopman 2003 §5.2) eliminates kappa-leakage from the original
// approximate recursion.
//
// The hot loop uses flat row-major float64 buffers and exploits the
// sparsity of T (a few nonzeros per row, since T is the companion form of
// the ARMA polynomial plus a shift register for the integrated states).
// All scratch buffers are allocated once before the loop.
//
// Returns (negLogLik, sigma2_concentrated, innovations).
func kalmanARIMAFull(
	y []float64, d, m, D int,
	phi, theta, sPhi, sTheta []float64,
	kappa float64,
) (negLogLik, sigma2 float64, innovations []float64) {
	return kalmanARIMAFullImpl(y, d, m, D, phi, theta, sPhi, sTheta, kappa, DiffuseR)
}

// tNZ is one nonzero entry of the transition matrix T: T[i,j] = v.
type tNZ struct {
	i, j int
	v    float64
}

func kalmanARIMAFullImpl(
	y []float64, d, m, D int,
	phi, theta, sPhi, sTheta []float64,
	kappa float64, conv DiffuseConv,
) (negLogLik, sigma2 float64, innovations []float64) {
	if kappa == 0 {
		kappa = 1e6
	}
	_ = kappa // exact diffuse algorithm — kappa is only a fallback marker
	fullPhi := expandSARMA(phi, sPhi, m)
	fullTheta := expandSMA(theta, sTheta, m)
	p := len(fullPhi)
	q := len(fullTheta)
	r := p
	if q+1 > r {
		r = q + 1
	}
	if r == 0 {
		// pure white noise
		sse := 0.0
		for _, v := range y {
			sse += v * v
		}
		s2 := sse / float64(len(y))
		return float64(len(y)) / 2 * (math.Log(2*math.Pi*s2) + 1), s2, nil
	}

	deltaCoefs := buildDeltaCoeffs(d, D, m)
	dInt := len(deltaCoefs)
	rd := r + dInt
	rd2 := rd * rd

	// ---- T as a sparse list of nonzero (i, j, v) entries.
	// Top r rows: ARMA companion form. T[i,0] = phi[i] (if i<p), T[i,i+1]=1.
	// Row r (when dInt>0): reconstructs y_t from ARMA + integrated history.
	// Rows r+1..rd-1: shift register for integrated states.
	maxNZ := 2*r + 1 + dInt + dInt
	nzT := make([]tNZ, 0, maxNZ)
	for i := 0; i < r; i++ {
		if i < p {
			nzT = append(nzT, tNZ{i, 0, fullPhi[i]})
		}
		if i+1 < r {
			nzT = append(nzT, tNZ{i, i + 1, 1})
		}
	}
	if dInt > 0 {
		nzT = append(nzT, tNZ{r, 0, 1})
		for j, dc := range deltaCoefs {
			if dc != 0 {
				nzT = append(nzT, tNZ{r, r + j, dc})
			}
		}
		for i := 1; i < dInt; i++ {
			nzT = append(nzT, tNZ{r + i, r + i - 1, 1})
		}
	}

	// Z row: (1, 0..., deltaCoefs...).
	zRow := make([]float64, rd)
	zRow[0] = 1
	if dInt > 0 {
		copy(zRow[r:], deltaCoefs)
	}

	// R column: (1, theta_1, ..., theta_{r-1}, 0, ..., 0). RR' precomputed.
	rCol := make([]float64, rd)
	rCol[0] = 1
	for j := 0; j < q; j++ {
		if j+1 < r {
			rCol[j+1] = fullTheta[j]
		}
	}
	RRt := make([]float64, rd2)
	for i := 0; i < rd; i++ {
		ri := rCol[i]
		if ri == 0 {
			continue
		}
		base := i * rd
		for j := 0; j < rd; j++ {
			RRt[base+j] = ri * rCol[j]
		}
	}

	// Initial covariance.
	Pstar := make([]float64, rd2)
	Pinf := make([]float64, rd2)
	if r > 0 {
		Pst := stationaryCovGardner(fullPhi, fullTheta)
		pr, _ := Pst.Dims()
		for i := 0; i < pr; i++ {
			base := i * rd
			for j := 0; j < pr; j++ {
				Pstar[base+j] = Pst.At(i, j)
			}
		}
	}
	for i := 0; i < dInt; i++ {
		Pinf[(r+i)*rd+(r+i)] = 1
	}

	a := make([]float64, rd)
	n := len(y)
	innov := make([]float64, n)
	logF := 0.0
	sumVF := 0.0
	nu := 0
	diffuseDone := dInt == 0
	const tol = 1e-12

	// Pre-allocated scratch buffers (reused every step).
	newA := make([]float64, rd)
	Minf := make([]float64, rd)
	Mstar := make([]float64, rd)
	K := make([]float64, rd)
	TPstar := make([]float64, rd2)
	newPstar := make([]float64, rd2)
	TPinf := make([]float64, rd2)
	newPinf := make([]float64, rd2)

	for t := 0; t < n; t++ {
		// v_t = y_t - Z * a
		predY := 0.0
		for i := 0; i < rd; i++ {
			predY += zRow[i] * a[i]
		}
		v := y[t] - predY
		innov[t] = v

		// Minf = Pinf @ z, Mstar = Pstar @ z (row-major matvec).
		for i := 0; i < rd; i++ {
			base := i * rd
			si := 0.0
			ss := 0.0
			for j := 0; j < rd; j++ {
				zj := zRow[j]
				si += Pinf[base+j] * zj
				ss += Pstar[base+j] * zj
			}
			Minf[i] = si
			Mstar[i] = ss
		}
		Finf := 0.0
		Fstar := 0.0
		for i := 0; i < rd; i++ {
			Finf += zRow[i] * Minf[i]
			Fstar += zRow[i] * Mstar[i]
		}

		if !diffuseDone && Finf > tol {
			invFinf := 1.0 / Finf
			for i := 0; i < rd; i++ {
				K[i] = Minf[i] * invFinf
				a[i] += K[i] * v
			}
			fStarOverFinf2 := Fstar * invFinf * invFinf
			for i := 0; i < rd; i++ {
				ki := K[i]
				mi := Minf[i]
				msi := Mstar[i]
				base := i * rd
				for j := 0; j < rd; j++ {
					mj := Minf[j]
					Pinf[base+j] -= ki * mj
					Pstar[base+j] += fStarOverFinf2*mi*mj - invFinf*(mi*Mstar[j]+msi*mj)
				}
			}
		} else {
			diffuseDone = true
			F := Fstar
			if F <= tol || math.IsNaN(F) {
				return math.Inf(1), 0, innov
			}
			invF := 1.0 / F
			for i := 0; i < rd; i++ {
				K[i] = Mstar[i] * invF
				a[i] += K[i] * v
			}
			for i := 0; i < rd; i++ {
				ki := K[i]
				base := i * rd
				for j := 0; j < rd; j++ {
					Pstar[base+j] -= ki * Mstar[j]
				}
			}
			logF += math.Log(F)
			sumVF += v * v * invF
			nu++
		}
		_ = conv // both DiffuseR and DiffuseStatsmodels use the same likelihood

		// ---- Predict step.
		// a' = T @ a (sparse).
		for i := 0; i < rd; i++ {
			newA[i] = 0
		}
		for _, e := range nzT {
			newA[e.i] += e.v * a[e.j]
		}
		copy(a, newA)

		// Pstar' = T @ Pstar @ T' + RR'.
		// Step 1: TPstar = T @ Pstar (sparse-T, dense-P, row-major matmul).
		for k := range TPstar {
			TPstar[k] = 0
		}
		for _, e := range nzT {
			ti := e.i * rd
			tj := e.j * rd
			tv := e.v
			for j := 0; j < rd; j++ {
				TPstar[ti+j] += tv * Pstar[tj+j]
			}
		}
		// Step 2: newPstar = TPstar @ T' + RR'. (T' has nonzeros at the
		// transposed positions, so for each T-nz (kT, jT, vT) we add
		// vT*TPstar[i, jT] to newPstar[i, kT].)
		copy(newPstar, RRt)
		for i := 0; i < rd; i++ {
			row := i * rd
			for _, e := range nzT {
				newPstar[row+e.i] += TPstar[row+e.j] * e.v
			}
		}
		Pstar, newPstar = newPstar, Pstar

		if !diffuseDone {
			// Pinf' = T @ Pinf @ T'.
			for k := range TPinf {
				TPinf[k] = 0
			}
			for _, e := range nzT {
				ti := e.i * rd
				tj := e.j * rd
				tv := e.v
				for j := 0; j < rd; j++ {
					TPinf[ti+j] += tv * Pinf[tj+j]
				}
			}
			for k := range newPinf {
				newPinf[k] = 0
			}
			for i := 0; i < rd; i++ {
				row := i * rd
				for _, e := range nzT {
					newPinf[row+e.i] += TPinf[row+e.j] * e.v
				}
			}
			Pinf, newPinf = newPinf, Pinf
			tr := 0.0
			for i := 0; i < rd; i++ {
				tr += Pinf[i*rd+i]
			}
			if tr < tol {
				diffuseDone = true
			}
		}
	}

	if nu == 0 || sumVF <= 0 {
		return math.Inf(1), 0, innov
	}
	s2 := sumVF / float64(nu)
	negLL := 0.5 * (float64(nu)*(math.Log(2*math.Pi*s2)+1) + logF)
	_ = conv
	return negLL, s2, innov
}

// buildDeltaCoeffs returns the negated tail of the differencing polynomial
// (1-B)^d (1-B^m)^D — that is, R's `Delta` vector after `Delta <- -Delta[-1L]`.
//
// Length of returned slice = d + D*m.
func buildDeltaCoeffs(d, D, m int) []float64 {
	poly := []float64{1}
	for i := 0; i < d; i++ {
		poly = polyMul(poly, []float64{1, -1})
	}
	if D > 0 && m > 1 {
		seas := make([]float64, m+1)
		seas[0] = 1
		seas[m] = -1
		for i := 0; i < D; i++ {
			poly = polyMul(poly, seas)
		}
	}
	if len(poly) <= 1 {
		return nil
	}
	out := make([]float64, len(poly)-1)
	for i := 1; i < len(poly); i++ {
		out[i-1] = -poly[i]
	}
	return out
}

func polyMul(a, b []float64) []float64 {
	out := make([]float64, len(a)+len(b)-1)
	for i, ai := range a {
		for j, bj := range b {
			out[i+j] += ai * bj
		}
	}
	return out
}
