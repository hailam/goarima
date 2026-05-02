package arima

import (
	"math"

	"gonum.org/v1/gonum/mat"
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
// r + d + D*m where r = max(p, q+1). Integrated states are initialized with
// a large diffuse covariance kappa, and observations whose prediction
// variance exceeds 1e4 (R's threshold) are skipped from the likelihood sum.
// This matches R's behaviour for most models; for AR-with-seasonal-
// differencing models on high-magnitude data, the kappa-leakage at the
// transition between the diffuse phase and steady state can lead the
// optimizer to a different (lower-IC) local maximum than R reports.
//
// Returns (negLogLik, sigma2_concentrated, innovations).
func kalmanARIMAFull(
	y []float64, d, m, D int,
	phi, theta, sPhi, sTheta []float64,
	kappa float64,
) (negLogLik, sigma2 float64, innovations []float64) {
	return kalmanARIMAFullImpl(y, d, m, D, phi, theta, sPhi, sTheta, kappa, DiffuseR)
}

func kalmanARIMAFullImpl(
	y []float64, d, m, D int,
	phi, theta, sPhi, sTheta []float64,
	kappa float64, conv DiffuseConv,
) (negLogLik, sigma2 float64, innovations []float64) {
	if kappa == 0 {
		kappa = 1e6
	}
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

	// Build the differencing polynomial coefficients (R's `Delta`).
	deltaCoefs := buildDeltaCoeffs(d, D, m)
	dInt := len(deltaCoefs)

	rd := r + dInt

	// Build T (rd x rd):
	//   Top-left r x r: companion-form ARMA dynamics
	//   Row r (zero-indexed): reconstructs y_t from ARMA + integrated history
	//   Rows r+1..rd-1: shift integrated history one slot down
	T := mat.NewDense(rd, rd, nil)
	for i := 0; i < r; i++ {
		if i < p {
			T.Set(i, 0, fullPhi[i])
		}
		if i+1 < r {
			T.Set(i, i+1, 1)
		}
	}
	if dInt > 0 {
		// Row r is Z = (1, 0, ..., 0, deltaCoefs...).
		T.Set(r, 0, 1)
		for j, dc := range deltaCoefs {
			T.Set(r, r+j, dc)
		}
		// Rows r+1..rd-1: integrated shift register
		for i := 1; i < dInt; i++ {
			T.Set(r+i, r+i-1, 1)
		}
	}

	// Z = (1, 0, ..., 0, deltaCoefs...) (length rd)
	zRow := make([]float64, rd)
	zRow[0] = 1
	if dInt > 0 {
		copy(zRow[r:], deltaCoefs)
	}

	// R column (rd-vector) = (1, theta_1, ..., theta_{r-1}, 0, ..., 0)
	rCol := mat.NewVecDense(rd, nil)
	rCol.SetVec(0, 1)
	for j := 0; j < q; j++ {
		if j+1 < r {
			rCol.SetVec(j+1, fullTheta[j])
		}
	}
	// RR^T (sigma^2 will scale at the end)
	RRt := mat.NewDense(rd, rd, nil)
	RRt.Mul(rCol, rCol.T())

	// Initial covariance: ARMA block via Gardner's stationary algorithm
	// (R's getQ0), integrated block diagonal kappa.
	P0 := mat.NewDense(rd, rd, nil)
	if r > 0 {
		Pst := stationaryCovGardner(fullPhi, fullTheta)
		pr, _ := Pst.Dims()
		for i := 0; i < pr; i++ {
			for j := 0; j < pr; j++ {
				P0.Set(i, j, Pst.At(i, j))
			}
		}
	}
	// Exact diffuse Kalman filter (Durbin-Koopman 2003, eqs 5.10-5.18).
	// Decompose P = lim_{kappa→∞} kappa * P_inf + P_*. P_inf carries the
	// rank-deficient diffuse prior (identity on integrated states); P_* is
	// the regular finite covariance (stationary ARMA on top, zero elsewhere).
	_ = kappa // unused with the exact algorithm
	Pstar := P0
	Pinf := mat.NewDense(rd, rd, nil)
	for i := 0; i < dInt; i++ {
		Pinf.Set(r+i, r+i, 1)
	}

	a := mat.NewVecDense(rd, nil)
	n := len(y)
	innov := make([]float64, n)
	logF := 0.0
	sumVF := 0.0
	nu := 0
	diffuseDone := dInt == 0
	const tol = 1e-12

	for t := 0; t < n; t++ {
		// v_t = y_t - Z * a_{t|t-1}
		predY := 0.0
		for i, zi := range zRow {
			predY += zi * a.AtVec(i)
		}
		v := y[t] - predY
		innov[t] = v

		// M_inf = P_inf * Z'  (column vector)
		// M_*   = P_*   * Z'
		Minf := make([]float64, rd)
		Mstar := make([]float64, rd)
		for i := 0; i < rd; i++ {
			si := 0.0
			ss := 0.0
			for j := 0; j < rd; j++ {
				si += Pinf.At(i, j) * zRow[j]
				ss += Pstar.At(i, j) * zRow[j]
			}
			Minf[i] = si
			Mstar[i] = ss
		}
		Finf := 0.0
		Fstar := 0.0
		for i, zi := range zRow {
			Finf += zi * Minf[i]
			Fstar += zi * Mstar[i]
		}

		if !diffuseDone && Finf > tol {
			// === Exact diffuse step (kappa→∞ limit of standard Kalman) ===
			invFinf := 1.0 / Finf
			Kinf := make([]float64, rd)
			for i := 0; i < rd; i++ {
				Kinf[i] = Minf[i] * invFinf
			}
			for i := 0; i < rd; i++ {
				a.SetVec(i, a.AtVec(i)+Kinf[i]*v)
			}
			fStarOverFinf2 := Fstar * invFinf * invFinf
			for i := 0; i < rd; i++ {
				for j := 0; j < rd; j++ {
					Pinf.Set(i, j, Pinf.At(i, j)-Kinf[i]*Minf[j])
					Pstar.Set(i, j,
						Pstar.At(i, j)+
							fStarOverFinf2*Minf[i]*Minf[j]-
							invFinf*(Minf[i]*Mstar[j]+Mstar[i]*Minf[j]))
				}
			}
			// Both R's stats::arima and statsmodels SARIMAX use n_effective =
			// n_std observations in the concentrated likelihood — diffuse-phase
			// observations are excluded from logF and v²/F. The DiffuseConv
			// flag is reserved for future variants that diverge here; today
			// both conventions use the same likelihood form.
			_ = conv
		} else {
			// === Standard step on P_* (no remaining diffuse rank) ===
			diffuseDone = true
			F := Fstar
			if F <= tol || math.IsNaN(F) {
				return math.Inf(1), 0, innov
			}
			invF := 1.0 / F
			K := make([]float64, rd)
			for i := 0; i < rd; i++ {
				K[i] = Mstar[i] * invF
			}
			for i := 0; i < rd; i++ {
				a.SetVec(i, a.AtVec(i)+K[i]*v)
			}
			for i := 0; i < rd; i++ {
				for j := 0; j < rd; j++ {
					Pstar.Set(i, j, Pstar.At(i, j)-K[i]*Mstar[j])
				}
			}
			logF += math.Log(F)
			sumVF += v * v * invF
			nu++
		}

		// Predict to t+1: a' = T*a, P_inf' = T*P_inf*T', P_*' = T*P_**T' + RR'
		newA := mat.NewVecDense(rd, nil)
		newA.MulVec(T, a)
		a = newA

		var TPstar, TPstarTt, newPstar mat.Dense
		TPstar.Mul(T, Pstar)
		TPstarTt.Mul(&TPstar, T.T())
		newPstar.Add(&TPstarTt, RRt)
		Pstar = mat.DenseCopyOf(&newPstar)

		if !diffuseDone {
			var TPinf, TPinfTt mat.Dense
			TPinf.Mul(T, Pinf)
			TPinfTt.Mul(&TPinf, T.T())
			Pinf = mat.DenseCopyOf(&TPinfTt)
			tr := 0.0
			for i := 0; i < rd; i++ {
				tr += Pinf.At(i, i)
			}
			if tr < tol {
				diffuseDone = true
			}
		}
	}

	if nu == 0 || sumVF <= 0 {
		return math.Inf(1), 0, innov
	}
	// Standard concentrated-σ² Gaussian likelihood (R's stats::arima form;
	// statsmodels SARIMAX uses the same formula via `nobs_effective = n_std`).
	s2 := sumVF / float64(nu)
	negLL := 0.5 * (float64(nu)*(math.Log(2*math.Pi*s2)+1) + logF)
	_ = conv // both DiffuseR and DiffuseStatsmodels use this likelihood formula
	return negLL, s2, innov
}

// buildDeltaCoeffs returns the negated tail of the differencing polynomial
// (1-B)^d (1-B^m)^D — that is, R's `Delta` vector after `Delta <- -Delta[-1L]`.
//
// Length of returned slice = d + D*m.
func buildDeltaCoeffs(d, D, m int) []float64 {
	// Build polynomial `poly` representing (1-B)^d (1-B^m)^D. Coefficient
	// at index i is the multiplier of B^i.
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
	// Drop leading 1 and negate the rest, per R's `Delta <- -Delta[-1L]`.
	if len(poly) <= 1 {
		return nil
	}
	out := make([]float64, len(poly)-1)
	for i := 1; i < len(poly); i++ {
		out[i-1] = -poly[i]
	}
	return out
}

// polyMul multiplies two polynomial coefficient slices (lowest-degree first).
func polyMul(a, b []float64) []float64 {
	out := make([]float64, len(a)+len(b)-1)
	for i, ai := range a {
		for j, bj := range b {
			out[i+j] += ai * bj
		}
	}
	return out
}
