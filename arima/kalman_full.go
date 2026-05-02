package arima

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// kalmanARIMAFull computes the Gaussian log-likelihood for ARIMA(p,d,q)(P,D,Q,m)
// using the full integrated state-space form (no pre-differencing).
//
// Mirrors `stats::makeARIMA` + `C_ARIMA_Like` from R: state dimension is
// r + d + D*m where r = max(p, q+1). Integrated states are initialized with
// diffuse (large) covariance kappa.
//
// y is the *un-differenced* training series. phi/theta refer to the
// non-seasonal AR/MA; sPhi/sTheta to seasonal AR/MA. Internal expansion
// produces the full ARMA polynomials.
//
// Returns (negLogLik, sigma2_concentrated, innovations).
func kalmanARIMAFull(
	y []float64, d, m, D int,
	phi, theta, sPhi, sTheta []float64,
	kappa float64,
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

	// Initial state covariance:
	//   ARMA block (r x r): stationary
	//   Integrated block (dInt x dInt diagonal): kappa
	P0 := mat.NewDense(rd, rd, nil)
	if r > 0 {
		Tarma := mat.NewDense(r, r, nil)
		for i := 0; i < r; i++ {
			if i < p {
				Tarma.Set(i, 0, fullPhi[i])
			}
			if i+1 < r {
				Tarma.Set(i, i+1, 1)
			}
		}
		Rarma := mat.NewVecDense(r, nil)
		Rarma.SetVec(0, 1)
		for j := 0; j < q; j++ {
			if j+1 < r {
				Rarma.SetVec(j+1, fullTheta[j])
			}
		}
		QQ := mat.NewDense(r, r, nil)
		QQ.Mul(Rarma, Rarma.T())
		Pst, ok := stationaryCov(Tarma, QQ, r)
		if ok {
			for i := 0; i < r; i++ {
				for j := 0; j < r; j++ {
					P0.Set(i, j, Pst.At(i, j))
				}
			}
		} else {
			// fallback: large
			for i := 0; i < r; i++ {
				P0.Set(i, i, 1)
			}
		}
	}
	for i := 0; i < dInt; i++ {
		P0.Set(r+i, r+i, kappa)
	}

	a := mat.NewVecDense(rd, nil) // mean = 0
	P := mat.DenseCopyOf(P0)

	n := len(y)
	innov := make([]float64, n)
	fSteps := make([]float64, n)
	vSteps := make([]float64, n)

	for t := 0; t < n; t++ {
		// y_t - Z*a
		predY := 0.0
		for i, zi := range zRow {
			predY += zi * a.AtVec(i)
		}
		v := y[t] - predY
		// F_t = Z * P * Z^T
		F := 0.0
		// Compute P * Z^T → vector of length rd, then dot with Z.
		PzT := make([]float64, rd)
		for i := 0; i < rd; i++ {
			s := 0.0
			for j := 0; j < rd; j++ {
				s += P.At(i, j) * zRow[j]
			}
			PzT[i] = s
		}
		for i, zi := range zRow {
			F += zi * PzT[i]
		}
		if F <= 0 || math.IsNaN(F) {
			return math.Inf(1), 0, innov
		}
		// K_t = P * Z^T / F
		K := make([]float64, rd)
		for i := 0; i < rd; i++ {
			K[i] = PzT[i] / F
		}
		// Update a: a + K*v
		for i := 0; i < rd; i++ {
			a.SetVec(i, a.AtVec(i)+K[i]*v)
		}
		// Update P: P - K * (Z * P) (rank-1)
		// First compute ZP = Z * P (1 x rd)
		ZP := make([]float64, rd)
		for j := 0; j < rd; j++ {
			s := 0.0
			for i := 0; i < rd; i++ {
				s += zRow[i] * P.At(i, j)
			}
			ZP[j] = s
		}
		for i := 0; i < rd; i++ {
			ki := K[i]
			for j := 0; j < rd; j++ {
				P.Set(i, j, P.At(i, j)-ki*ZP[j])
			}
		}

		innov[t] = v
		fSteps[t] = F
		vSteps[t] = v

		// Predict: a = T*a, P = T P T^T + RR^T.
		newA := mat.NewVecDense(rd, nil)
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

	// Exact-diffuse adjustment (Koopman 1997 / R's `ARIMA_Like`):
	// drop the first dInt observations from the likelihood sum because
	// they are dominated by the kappa prior. This recovers the proper
	// likelihood that R and statsmodels report.
	skip := dInt
	if skip > n {
		skip = n
	}
	useN := n - skip
	if useN <= 0 {
		return math.Inf(1), 0, innov
	}
	logF := 0.0
	sumVF := 0.0
	for t := skip; t < n; t++ {
		logF += math.Log(fSteps[t])
		sumVF += vSteps[t] * vSteps[t] / fSteps[t]
	}
	if sumVF <= 0 {
		return math.Inf(1), 0, innov
	}
	s2 := sumVF / float64(useN)
	negLL := 0.5 * (float64(useN)*(math.Log(2*math.Pi*s2)+1) + logF)
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
