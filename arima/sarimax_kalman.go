package arima

import (
	"math"
)

// PublicSarimaxKalman exposes kalmanSARIMAX for parity diagnostics.
func PublicSarimaxKalman(y []float64, d, m, D int, phi, theta, sPhi, sTheta []float64) (float64, float64, []float64) {
	return kalmanSARIMAX(y, d, m, D, phi, theta, sPhi, sTheta, 1e6)
}

// PublicSarimaxKalmanK exposes kalmanSARIMAX with custom kappa.
func PublicSarimaxKalmanK(y []float64, d, m, D int, phi, theta, sPhi, sTheta []float64, kappa float64) (float64, float64, []float64) {
	return kalmanSARIMAX(y, d, m, D, phi, theta, sPhi, sTheta, kappa)
}

// PublicSarimaxKalmanAbs exposes kalmanSARIMAXAbs (absolute units, sigma2 as input).
func PublicSarimaxKalmanAbs(y []float64, d, m, D int, phi, theta, sPhi, sTheta []float64, sigma2, kappa float64) (float64, []float64) {
	return kalmanSARIMAXAbs(y, d, m, D, phi, theta, sPhi, sTheta, sigma2, kappa)
}

// kalmanSARIMAXAbs runs the SARIMAX Kalman filter in absolute units (sigma2
// passed as a parameter, NOT concentrated). Matches statsmodels exactly:
// kappa=1e6 is in absolute (sigma2-included) units, stationary ARMA cov is
// scaled by sigma2.
//
// Returns negLogLik (negative full Gaussian log-likelihood — no concentration).
func kalmanSARIMAXAbs(
	y []float64, d, m, D int,
	phi, theta, sPhi, sTheta []float64,
	sigma2, kappa float64,
) (negLogLik float64, innovations []float64) {
	if kappa == 0 {
		kappa = 1e6
	}
	nzT, zRow, rCol, rd, kStatesDiff := sarimaxStateSpaceSparse(phi, theta, sPhi, sTheta, d, D, m)
	if rd == 0 || sigma2 <= 0 {
		return math.Inf(1), nil
	}
	rd2 := rd * rd

	RRt := make([]float64, rd2)
	for i := 0; i < rd; i++ {
		ri := rCol[i]
		if ri == 0 {
			continue
		}
		base := i * rd
		s := sigma2 * ri
		for j := 0; j < rd; j++ {
			RRt[base+j] = s * rCol[j]
		}
	}

	P := make([]float64, rd2)
	for i := 0; i < kStatesDiff; i++ {
		P[i*rd+i] = kappa
	}
	if kArma := rd - kStatesDiff; kArma > 0 {
		fullPhi := expandSARMA(phi, sPhi, m)
		fullTheta := expandSMA(theta, sTheta, m)
		Pst, pr := stationaryCovGardner(fullPhi, fullTheta)
		for i := 0; i < pr && kStatesDiff+i < rd; i++ {
			src := i * pr
			for j := 0; j < pr && kStatesDiff+j < rd; j++ {
				P[(kStatesDiff+i)*rd+(kStatesDiff+j)] = sigma2 * Pst[src+j]
			}
		}
	}

	a := make([]float64, rd)
	n := len(y)
	innov := make([]float64, n)
	logL := 0.0
	burn := kStatesDiff

	PzT := make([]float64, rd)
	K := make([]float64, rd)
	newA := make([]float64, rd)
	TP := make([]float64, rd2)
	newP := make([]float64, rd2)

	for t := 0; t < n; t++ {
		predY := 0.0
		for i := 0; i < rd; i++ {
			predY += zRow[i] * a[i]
		}
		v := y[t] - predY
		innov[t] = v
		for i := 0; i < rd; i++ {
			base := i * rd
			s := 0.0
			for j := 0; j < rd; j++ {
				s += P[base+j] * zRow[j]
			}
			PzT[i] = s
		}
		F := 0.0
		for i := 0; i < rd; i++ {
			F += zRow[i] * PzT[i]
		}
		if F <= 0 || math.IsNaN(F) {
			return math.Inf(1), innov
		}
		invF := 1.0 / F
		for i := 0; i < rd; i++ {
			K[i] = PzT[i] * invF
			a[i] += K[i] * v
		}
		// ZP[j] = sum_i z[i]*P[i,j].
		for j := 0; j < rd; j++ {
			s := 0.0
			for i := 0; i < rd; i++ {
				s += zRow[i] * P[i*rd+j]
			}
			PzT[j] = s // reuse buffer as ZP
		}
		// KAL-2: Joseph-form covariance update (replaces rank-1 form).
		// Generalises the kalman.go specialisation (where H = [1,0,…,0]
		// makes ZP[j] = P[0,j] = row0[j]) to a general Z row vector:
		//   P_new[i,j] = P[i,j] - K[i]·ZP[j] - K[j]·ZP[i] + K[i]·K[j]·F
		// Symmetric by construction; preserves PSD against rounding.
		// Same root-cause mitigation as kalman.go's KAL-1 fix.
		//
		// Implemented as full rd×rd traversal even though the result
		// is symmetric — the alternative (upper-triangle + mirror)
		// requires strided writes to P[j*rd+i] which cache-thrash
		// more than the saved arithmetic costs. Sequential writes to
		// P[base+j] beat the symmetric optimisation in benches.
		for i := 0; i < rd; i++ {
			ki := K[i]
			zpi := PzT[i]
			base := i * rd
			for j := 0; j < rd; j++ {
				kj := K[j]
				P[base+j] += -ki*PzT[j] - kj*zpi + ki*kj*F
			}
		}
		if t >= burn {
			logL -= 0.5 * (math.Log(2*math.Pi*F) + v*v*invF)
		}
		// Predict: a' = T*a (sparse).
		for i := 0; i < rd; i++ {
			newA[i] = 0
		}
		for _, e := range nzT {
			newA[e.i] += e.v * a[e.j]
		}
		copy(a, newA)
		// P' = T*P*T' + RR'.
		for k := range TP {
			TP[k] = 0
		}
		for _, e := range nzT {
			ti := e.i * rd
			tj := e.j * rd
			tv := e.v
			for j := 0; j < rd; j++ {
				TP[ti+j] += tv * P[tj+j]
			}
		}
		copy(newP, RRt)
		for i := 0; i < rd; i++ {
			row := i * rd
			for _, e := range nzT {
				newP[row+e.i] += TP[row+e.j] * e.v
			}
		}
		P, newP = newP, P
	}
	return -logL, innov
}

// kalmanSARIMAX runs a standard Kalman filter using statsmodels' SARIMAX
// state-space form and approximate-diffuse initialization (kappa for the
// integrated states, stationary cov for the ARMA block).
//
// Likelihood convention: drop the first kStatesDiff observations (statsmodels'
// loglikelihood_burn = number of integrated states), matching their default.
//
// Returns (negLogLik, sigma2_concentrated, innovations).
func kalmanSARIMAX(
	y []float64, d, m, D int,
	phi, theta, sPhi, sTheta []float64,
	kappa float64,
) (negLogLik, sigma2 float64, innovations []float64) {
	if kappa == 0 {
		kappa = 1e6
	}
	nzT, zRow, rCol, rd, kStatesDiff := sarimaxStateSpaceSparse(phi, theta, sPhi, sTheta, d, D, m)
	if rd == 0 {
		// pure white noise
		sse := 0.0
		for _, v := range y {
			sse += v * v
		}
		s2 := sse / float64(len(y))
		return float64(len(y)) / 2 * (math.Log(2*math.Pi*s2) + 1), s2, nil
	}
	rd2 := rd * rd

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

	P := make([]float64, rd2)
	for i := 0; i < kStatesDiff; i++ {
		P[i*rd+i] = kappa
	}
	if kArma := rd - kStatesDiff; kArma > 0 {
		fullPhi := expandSARMA(phi, sPhi, m)
		fullTheta := expandSMA(theta, sTheta, m)
		Pst, pr := stationaryCovGardner(fullPhi, fullTheta)
		for i := 0; i < pr && kStatesDiff+i < rd; i++ {
			src := i * pr
			for j := 0; j < pr && kStatesDiff+j < rd; j++ {
				P[(kStatesDiff+i)*rd+(kStatesDiff+j)] = Pst[src+j]
			}
		}
	}

	a := make([]float64, rd)
	n := len(y)
	innov := make([]float64, n)
	logF := 0.0
	sumVF := 0.0
	nu := 0
	burn := kStatesDiff

	PzT := make([]float64, rd)
	ZP := make([]float64, rd)
	K := make([]float64, rd)
	newA := make([]float64, rd)
	TP := make([]float64, rd2)
	newP := make([]float64, rd2)

	for t := 0; t < n; t++ {
		predY := 0.0
		for i := 0; i < rd; i++ {
			predY += zRow[i] * a[i]
		}
		v := y[t] - predY
		innov[t] = v

		for i := 0; i < rd; i++ {
			base := i * rd
			s := 0.0
			for j := 0; j < rd; j++ {
				s += P[base+j] * zRow[j]
			}
			PzT[i] = s
		}
		F := 0.0
		for i := 0; i < rd; i++ {
			F += zRow[i] * PzT[i]
		}
		if F <= 0 || math.IsNaN(F) {
			return math.Inf(1), 0, innov
		}
		invF := 1.0 / F
		for i := 0; i < rd; i++ {
			K[i] = PzT[i] * invF
			a[i] += K[i] * v
		}
		for j := 0; j < rd; j++ {
			s := 0.0
			for i := 0; i < rd; i++ {
				s += zRow[i] * P[i*rd+j]
			}
			ZP[j] = s
		}
		for i := 0; i < rd; i++ {
			ki := K[i]
			base := i * rd
			for j := 0; j < rd; j++ {
				P[base+j] -= ki * ZP[j]
			}
		}

		if t >= burn {
			logF += math.Log(F)
			sumVF += v * v * invF
			nu++
		}

		// Predict.
		for i := 0; i < rd; i++ {
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
			ti := e.i * rd
			tj := e.j * rd
			tv := e.v
			for j := 0; j < rd; j++ {
				TP[ti+j] += tv * P[tj+j]
			}
		}
		copy(newP, RRt)
		for i := 0; i < rd; i++ {
			row := i * rd
			for _, e := range nzT {
				newP[row+e.i] += TP[row+e.j] * e.v
			}
		}
		P, newP = newP, P
	}

	if nu == 0 || sumVF <= 0 {
		return math.Inf(1), 0, innov
	}
	s2 := sumVF / float64(nu)
	negLL := 0.5 * (float64(nu)*(math.Log(2*math.Pi*s2)+1) + logF)
	return negLL, s2, innov
}
