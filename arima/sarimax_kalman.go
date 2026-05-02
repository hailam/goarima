package arima

import (
	"math"

	"gonum.org/v1/gonum/mat"
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
	T2D, zRow, rCol, kStatesDiff := sarimaxStateSpace(phi, theta, sPhi, sTheta, d, D, m)
	rd := len(zRow)
	if rd == 0 || sigma2 <= 0 {
		return math.Inf(1), nil
	}
	Tflat := make([]float64, rd*rd)
	for i := 0; i < rd; i++ {
		for j := 0; j < rd; j++ {
			Tflat[i*rd+j] = T2D[i][j]
		}
	}
	T := mat.NewDense(rd, rd, Tflat)

	// Q = sigma2 * R*R'
	RRtFlat := make([]float64, rd*rd)
	for i := 0; i < rd; i++ {
		for j := 0; j < rd; j++ {
			RRtFlat[i*rd+j] = sigma2 * rCol[i] * rCol[j]
		}
	}
	RRt := mat.NewDense(rd, rd, RRtFlat)

	// Initial covariance: kappa on integrated diagonal, sigma2*stationary on ARMA.
	P := mat.NewDense(rd, rd, nil)
	for i := 0; i < kStatesDiff; i++ {
		P.Set(i, i, kappa)
	}
	if kArma := rd - kStatesDiff; kArma > 0 {
		fullPhi := expandSARMA(phi, sPhi, m)
		fullTheta := expandSMA(theta, sTheta, m)
		Pst := stationaryCovGardner(fullPhi, fullTheta)
		pr, _ := Pst.Dims()
		for i := 0; i < pr && kStatesDiff+i < rd; i++ {
			for j := 0; j < pr && kStatesDiff+j < rd; j++ {
				P.Set(kStatesDiff+i, kStatesDiff+j, sigma2*Pst.At(i, j))
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
	ZP := make([]float64, rd)
	newA := make([]float64, rd)

	for t := 0; t < n; t++ {
		predY := 0.0
		for i, zi := range zRow {
			predY += zi * a[i]
		}
		v := y[t] - predY
		innov[t] = v
		for i := 0; i < rd; i++ {
			s := 0.0
			for j := 0; j < rd; j++ {
				s += P.At(i, j) * zRow[j]
			}
			PzT[i] = s
		}
		F := 0.0
		for i, zi := range zRow {
			F += zi * PzT[i]
		}
		if F <= 0 || math.IsNaN(F) {
			return math.Inf(1), innov
		}
		invF := 1.0 / F
		for i := 0; i < rd; i++ {
			K[i] = PzT[i] * invF
		}
		for i := 0; i < rd; i++ {
			a[i] += K[i] * v
		}
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
		if t >= burn {
			logL -= 0.5 * (math.Log(2*math.Pi*F) + v*v*invF)
		}
		for i := 0; i < rd; i++ {
			s := 0.0
			for j := 0; j < rd; j++ {
				s += T.At(i, j) * a[j]
			}
			newA[i] = s
		}
		copy(a, newA)
		var TP mat.Dense
		TP.Mul(T, P)
		var TPTt mat.Dense
		TPTt.Mul(&TP, T.T())
		var newP mat.Dense
		newP.Add(&TPTt, RRt)
		P = mat.DenseCopyOf(&newP)
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
	T2D, zRow, rCol, kStatesDiff := sarimaxStateSpace(phi, theta, sPhi, sTheta, d, D, m)
	rd := len(zRow)
	if rd == 0 {
		// pure white noise
		sse := 0.0
		for _, v := range y {
			sse += v * v
		}
		s2 := sse / float64(len(y))
		return float64(len(y)) / 2 * (math.Log(2*math.Pi*s2) + 1), s2, nil
	}

	// Convert T to mat.Dense for predict-step matrix products.
	Tflat := make([]float64, rd*rd)
	for i := 0; i < rd; i++ {
		for j := 0; j < rd; j++ {
			Tflat[i*rd+j] = T2D[i][j]
		}
	}
	T := mat.NewDense(rd, rd, Tflat)

	// RR^T (selection outer product)
	RRtFlat := make([]float64, rd*rd)
	for i := 0; i < rd; i++ {
		for j := 0; j < rd; j++ {
			RRtFlat[i*rd+j] = rCol[i] * rCol[j]
		}
	}
	RRt := mat.NewDense(rd, rd, RRtFlat)

	// Initial covariance:
	//   integrated states (positions [0, kStatesDiff)): diagonal kappa
	//   ARMA block       (positions [kStatesDiff, rd)): stationary via Gardner getQ0
	P := mat.NewDense(rd, rd, nil)
	for i := 0; i < kStatesDiff; i++ {
		P.Set(i, i, kappa)
	}
	if kArma := rd - kStatesDiff; kArma > 0 {
		// Compute stationary cov of the ARMA companion block.
		// We need the AR/MA coefficients in the form expandSARMA / expandSMA produces.
		fullPhi := expandSARMA(phi, sPhi, m)
		fullTheta := expandSMA(theta, sTheta, m)
		// statsmodels initialization for the ARMA block: stationary distribution.
		Pst := stationaryCovGardner(fullPhi, fullTheta)
		pr, _ := Pst.Dims()
		// The Pst matrix is the standard Hamilton stationary form for ARMA.
		// statsmodels' ARMA companion uses the SAME shape; copy directly.
		for i := 0; i < pr && kStatesDiff+i < rd; i++ {
			for j := 0; j < pr && kStatesDiff+j < rd; j++ {
				P.Set(kStatesDiff+i, kStatesDiff+j, Pst.At(i, j))
			}
		}
	}

	a := make([]float64, rd) // a_{t|t-1}
	n := len(y)
	innov := make([]float64, n)
	logF := 0.0
	sumVF := 0.0
	nu := 0
	burn := kStatesDiff // drop first kStatesDiff observations from likelihood

	PzT := make([]float64, rd)
	K := make([]float64, rd)
	ZP := make([]float64, rd)
	newA := make([]float64, rd)

	for t := 0; t < n; t++ {
		// v = y - Z*a
		predY := 0.0
		for i, zi := range zRow {
			predY += zi * a[i]
		}
		v := y[t] - predY
		innov[t] = v

		// PzT = P * Z'
		for i := 0; i < rd; i++ {
			s := 0.0
			for j := 0; j < rd; j++ {
				s += P.At(i, j) * zRow[j]
			}
			PzT[i] = s
		}
		// F = Z * PzT
		F := 0.0
		for i, zi := range zRow {
			F += zi * PzT[i]
		}
		if F <= 0 || math.IsNaN(F) {
			return math.Inf(1), 0, innov
		}
		invF := 1.0 / F

		// K = PzT / F
		for i := 0; i < rd; i++ {
			K[i] = PzT[i] * invF
		}
		// a_{t|t} = a + K * v
		for i := 0; i < rd; i++ {
			a[i] += K[i] * v
		}
		// ZP = Z * P
		for j := 0; j < rd; j++ {
			s := 0.0
			for i := 0; i < rd; i++ {
				s += zRow[i] * P.At(i, j)
			}
			ZP[j] = s
		}
		// P -= K * ZP
		for i := 0; i < rd; i++ {
			ki := K[i]
			for j := 0; j < rd; j++ {
				P.Set(i, j, P.At(i, j)-ki*ZP[j])
			}
		}

		if t >= burn {
			logF += math.Log(F)
			sumVF += v * v * invF
			nu++
		}

		// Predict: a' = T*a, P' = T P T' + RR'
		for i := 0; i < rd; i++ {
			s := 0.0
			for j := 0; j < rd; j++ {
				s += T.At(i, j) * a[j]
			}
			newA[i] = s
		}
		copy(a, newA)
		var TP mat.Dense
		TP.Mul(T, P)
		var TPTt mat.Dense
		TPTt.Mul(&TP, T.T())
		var newP mat.Dense
		newP.Add(&TPTt, RRt)
		P = mat.DenseCopyOf(&newP)
	}

	if nu == 0 || sumVF <= 0 {
		return math.Inf(1), 0, innov
	}
	s2 := sumVF / float64(nu)
	negLL := 0.5 * (float64(nu)*(math.Log(2*math.Pi*s2)+1) + logF)
	return negLL, s2, innov
}
