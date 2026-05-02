package arima

import (
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/mat"
)

// ARRoots returns the roots of the AR(p) characteristic polynomial
// 1 - phi_1 z - phi_2 z^2 - ... - phi_p z^p (in z, not B).
//
// Returns an empty slice for pure MA models. Mirrors statsmodels' arroots.
func (m *ARIMA) ARRoots() []complex128 {
	full := expandSARMA(m.phi, m.Phi, m.Seasonal.M)
	if len(full) == 0 {
		return nil
	}
	// Polynomial: 1 + sum_{i=1..p} -phi_i * z^i  (i.e. coeffs [1, -phi_1, ..., -phi_p])
	coeffs := make([]float64, len(full)+1)
	coeffs[0] = 1
	for i, v := range full {
		coeffs[i+1] = -v
	}
	return polyRoots(coeffs)
}

// MARoots returns the roots of the MA(q) characteristic polynomial
// 1 + theta_1 z + theta_2 z^2 + ... + theta_q z^q.
func (m *ARIMA) MARoots() []complex128 {
	full := expandSMA(m.theta, m.Theta, m.Seasonal.M)
	if len(full) == 0 {
		return nil
	}
	coeffs := make([]float64, len(full)+1)
	coeffs[0] = 1
	for i, v := range full {
		coeffs[i+1] = v
	}
	return polyRoots(coeffs)
}

// IsStationary returns true iff every AR root lies strictly outside the unit circle.
func (m *ARIMA) IsStationary() bool {
	roots := m.ARRoots()
	if len(roots) == 0 {
		return true
	}
	for _, r := range roots {
		if cmplx.Abs(r) <= 1 {
			return false
		}
	}
	return true
}

// IsInvertible returns true iff every MA root lies strictly outside the unit circle.
func (m *ARIMA) IsInvertible() bool {
	roots := m.MARoots()
	if len(roots) == 0 {
		return true
	}
	for _, r := range roots {
		if cmplx.Abs(r) <= 1 {
			return false
		}
	}
	return true
}

// polyRoots returns the roots of the polynomial whose coefficients (lowest
// degree first) are given. Uses the eigenvalues of the companion matrix.
func polyRoots(coeffs []float64) []complex128 {
	// strip trailing zeros
	for len(coeffs) > 1 && coeffs[len(coeffs)-1] == 0 {
		coeffs = coeffs[:len(coeffs)-1]
	}
	deg := len(coeffs) - 1
	if deg <= 0 {
		return nil
	}
	// Normalize so leading coeff = 1 (monic).
	lead := coeffs[deg]
	if lead == 0 {
		return nil
	}
	mon := make([]float64, deg+1)
	for i, v := range coeffs {
		mon[i] = v / lead
	}
	// Companion matrix C (deg x deg):
	//   row i: 0..0 1 0..0  for i = 0..deg-2 (1 in column i+1)
	//   last row: -mon[0], -mon[1], ..., -mon[deg-1]
	C := mat.NewDense(deg, deg, nil)
	for i := 0; i < deg-1; i++ {
		C.Set(i, i+1, 1)
	}
	for j := 0; j < deg; j++ {
		C.Set(deg-1, j, -mon[j])
	}
	var eig mat.Eigen
	if !eig.Factorize(C, mat.EigenRight) {
		return nil
	}
	values := eig.Values(nil)
	out := make([]complex128, len(values))
	copy(out, values)
	return out
}

// MinRootAbs returns the smallest |root| across both AR and MA polynomials.
// Useful for nearness-to-non-stationarity checks. Returns +Inf if no roots.
func (m *ARIMA) MinRootAbs() float64 {
	min := math.Inf(1)
	for _, r := range m.ARRoots() {
		if a := cmplx.Abs(r); a < min {
			min = a
		}
	}
	for _, r := range m.MARoots() {
		if a := cmplx.Abs(r); a < min {
			min = a
		}
	}
	return min
}
