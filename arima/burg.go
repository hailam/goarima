package arima

import (
	"errors"
	"math"
)

// burgAR estimates AR(p) coefficients for a stationary series using
// Burg's lattice algorithm (Burg 1968). Returns phi[1..p] in goarima's
// convention:
//
//	y_t = phi[0]·y_{t-1} + phi[1]·y_{t-2} + … + phi[p-1]·y_{t-p} + e_t
//
// Burg has two practical advantages over Yule-Walker / OLS for the
// Hannan-Rissanen Stage-1 long-AR fit:
//   - No data windowing — uses all n samples efficiently. OLS truncates
//     the lag matrix and can be biased on short series.
//   - Always produces stationary AR coefficients (|reflection
//     coefficients| ≤ 1 by construction), so the warm-start lands
//     inside the invertibility region — no boundary surprises in the
//     subsequent ML refinement.
//
// Statsmodels uses Burg as the default AR estimator; R's ar() supports
// it via method="burg". This is the same algorithm.
//
// Returns an error when n ≤ p or when the lattice update would divide
// by zero (degenerate input).
func burgAR(y []float64, p int) ([]float64, error) {
	n := len(y)
	if p < 1 {
		return nil, errors.New("burgAR: p must be ≥ 1")
	}
	if p >= n {
		return nil, errors.New("burgAR: p must be < n")
	}

	// Demean to match the typical AR(p) formulation (zero-mean process).
	// The intercept is fitted separately by the surrounding code.
	mean := 0.0
	for _, v := range y {
		mean += v
	}
	mean /= float64(n)
	yc := make([]float64, n)
	for i, v := range y {
		yc[i] = v - mean
	}

	// Forward and backward errors, initialised to the demeaned series.
	// f[t] = forward prediction error at time t, current order
	// b[t] = backward prediction error at time t, current order
	// At order 0 both equal yc[t].
	f := append([]float64{}, yc...)
	b := append([]float64{}, yc...)

	// AR coefficients in Burg's convention: y_t + Σ a[i]·y_{t-i} = e_t.
	// We negate at the end to convert to goarima's y_t = Σ phi[i]·y_{t-i} + e_t.
	a := make([]float64, p+1)
	a[0] = 1
	aPrev := make([]float64, p+1)

	for k := 1; k <= p; k++ {
		num := 0.0
		den := 0.0
		for t := k; t < n; t++ {
			num += f[t] * b[t-1]
			den += f[t]*f[t] + b[t-1]*b[t-1]
		}
		if den == 0 {
			return nil, errors.New("burgAR: degenerate input — lattice denominator zero")
		}
		kRefl := -2 * num / den
		// kRefl is the partial autocorrelation at lag k. By construction
		// |kRefl| ≤ 1 for any input — guarantees stationarity.
		if math.Abs(kRefl) > 1 {
			// Numerical drift safety; clip to interior.
			if kRefl > 1 {
				kRefl = 1 - 1e-12
			} else {
				kRefl = -1 + 1e-12
			}
		}

		// Levinson-Durbin AR-coefficient update.
		copy(aPrev, a)
		a[k] = kRefl
		for i := 1; i < k; i++ {
			a[i] = aPrev[i] + kRefl*aPrev[k-i]
		}

		// In-place forward / backward error update.
		// We iterate t from n-1 DOWN to k so that writes to f[t]/b[t]
		// don't clobber f[t-1]/b[t-2] which are read at iteration t-1.
		// (b[t-1] is read at this iteration; the write goes to b[t],
		// effectively shifting the backward-error array by one.)
		for t := n - 1; t >= k; t-- {
			ft := f[t]
			btm1 := b[t-1]
			f[t] = ft + kRefl*btm1
			b[t] = btm1 + kRefl*ft
		}
	}

	// Convert Burg's a (with `+ Σ a y` convention) to goarima's phi
	// (with `Σ phi y_{t-i}` convention): phi[i-1] = -a[i].
	phi := make([]float64, p)
	for i := 1; i <= p; i++ {
		phi[i-1] = -a[i]
	}
	return phi, nil
}
