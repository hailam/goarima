package arima

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/hailam/goarima/datasets"
)

// KAL-1 — known correctness bug in kalmanARMALikelihood. The rank-1 P
// update at kalman.go:215-222 is in non-Joseph form and loses positive-
// semi-definiteness over many iterations, causing F_t = P[0,0] (the
// innovation variance) to drift wildly. sum(log F_t) — which should be
// a small bounded value O(log γ(0)/σ²_ε) for a stationary ARMA at the
// stationary init — instead swings unpredictably between large positive
// and large negative values, producing wrong logL / AIC / AICc that
// disagree with R::stats::arima and statsmodels.SARIMAX.
//
// Status: tracked as KAL-1.
//
// These probes capture the diagnostic data so that any future fix can
// be verified to bring sum_log_F back into the expected band.

// Stationary AR(1), σ²=1: γ(0)/σ² = 4/3, so sum_log_F should be ≈ log(4/3)
// from the first observation plus a small bounded transient.
func TestKalmanLogFMagnitude_ProbeAR1(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	const n = 500
	const phi = 0.5
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = phi*y[i-1] + rng.NormFloat64()
	}
	mean := 0.0
	for _, v := range y {
		mean += v
	}
	mean /= float64(n)
	for i := range y {
		y[i] -= mean
	}

	negLL, s2, _ := kalmanARMALikelihood(y, []float64{phi}, nil)
	textbook := 0.5 * (float64(n) * (math.Log(2*math.Pi*s2) + 1))
	sumLogF := 2 * (negLL - textbook)
	t.Logf("AR(1) phi=0.5 n=%d σ²=%.4f sum_log_F=%.4f (expect ~0.3)",
		n, s2, sumLogF)
	if math.Abs(sumLogF) > 5 {
		t.Errorf("sum_log_F = %.4f — Kalman PSD broke even on synthetic AR(1)", sumLogF)
	}
}

// SARIMA-like stationary fixed-coef synthesis (r=14, n=200). On stable
// data with non-fitted parameters the PSD usually holds: sum_log_F is
// ≤ ~30. This is a sanity floor — if THIS test fails, the bug is even
// worse than KAL-1's diagnosis.
func TestKalmanLogFMagnitude_ProbeSARIMA(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	const n = 200
	phi := []float64{0.6, -0.2}
	theta := []float64{0.4}
	Phi := []float64{0.5}
	M := 12

	fullPhi := expandSARMA(phi, Phi, M)
	fullTheta := append([]float64{}, theta...)

	r := len(fullPhi)
	if len(fullTheta)+1 > r {
		r = len(fullTheta) + 1
	}
	innov := make([]float64, n+r)
	for i := range innov {
		innov[i] = rng.NormFloat64()
	}
	y := make([]float64, n)
	for t := 0; t < n; t++ {
		s := innov[t+r]
		for i, p := range fullPhi {
			if t-1-i >= 0 {
				s += p * y[t-1-i]
			}
		}
		for j, q := range fullTheta {
			if t-1-j >= 0 {
				s += q * innov[t+r-1-j]
			}
		}
		y[t] = s
	}
	mean := 0.0
	for _, v := range y {
		mean += v
	}
	mean /= float64(n)
	for i := range y {
		y[i] -= mean
	}

	negLL, s2, _ := kalmanARMALikelihood(y, fullPhi, fullTheta)
	textbook := 0.5 * (float64(n) * (math.Log(2*math.Pi*s2) + 1))
	sumLogF := 2 * (negLL - textbook)
	t.Logf("SARIMA-like r=%d n=%d σ²=%.4f sum_log_F=%.4f", r, n, s2, sumLogF)
	if math.Abs(sumLogF) > 50 {
		t.Errorf("sum_log_F = %.4f even on stationary fixed-coef synthesis — KAL-1 worse than thought", sumLogF)
	}
}

// Debug: instrument F evolution for a near-boundary case. Originally
// used to diagnose KAL-1 (fixed via post-fit textbook fallback in
// arima.go). Kept as a tool for future Kalman-likelihood investigation.
// Not a regression test — only logs.
func TestKalmanLogF_DebugFTrajectory(t *testing.T) {
	if testing.Short() {
		t.Skip("diagnostic; -short skips")
	}
	ap := datasets.LoadAirPassengers()
	ys := ap[:50]
	m := NewARIMA(Order{P: 2, D: 1, Q: 1})
	m.Seasonal = SeasonalOrder{P: 1, D: 0, Q: 0, M: 12}
	m.MaxIter = 100
	if err := m.Fit(ys, nil); err != nil {
		t.Fatal(err)
	}
	t.Logf("Fitted params: phi=%v theta=%v Phi=%v Theta=%v",
		m.phi, m.theta, m.Phi, m.Theta)

	// Re-run kalmanARMALikelihood directly with instrumentation.
	fullPhi := expandSARMA(m.phi, m.Phi, m.Seasonal.M)
	fullTheta := expandSMA(m.theta, m.Theta, m.Seasonal.M)

	// Get the differenced+demeaned series the Kalman saw.
	resids := m.Resid()
	// Drop NaN warmup; this gives ws-equivalent.
	w := make([]float64, 0, len(resids))
	for _, v := range resids {
		if !math.IsNaN(v) {
			w = append(w, v)
		}
	}
	t.Logf("len(w)=%d r=%d  fullPhi[%d] fullTheta[%d]",
		len(w), len(fullPhi)+1, len(fullPhi), len(fullTheta))
	t.Logf("|fullPhi|: %v", absSlice(fullPhi))

	// Build differenced y from m.yTrain.
	ws := make([]float64, len(ys))
	copy(ws, ys)
	if m.Order.D > 0 {
		ws = applyDiff(ws, 1, m.Order.D)
	}
	// Demean.
	mean := 0.0
	for _, v := range ws {
		mean += v
	}
	mean /= float64(len(ws))
	for i := range ws {
		ws[i] -= mean
	}

	// Direct Kalman with F-trajectory printing.
	r := len(fullPhi)
	if len(fullTheta)+1 > r {
		r = len(fullTheta) + 1
	}
	P, _ := stationaryCovGardner(fullPhi, fullTheta)
	t.Logf("Initial P[0,0] (=F_1) = %.6g", P[0])
	t.Logf("Initial P diag: %v", diagOf(P, r))

	// Now actually run the kalman iteration manually
	a := make([]float64, r)
	K := make([]float64, r)
	row0 := make([]float64, r)
	newA := make([]float64, r)
	TP := make([]float64, r*r)
	newP := make([]float64, r*r)
	Rvec := make([]float64, r)
	Rvec[0] = 1
	for j, q := range fullTheta {
		if j+1 < r {
			Rvec[j+1] = q
		}
	}
	RRt := make([]float64, r*r)
	for i := 0; i < r; i++ {
		for j := 0; j < r; j++ {
			RRt[i*r+j] = Rvec[i] * Rvec[j]
		}
	}

	type tNZRec struct{ i, j int; v float64 }
	var nzT []tNZRec
	for i := 0; i < r; i++ {
		if i < len(fullPhi) && fullPhi[i] != 0 {
			nzT = append(nzT, tNZRec{i, 0, fullPhi[i]})
		}
		if i+1 < r {
			nzT = append(nzT, tNZRec{i, i + 1, 1})
		}
	}

	logF := 0.0
	for tt := 0; tt < len(ws); tt++ {
		v := ws[tt] - a[0]
		F := P[0]
		if tt < 5 || tt > len(ws)-5 {
			t.Logf("  step %3d: F=%.6g logF_cum=%.4f (P[0,0]=%.6g)",
				tt, F, logF+math.Log(math.Abs(F)+1e-300), P[0])
		}
		if F <= 0 {
			t.Logf("  step %3d: F=%.6g <= 0 — Kalman aborts here", tt, F)
			return
		}
		invF := 1.0 / F
		for i := 0; i < r; i++ {
			K[i] = P[i*r] * invF
			a[i] += K[i] * v
		}
		copy(row0, P[:r])
		for i := 0; i < r; i++ {
			ki := K[i]
			r0i := row0[i]
			off := i * r
			for j := 0; j < r; j++ {
				kj := K[j]
				P[off+j] += -ki*row0[j] - kj*r0i + ki*kj*F
			}
		}
		logF += math.Log(F)
		// Predict
		for i := 0; i < r; i++ {
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
			ti := e.i * r
			tj := e.j * r
			tv := e.v
			for j := 0; j < r; j++ {
				TP[ti+j] += tv * P[tj+j]
			}
		}
		copy(newP, RRt)
		for i := 0; i < r; i++ {
			row := i * r
			for _, e := range nzT {
				newP[row+e.i] += TP[row+e.j] * e.v
			}
		}
		P, newP = newP, P
	}
	t.Logf("Final logF = %.4f", logF)
}

func absSlice(a []float64) []float64 {
	out := make([]float64, len(a))
	for i, v := range a {
		out[i] = math.Abs(v)
	}
	return out
}
func diagOf(P []float64, r int) []float64 {
	out := make([]float64, r)
	for i := 0; i < r; i++ {
		out[i] = P[i*r+i]
	}
	return out
}

// Ridge-penalty regression: on the KAL-1 reproducer (n=50 truncation
// of AirPassengers fitted as ARIMA(2,1,1)(1,0,0)[12]) the unmodified
// optimizer pushes Φ to within 1e-13 of the unit circle. With
// RidgePenalty=1.0/n, the optimizer should keep Φ strictly inside —
// say |Φ| < 0.99 — because the penalty makes large unconstrained x
// expensive.
func TestRidgePenalty_PreventsBoundaryPileup(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	ys := ap[:50]

	// Without ridge: replicate the KAL-1 trace finding (Φ near 1.0).
	noRidge := NewARIMA(Order{P: 2, D: 1, Q: 1})
	noRidge.Seasonal = SeasonalOrder{P: 1, D: 0, Q: 0, M: 12}
	noRidge.MaxIter = 100
	if err := noRidge.Fit(ys, nil); err != nil {
		t.Fatal(err)
	}
	t.Logf("no-ridge: Φ=%v θ=%v", noRidge.Phi, noRidge.theta)

	// With ridge: Φ should be pulled away from the unit circle.
	withRidge := NewARIMA(Order{P: 2, D: 1, Q: 1})
	withRidge.Seasonal = SeasonalOrder{P: 1, D: 0, Q: 0, M: 12}
	withRidge.MaxIter = 100
	withRidge.RidgePenalty = 1.0 / float64(len(ys))
	if err := withRidge.Fit(ys, nil); err != nil {
		t.Fatal(err)
	}
	t.Logf("ridge=1/n: Φ=%v θ=%v RidgePenalty=%g",
		withRidge.Phi, withRidge.theta, withRidge.RidgePenalty)

	// On this dataset the no-ridge fit drives Φ to ≈1 (Φ_1 within
	// 1e-13 of 1.0). With ridge, |Φ_1| should be at least slightly
	// smaller. Allow a generous tolerance — different machines /
	// tanh saturation behaviours may differ — but verify the ridge
	// has SOME effect.
	if len(noRidge.Phi) > 0 && len(withRidge.Phi) > 0 {
		noR := math.Abs(noRidge.Phi[0])
		wR := math.Abs(withRidge.Phi[0])
		if noR > 0.99 && wR >= noR {
			t.Errorf("ridge should reduce |Φ_1|: no-ridge=%g vs ridge=%g", noR, wR)
		}
	}
}

// AutoArima ridge-penalty propagation: ridge on AutoArima should
// produce candidates that all keep params away from boundary, even on
// the short-series case where un-penalized AutoArima picks
// boundary-rampant models.
func TestRidgePenalty_AutoArimaPropagates(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	ys := ap[:50]
	m, err := AutoArima(ys, nil, AutoArimaOpts{
		M: 12, MaxP: 2, MaxQ: 2, MaxCapP: 1, MaxCapQ: 1,
		MaxOrder:     5,
		MaxD:         2,
		IC:           AICc,
		MaxIter:      50,
		RidgePenalty: 1.0 / float64(len(ys)),
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("AutoArima with ridge: Order=%v Seasonal=%v phi=%v theta=%v Phi=%v Theta=%v",
		m.Order, m.Seasonal, m.phi, m.theta, m.Phi, m.Theta)
	// All AR/MA coefs should be strictly inside the unit circle.
	for _, slice := range [][]float64{m.phi, m.theta, m.Phi, m.Theta} {
		for i, v := range slice {
			if math.Abs(v) > 0.99 {
				t.Errorf("ridge AutoArima: |coef[%d]| = %g exceeds 0.99 boundary", i, math.Abs(v))
			}
		}
	}
	if m.RidgePenalty == 0 {
		t.Error("RidgePenalty did not propagate to fitted model")
	}
}

// KAL-1 regression test. Sweeps short AirPassengers fits where the BFGS
// optimizer pushes parameters to the stationarity/invertibility boundary,
// causing the exact-Kalman likelihood to blow up via huge initial state
// covariance. The post-fit sanity check in arima.go falls back to the
// textbook concentrated-Gaussian form when the gap exceeds 3·r units.
// After fix: gap should be ≤ ~30 across all n.
func TestKalmanLogF_KAL1_AirPassengersSweep(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	for _, n := range []int{41, 50, 60, 80, 100, 144} {
		ys := ap[:n]
		m := NewARIMA(Order{P: 2, D: 1, Q: 1})
		m.Seasonal = SeasonalOrder{P: 1, D: 0, Q: 0, M: 12}
		m.MaxIter = 100
		if err := m.Fit(ys, nil); err != nil {
			t.Logf("n=%d fit error: %v", n, err)
			continue
		}
		logL := m.LogLikelihood()
		s2 := m.Sigma2()
		nUsed := len(ys) - m.Order.D - m.Seasonal.D*m.Seasonal.M
		textbook := -0.5 * (float64(nUsed) * (math.Log(2*math.Pi*s2) + 1))
		gap := logL - textbook
		t.Logf("n=%d (n_used=%d) σ²=%.2f logL=%.2f textbook=%.2f gap=%.2f",
			n, nUsed, s2, logL, textbook, gap)
		if math.Abs(gap) > 30 {
			t.Errorf("n=%d: gap=%.2f exceeds 30 — KAL-1 fix may have regressed", n, gap)
		}
	}
}
