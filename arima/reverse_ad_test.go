package arima

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestReverseAD_VsNumerical validates kalmanARMALikelihoodGradAD's
// gradient against central-difference at 1e-5 tolerance across a range
// of ARMA shapes.
//
// We pass a FIXED P_0 to both AD and numerical perturbations so the
// numerical baseline doesn't include ∂P_0/∂(phi, theta) — which AD
// intentionally skips (P_0 treated as constant). With a shared
// constant P_0 the two should match to floating-point precision.
func TestReverseAD_VsNumerical(t *testing.T) {
	cases := []struct {
		label string
		phi   []float64
		theta []float64
		n     int
	}{
		{"AR(1)", []float64{0.5}, []float64{}, 200},
		{"MA(1)", []float64{}, []float64{0.4}, 200},
		{"ARMA(1,1)", []float64{0.5}, []float64{-0.3}, 200},
		{"ARMA(2,2)", []float64{0.4, 0.2}, []float64{-0.3, 0.1}, 500},
		{"ARMA(2,5)", []float64{0.4, 0.2}, []float64{-0.3, 0.2, -0.1, 0.05, -0.02}, 500},
		{"ARMA(3,3)", []float64{0.3, 0.2, 0.1}, []float64{-0.2, 0.15, -0.05}, 700},
	}
	for _, c := range cases {
		t.Run(c.label, func(t *testing.T) {
			y := arGenerate(c.n, c.phi, c.theta, 42)

			// Compute a fixed P_0 once (constant for AD purposes).
			var gw gardnerWorkspace
			P0src, _ := stationaryCovGardnerInto(&gw, c.phi, c.theta)
			P0 := append([]float64(nil), P0src...) // detach from workspace

			var ws kalmanWorkspace
			_, _, gpAD, gtAD, ok := kalmanARMALikelihoodGradAD(y, c.phi, c.theta, P0, &ws)
			if !ok {
				t.Fatalf("AD failed on %s", c.label)
			}
			gpNum, gtNum := numericalGradARMA(y, c.phi, c.theta, P0)
			maxDiff := 0.0
			for i := range gpAD {
				d := math.Abs(gpAD[i] - gpNum[i])
				if d > maxDiff {
					maxDiff = d
				}
			}
			for j := range gtAD {
				d := math.Abs(gtAD[j] - gtNum[j])
				if d > maxDiff {
					maxDiff = d
				}
			}
			if maxDiff > 1e-5 {
				t.Errorf("%s: maxDiff(AD vs num) = %.3e (want < 1e-5)", c.label, maxDiff)
				t.Logf("  gpAD=%v\n  gpNum=%v\n  gtAD=%v\n  gtNum=%v",
					gpAD, gpNum, gtAD, gtNum)
			}
		})
	}
}

// arGenerate produces a synthetic ARMA(p,q) series for testing.
func arGenerate(n int, phi, theta []float64, seed uint64) []float64 {
	rng := rand.New(rand.NewPCG(seed, seed^0x9E))
	q := len(theta)
	res := make([]float64, n)
	y := make([]float64, n)
	for t := 0; t < n; t++ {
		e := rng.NormFloat64()
		var s float64
		for i := 0; i < len(phi) && t-1-i >= 0; i++ {
			s += phi[i] * y[t-1-i]
		}
		for j := 0; j < q && t-1-j >= 0; j++ {
			s += theta[j] * res[t-1-j]
		}
		y[t] = s + e
		res[t] = e
	}
	return y
}

// numericalGradARMA computes ∂negLL/∂phi, ∂negLL/∂theta via central
// difference, holding P_0 fixed (matches AD's constant-P_0 assumption).
func numericalGradARMA(y, phi, theta, P0 []float64) (gradPhi, gradTheta []float64) {
	const eps = 1e-7
	p := len(phi)
	q := len(theta)
	gradPhi = make([]float64, p)
	gradTheta = make([]float64, q)
	llCall := func(phi, theta []float64) float64 {
		var ws kalmanWorkspace
		ll, _, _, _, _ := kalmanARMALikelihoodGradAD(y, phi, theta, P0, &ws)
		return ll
	}
	for i := 0; i < p; i++ {
		phiP := append([]float64{}, phi...)
		phiP[i] += eps
		fp := llCall(phiP, theta)
		phiM := append([]float64{}, phi...)
		phiM[i] -= eps
		fm := llCall(phiM, theta)
		gradPhi[i] = (fp - fm) / (2 * eps)
	}
	for j := 0; j < q; j++ {
		thetaP := append([]float64{}, theta...)
		thetaP[j] += eps
		fp := llCall(phi, thetaP)
		thetaM := append([]float64{}, theta...)
		thetaM[j] -= eps
		fm := llCall(phi, thetaM)
		gradTheta[j] = (fp - fm) / (2 * eps)
	}
	return
}

// TestReverseAD_NegLLMatchesJoseph: verifies that the rank-1 form
// kalmanARMALikelihoodGradAD's negLL is close to the production
// Joseph-form kalmanARMALikelihoodInto's negLL on canonical fits.
// They should agree to ~1e-6 in exact arithmetic; small floating-point
// drift is expected near unit-circle.
func TestReverseAD_NegLLMatchesJoseph(t *testing.T) {
	cases := []struct {
		label string
		phi   []float64
		theta []float64
		n     int
	}{
		{"AR(1)", []float64{0.3}, []float64{}, 200},
		{"ARMA(1,1)", []float64{0.4}, []float64{-0.2}, 200},
		{"ARMA(2,2)", []float64{0.3, 0.15}, []float64{-0.2, 0.1}, 500},
	}
	for _, c := range cases {
		t.Run(c.label, func(t *testing.T) {
			y := arGenerate(c.n, c.phi, c.theta, 17)
			var wsAD kalmanWorkspace
			llAD, s2AD, _, _, ok := kalmanARMALikelihoodGradAD(y, c.phi, c.theta, nil, &wsAD)
			if !ok {
				t.Fatalf("AD failed")
			}
			var wsJ kalmanWorkspace
			llJ, s2J := kalmanARMALikelihoodInto(y, c.phi, c.theta, &wsJ)
			if math.Abs(llAD-llJ) > 1e-4 {
				t.Errorf("negLL mismatch: AD=%g Joseph=%g (Δ=%g)", llAD, llJ, llAD-llJ)
			}
			if math.Abs(s2AD-s2J) > 1e-6 {
				t.Errorf("sigma² mismatch: AD=%g Joseph=%g (Δ=%g)", s2AD, s2J, s2AD-s2J)
			}
		})
	}
}
