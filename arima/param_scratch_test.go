package arima

import (
	"math/rand/v2"
	"testing"
)

// G-NEW-2: verify the buffer-reuse variants produce bit-identical output
// to the originals across a range of input sizes and random seeds.
func TestParamScratch_Equivalence(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	scratch := acquireParamScratch()
	defer releaseParamScratch(scratch)

	for trial := 0; trial < 50; trial++ {
		// Random shape: p, q, P, Q ∈ [0, 4]; m ∈ {1, 4, 12, 24}.
		p := rng.IntN(5)
		q := rng.IntN(5)
		P := rng.IntN(3)
		Q := rng.IntN(3)
		Pmix := []int{1, 4, 12, 24}
		m := Pmix[rng.IntN(len(Pmix))]

		params := make([]float64, p+q+P+Q+1)
		for i := range params {
			params[i] = rng.NormFloat64()
		}

		// Compare arTransparams vs *Into.
		if p > 0 {
			want := arTransparams(params[:p])
			tmp := make([]float64, p)
			got := arTransparamsInto(make([]float64, p), tmp, params[:p])
			for i := range want {
				if got[i] != want[i] {
					t.Errorf("arTransparams trial %d idx %d: got %g want %g", trial, i, got[i], want[i])
				}
			}
		}
		// Compare maTransparams vs *Into.
		if q > 0 {
			want := maTransparams(params[p : p+q])
			tmp := make([]float64, q)
			got := maTransparamsInto(make([]float64, q), tmp, params[p:p+q])
			for i := range want {
				if got[i] != want[i] {
					t.Errorf("maTransparams trial %d idx %d: got %g want %g", trial, i, got[i], want[i])
				}
			}
		}

		// Compare expandSARMA vs *Into.
		phi := arTransparams(params[:p])
		Phi := arTransparams(params[p+q : p+q+P])
		want := expandSARMA(phi, Phi, m)
		got := expandSARMAInto(scratch, phi, Phi, m)
		if len(got) != len(want) {
			t.Errorf("expandSARMA trial %d len: got %d want %d", trial, len(got), len(want))
		} else {
			for i := range want {
				if got[i] != want[i] {
					t.Errorf("expandSARMA trial %d idx %d: got %g want %g", trial, i, got[i], want[i])
				}
			}
		}

		// Compare expandSMA vs *Into.
		theta := maTransparams(params[p : p+q])
		Theta := maTransparams(params[p+q+P : p+q+P+Q])
		wantT := expandSMA(theta, Theta, m)
		gotT := expandSMAInto(scratch, theta, Theta, m)
		if len(gotT) != len(wantT) {
			t.Errorf("expandSMA trial %d len: got %d want %d", trial, len(gotT), len(wantT))
		} else {
			for i := range wantT {
				if gotT[i] != wantT[i] {
					t.Errorf("expandSMA trial %d idx %d: got %g want %g", trial, i, gotT[i], wantT[i])
				}
			}
		}
	}
}

// KAL-WORKSPACE: kalmanARMALikelihoodInto must produce bit-identical
// (negLL, sigma²) to kalmanARMALikelihood across a range of input
// shapes and reused workspaces.
func TestKalmanARMALikelihoodInto_Equivalence(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 32))
	scratch := acquireParamScratch()
	defer releaseParamScratch(scratch)

	for trial := 0; trial < 20; trial++ {
		// Random ARMA shape: p, q in 0..4, n in 50..200.
		p := rng.IntN(5)
		q := rng.IntN(5)
		if p+q == 0 {
			p = 1
		}
		n := 50 + rng.IntN(150)

		// Synthesize a stationary ARMA series.
		phi := make([]float64, p)
		for i := range phi {
			phi[i] = 0.1 + 0.1*rng.Float64() // small AR coefs to stay stationary
		}
		theta := make([]float64, q)
		for i := range theta {
			theta[i] = 0.1 + 0.2*rng.Float64()
		}
		y := make([]float64, n)
		for i := 1; i < n; i++ {
			ar := 0.0
			for j, ph := range phi {
				if i-1-j >= 0 {
					ar += ph * y[i-1-j]
				}
			}
			y[i] = ar + rng.NormFloat64()
		}

		wantNeg, wantSig, _ := kalmanARMALikelihood(y, phi, theta)
		gotNeg, gotSig := kalmanARMALikelihoodInto(y, phi, theta, &scratch.kalman)

		if gotNeg != wantNeg {
			t.Errorf("trial %d (p=%d q=%d n=%d): negLL got %g want %g (Δ=%g)",
				trial, p, q, n, gotNeg, wantNeg, gotNeg-wantNeg)
		}
		if gotSig != wantSig {
			t.Errorf("trial %d: sigma² got %g want %g", trial, gotSig, wantSig)
		}
	}
}

// G-NEW-2: unpackParamsXInto must produce values matching unpackParamsX.
func TestUnpackParamsXInto_Equivalence(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	scratch := acquireParamScratch()
	defer releaseParamScratch(scratch)

	for trial := 0; trial < 30; trial++ {
		p := rng.IntN(4)
		q := rng.IntN(4)
		P := rng.IntN(3)
		Q := rng.IntN(3)
		k := rng.IntN(5)
		withIntercept := rng.IntN(2) == 0

		size := p + q + P + Q + k
		if withIntercept {
			size++
		}
		params := make([]float64, size)
		for i := range params {
			params[i] = rng.NormFloat64()
		}

		wantPhi, wantTheta, wantSPhi, wantSTheta, wantC, wantBeta :=
			unpackParamsX(params, p, q, P, Q, withIntercept, k)
		gotPhi, gotTheta, gotSPhi, gotSTheta, gotC, gotBeta :=
			unpackParamsXInto(scratch, params, p, q, P, Q, withIntercept, k)

		eqSlice := func(name string, got, want []float64) {
			if len(got) != len(want) {
				t.Errorf("trial %d %s len: got %d want %d", trial, name, len(got), len(want))
				return
			}
			for i := range want {
				if got[i] != want[i] {
					t.Errorf("trial %d %s[%d]: got %g want %g", trial, name, i, got[i], want[i])
				}
			}
		}
		eqSlice("phi", gotPhi, wantPhi)
		eqSlice("theta", gotTheta, wantTheta)
		eqSlice("sPhi", gotSPhi, wantSPhi)
		eqSlice("sTheta", gotSTheta, wantSTheta)
		eqSlice("beta", gotBeta, wantBeta)
		if gotC != wantC {
			t.Errorf("trial %d c: got %g want %g", trial, gotC, wantC)
		}
	}
}
