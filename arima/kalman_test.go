package arima

import (
	"math"
	"math/rand/v2"
	"testing"
)


// SS-KALMAN-1: the steady-state freeze must not change the likelihood.
// kalmanARMALikelihood (legacy, no freeze) is the referee. Long series
// with well-damped coefficients engage the freeze within ~100 steps, so
// any per-step drift would accumulate over thousands of frozen steps.
func TestSSKalmanFreezeParity(t *testing.T) {
	cases := []struct {
		phi, theta []float64
		n          int
	}{
		{[]float64{0.5}, nil, 3000},
		{[]float64{0.6, -0.2}, []float64{0.4}, 3000},
		{[]float64{0.3, 0.1}, []float64{-0.4, 0.2, 0.1}, 2500},
		// Seasonal-expanded shape (r=13): slow but should still freeze.
		{[]float64{0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, -0.12}, []float64{0.2}, 2500},
		// Near-unit-root: freeze likely never fires — parity must still hold.
		{[]float64{0.999}, []float64{-0.5}, 2000},
	}
	rng := rand.New(rand.NewPCG(3, 5))
	for ci, c := range cases {
		n := c.n
		y := make([]float64, n)
		res := make([]float64, n)
		for t2 := 0; t2 < n; t2++ {
			e := rng.NormFloat64()
			s := 0.0
			for i := 0; i < len(c.phi) && t2-1-i >= 0; i++ {
				s += c.phi[i] * y[t2-1-i]
			}
			for j := 0; j < len(c.theta) && t2-1-j >= 0; j++ {
				s += c.theta[j] * res[t2-1-j]
			}
			y[t2] = s + e
			res[t2] = e
		}
		var ws kalmanWorkspace
		gotLL, gotS2 := kalmanARMALikelihoodInto(y, c.phi, c.theta, &ws)
		wantLL, wantS2, _ := kalmanARMALikelihood(y, c.phi, c.theta)
		if math.Abs(gotLL-wantLL) > 1e-7*(1+math.Abs(wantLL)) {
			t.Errorf("case %d: negLL freeze drift: got %.10f want %.10f (Δ=%.3e)",
				ci, gotLL, wantLL, gotLL-wantLL)
		}
		if math.Abs(gotS2-wantS2) > 1e-7*(1+math.Abs(wantS2)) {
			t.Errorf("case %d: sigma2 drift: got %.10f want %.10f", ci, gotS2, wantS2)
		}
	}
}

// SS-KALMAN-1 payoff benchmarks: a long well-damped series freezes
// within ~100 steps (the remaining ~20k steps run the O(r) path); the
// near-unit-root variant never freezes (full O(r²) every step). The
// damped case should be several × faster per op.
func benchKalmanLong(b *testing.B, phi []float64) {
	theta := []float64{0.3, -0.1}
	n := 20000
	rng := rand.New(rand.NewPCG(11, 13))
	y := make([]float64, n)
	res := make([]float64, n)
	for t := 0; t < n; t++ {
		e := rng.NormFloat64()
		s := 0.0
		for i := 0; i < len(phi) && t-1-i >= 0; i++ {
			s += phi[i] * y[t-1-i]
		}
		for j := 0; j < len(theta) && t-1-j >= 0; j++ {
			s += theta[j] * res[t-1-j]
		}
		y[t] = s + e
		res[t] = e
	}
	var ws kalmanWorkspace
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kalmanARMALikelihoodInto(y, phi, theta, &ws)
	}
}

func BenchmarkKalmanLongDamped(b *testing.B) {
	// r=13 seasonal-expanded shape, well-damped.
	benchKalmanLong(b, []float64{0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, -0.12})
}

func BenchmarkKalmanLongNearUnit(b *testing.B) {
	phi := make([]float64, 13)
	phi[0] = 1.6
	phi[1] = -0.6005 // roots just outside unit circle — never freezes
	benchKalmanLong(b, phi)
}

// GARD-COL-1: the series-form column-0 solve must match Gardner's full
// stationary covariance column to high precision across random models.
func TestStationaryCovColumn0Parity(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 77))
	checked := 0
	for trial := 0; trial < 80; trial++ {
		p := rng.IntN(4)
		q := rng.IntN(4)
		if p+q == 0 {
			p = 1
		}
		scale := 0.5
		if trial%4 == 3 {
			scale = 1.4
		}
		phi := make([]float64, p)
		theta := make([]float64, q)
		// PACF-style stationary draw.
		mk := func(out []float64) {
			cur := make([]float64, 0, len(out))
			for j := range out {
				a := math.Tanh(rng.NormFloat64() * scale)
				next := make([]float64, j+1)
				for i := 0; i < j; i++ {
					next[i] = cur[i] - a*cur[j-1-i]
				}
				next[j] = a
				cur = next
			}
			copy(out, cur)
		}
		mk(phi)
		mk(theta)
		r := p
		if q+1 > r {
			r = q + 1
		}
		if r < 3 {
			continue
		}
		col := make([]float64, r)
		vb := make([]float64, r)
		tb := make([]float64, r)
		if !stationaryCovColumn0Into(col, vb, tb, phi, theta, p, r) {
			continue // slow-converging draw — fallback path, fine
		}
		checked++
		full := PublicGardner(phi, theta)
		for i := 0; i < r; i++ {
			want := full[i][0]
			if d := math.Abs(col[i] - want); d > 1e-9*(1+math.Abs(want)) {
				t.Errorf("trial %d (p=%d q=%d r=%d) col[%d]: got %.14g want %.14g",
					trial, p, q, r, i, col[i], want)
			}
		}
	}
	if checked < 20 {
		t.Fatalf("only %d column solves converged — termination too strict", checked)
	}
}
