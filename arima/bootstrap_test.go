package arima

import (
	"math"
	"testing"
)

func TestPredictBootBasic(t *testing.T) {
	y := simulateAR1(300, 0.6, 1.0, 1)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 60
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	res, err := m.PredictBoot(10, 0.05, 500, 42, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Mean) != 10 || len(res.Lower) != 10 || len(res.Upper) != 10 {
		t.Errorf("lengths: mean=%d lo=%d hi=%d", len(res.Mean), len(res.Lower), len(res.Upper))
	}
	for i := range res.Mean {
		if !(res.Lower[i] <= res.Mean[i] && res.Mean[i] <= res.Upper[i]) {
			t.Errorf("CI ordering violated at h=%d: lo=%v mean=%v hi=%v",
				i, res.Lower[i], res.Mean[i], res.Upper[i])
		}
		if math.IsNaN(res.Mean[i]) || math.IsNaN(res.Lower[i]) || math.IsNaN(res.Upper[i]) {
			t.Errorf("NaN at h=%d", i)
		}
	}
	// Width should grow with horizon.
	w0 := res.Upper[0] - res.Lower[0]
	wEnd := res.Upper[9] - res.Lower[9]
	if wEnd < w0 {
		t.Errorf("CI shrunk over horizon: %.3f → %.3f", w0, wEnd)
	}
}

func TestPredictBootShortSeed(t *testing.T) {
	y := simulateAR1(80, 0.4, 1.0, 99)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 30
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	r1, err := m.PredictBoot(5, 0.1, 200, 7, nil)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := m.PredictBoot(5, 0.1, 200, 7, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range r1.Mean {
		if r1.Mean[i] != r2.Mean[i] {
			t.Errorf("not deterministic at h=%d: %v vs %v", i, r1.Mean[i], r2.Mean[i])
		}
	}
}

func TestPredictBootErrors(t *testing.T) {
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	if _, err := m.PredictBoot(5, 0.05, 100, 0, nil); err == nil {
		t.Error("expected error: not fitted")
	}
}

// L-7: PredictBoot is parallelized once nSim ≥ 64 (constant in bootstrap.go).
// To stay deterministic for users — same seed produces identical Paths
// regardless of how the work is partitioned across goroutines — each path
// uses a per-path PCG seeded from (seed, s). This test verifies that
// invariant by comparing two runs at the same seed but different nSim
// (which forces a different worker partition under the s % nWorkers
// scheduling).
func TestPredictBoot_DeterministicAcrossWorkerCounts(t *testing.T) {
	y := simulateAR1(300, 0.6, 1.0, 7)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 60
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	// nSim=128: parallel path (≥ threshold of 64).
	res128, err := m.PredictBoot(8, 0.05, 128, 999, nil)
	if err != nil {
		t.Fatal(err)
	}
	// First 50 paths should match a smaller-nSim run that goes serial
	// (50 < 64 → serial path). Both use the same per-path seeding, so
	// paths[0..49] must be bit-identical.
	res50, err := m.PredictBoot(8, 0.05, 50, 999, nil)
	if err != nil {
		t.Fatal(err)
	}
	for s := 0; s < 50; s++ {
		for h := 0; h < 8; h++ {
			if res128.Paths[s][h] != res50.Paths[s][h] {
				t.Errorf("path[%d][%d]: nSim=128 (parallel) = %g, nSim=50 (serial) = %g — per-path seeding broken",
					s, h, res128.Paths[s][h], res50.Paths[s][h])
				return
			}
		}
	}
}
