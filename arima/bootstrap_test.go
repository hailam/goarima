package arima

import (
	"math"
	"math/rand/v2"
	"sort"
	"testing"
)

// CDX-1: quickselect-based quantilesFromUnsorted must produce
// bit-identical results to the legacy sort-based path for any input.
// We compare against `sort.Float64s + quantile` across a battery of
// sizes / quantile pairs / seeds, including pathological inputs
// (constant, already-sorted, reverse-sorted, all-NaN-free with ties).
func TestQuantilesFromUnsorted_MatchesSortBased(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 7))
	type spec struct {
		name string
		gen  func(n int) []float64
	}
	specs := []spec{
		{"random", func(n int) []float64 {
			out := make([]float64, n)
			for i := range out {
				out[i] = rng.NormFloat64()
			}
			return out
		}},
		{"sorted", func(n int) []float64 {
			out := make([]float64, n)
			for i := range out {
				out[i] = float64(i)
			}
			return out
		}},
		{"reverse", func(n int) []float64 {
			out := make([]float64, n)
			for i := range out {
				out[i] = float64(n - i)
			}
			return out
		}},
		{"ties", func(n int) []float64 {
			out := make([]float64, n)
			for i := range out {
				out[i] = float64(i % 5) // many duplicates
			}
			return out
		}},
		{"constant", func(n int) []float64 {
			out := make([]float64, n)
			for i := range out {
				out[i] = 3.14
			}
			return out
		}},
	}
	quantilePairs := [][2]float64{
		{0.025, 0.975},
		{0.05, 0.95},
		{0.1, 0.9},
		{0.25, 0.75},
		{0.5, 0.5},
		{0.0, 1.0},
		{0.001, 0.999},
	}
	sizes := []int{1, 2, 3, 5, 10, 100, 1000}
	for _, sp := range specs {
		for _, n := range sizes {
			for _, qp := range quantilePairs {
				name := sp.name + "_n" + intToStr(n) + "_q" + floatToStr(qp[0]) + "-" + floatToStr(qp[1])
				t.Run(name, func(t *testing.T) {
					orig := sp.gen(n)
					// Reference: sort + quantile.
					sorted := append([]float64(nil), orig...)
					sort.Float64s(sorted)
					wantLo := quantile(sorted, qp[0])
					wantHi := quantile(sorted, qp[1])
					// Under test: quickselect.
					col := append([]float64(nil), orig...)
					gotLo, gotHi := quantilesFromUnsorted(col, qp[0], qp[1])
					if math.Abs(gotLo-wantLo) > 1e-12 {
						t.Errorf("loQ=%.4f: got %.6f, want %.6f", qp[0], gotLo, wantLo)
					}
					if math.Abs(gotHi-wantHi) > 1e-12 {
						t.Errorf("hiQ=%.4f: got %.6f, want %.6f", qp[1], gotHi, wantHi)
					}
				})
			}
		}
	}
}

func intToStr(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

func floatToStr(f float64) string {
	// Compact formatter for sub-test names: 0.025 → "0.025", 0.5 → "0.5".
	// Avoids fmt to keep the test self-contained for parallel runs.
	if f == 0 {
		return "0"
	}
	if f == 1 {
		return "1"
	}
	// 3-decimal precision is enough for our quantile test inputs.
	scaled := int(f*1000 + 0.5)
	return "0." + intToStr(scaled)
}

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

// BOOT-PARAM: parametric bootstrap (innovations ~ N(0, σ²)) must
// produce well-formed CIs that grow with horizon. On a clean
// Gaussian-residual case the parametric variant should give CIs
// of comparable width to the non-parametric version (within a
// reasonable tolerance — the latter's width depends on residual
// sample variance, which is itself an estimate of σ).
func TestPredictBoot_ParametricBasicShape(t *testing.T) {
	y := simulateAR1(300, 0.6, 1.0, 1)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 60
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	res, err := m.PredictBootWithOpts(10, PredictBootOpts{
		Alpha:      0.05,
		NSim:       500,
		Seed:       42,
		Parametric: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := range res.Mean {
		if !(res.Lower[i] <= res.Mean[i] && res.Mean[i] <= res.Upper[i]) {
			t.Errorf("CI ordering violated at h=%d", i)
		}
		if math.IsNaN(res.Mean[i]) {
			t.Errorf("NaN at h=%d", i)
		}
	}
	if res.Upper[9]-res.Lower[9] < res.Upper[0]-res.Lower[0] {
		t.Error("parametric CI must widen with horizon")
	}
}

// BOOT-PARAM: legacy PredictBoot wrapper must produce identical
// non-parametric output to PredictBootWithOpts(Parametric=false).
// Same seed → bit-identical paths.
func TestPredictBoot_LegacyEquivalentToOpts(t *testing.T) {
	y := simulateAR1(200, 0.5, 1.0, 7)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 60
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	want, err := m.PredictBoot(10, 0.05, 100, 99, nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.PredictBootWithOpts(10, PredictBootOpts{
		Alpha: 0.05, NSim: 100, Seed: 99, Parametric: false,
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := range want.Mean {
		if got.Mean[i] != want.Mean[i] {
			t.Errorf("Mean[%d]: got %g want %g", i, got.Mean[i], want.Mean[i])
		}
		if got.Lower[i] != want.Lower[i] {
			t.Errorf("Lower[%d]: got %g want %g", i, got.Lower[i], want.Lower[i])
		}
		if got.Upper[i] != want.Upper[i] {
			t.Errorf("Upper[%d]: got %g want %g", i, got.Upper[i], want.Upper[i])
		}
	}
}

// PRED-VAR: m.PredictVar must return well-formed per-horizon variances
// that grow monotonically and match the variance underlying Predict's
// CI bands.
func TestPredictVar_Shape(t *testing.T) {
	y := simulateAR1(300, 0.6, 1.0, 1)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 60
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	const h = 12
	v, err := m.PredictVar(h, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != h {
		t.Fatalf("len(v) = %d, want %d", len(v), h)
	}
	for i := 0; i < h; i++ {
		if v[i] <= 0 {
			t.Errorf("variance[%d] = %g, must be positive", i, v[i])
		}
		if i > 0 && v[i] < v[i-1] {
			t.Errorf("variance must grow monotonically; v[%d]=%g v[%d]=%g", i-1, v[i-1], i, v[i])
		}
	}
	// Cross-check vs Predict's CIs at alpha=0.05: width = 2 · z · √var.
	_, lo, hi, err := m.Predict(h, 0.05, nil)
	if err != nil {
		t.Fatal(err)
	}
	z := 1.959964 // normPPF(0.975)
	for i := 0; i < h; i++ {
		expected := 2 * z * math.Sqrt(v[i])
		got := hi[i] - lo[i]
		if math.Abs(got-expected)/expected > 1e-6 {
			t.Errorf("h=%d: PredictVar implies width %g, Predict bands give width %g",
				i, expected, got)
		}
	}
}

// PRED-VAR: BootResult.Variance must be populated and approximately
// equal to the analytical PredictVar on a clean Gaussian-residual
// case.
func TestPredictBoot_Variance(t *testing.T) {
	y := simulateAR1(300, 0.6, 1.0, 1)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 60
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	res, err := m.PredictBootWithOpts(10, PredictBootOpts{
		Alpha:      0.05,
		NSim:       2000,
		Seed:       42,
		Parametric: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Variance) != 10 {
		t.Fatalf("len(Variance) = %d, want 10", len(res.Variance))
	}
	for i, v := range res.Variance {
		if v <= 0 {
			t.Errorf("Variance[%d] = %g, must be positive", i, v)
		}
	}
	// Compare to analytical PredictVar — should be close with NSim=2000.
	pv, err := m.PredictVar(10, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10; i++ {
		ratio := res.Variance[i] / pv[i]
		if ratio < 0.5 || ratio > 2.0 {
			t.Errorf("h=%d empirical/analytical variance ratio = %g; expected in [0.5, 2.0]",
				i, ratio)
		}
	}
}
