package arima

import (
	"math"
	"math/rand/v2"
	"testing"
)

// BootstrapInference: mechanics — output shape aligns with Params(),
// SEs positive, Lower < Mean < Upper, samples available when not opted
// out.
func TestBootstrapInference_BasicShape(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	const n = 200
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = 0.6*y[i-1] + rng.NormFloat64()
	}
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 60
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	res, err := m.BootstrapInference(BootstrapInferenceOpts{
		B:     50,
		Alpha: 0.05,
		Seed:  42,
	})
	if err != nil {
		t.Fatal(err)
	}

	wantP := len(m.Params())
	for _, name := range []struct {
		n string
		s []float64
	}{
		{"Params", res.Params}, {"StdErr", res.StdErr},
		{"Lower", res.Lower}, {"Upper", res.Upper},
	} {
		if len(name.s) != wantP {
			t.Errorf("%s len = %d, want %d", name.n, len(name.s), wantP)
		}
	}
	if len(res.Samples) != 50 {
		t.Errorf("Samples len = %d, want 50", len(res.Samples))
	}

	for j := 0; j < wantP; j++ {
		if res.StdErr[j] <= 0 {
			t.Errorf("StdErr[%d] = %g, must be positive", j, res.StdErr[j])
		}
		if !(res.Lower[j] <= res.Params[j] && res.Params[j] <= res.Upper[j]) {
			t.Errorf("CI ordering violated at j=%d: lo=%g mean=%g hi=%g",
				j, res.Lower[j], res.Params[j], res.Upper[j])
		}
	}
}

// BootstrapInference with OmitSamples should leave Samples nil.
func TestBootstrapInference_OmitSamples(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	y := make([]float64, 150)
	for i := 1; i < 150; i++ {
		y[i] = 0.5*y[i-1] + rng.NormFloat64()
	}
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 60
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	res, err := m.BootstrapInference(BootstrapInferenceOpts{
		B:           30,
		Alpha:       0.05,
		Seed:        7,
		OmitSamples: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Samples != nil {
		t.Errorf("Samples should be nil under OmitSamples; got len=%d", len(res.Samples))
	}
	// SEs and CIs still populated.
	if len(res.StdErr) == 0 {
		t.Error("StdErr empty under OmitSamples")
	}
}

// BootstrapStdErrors convenience wrapper returns just the SEs.
func TestBootstrapStdErrors_Convenience(t *testing.T) {
	rng := rand.New(rand.NewPCG(15, 16))
	y := make([]float64, 200)
	for i := 1; i < 200; i++ {
		y[i] = 0.4*y[i-1] + rng.NormFloat64()
	}
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 60
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	se, err := m.BootstrapStdErrors(30, 99)
	if err != nil {
		t.Fatal(err)
	}
	if len(se) != len(m.Params()) {
		t.Errorf("SE len = %d, want %d", len(se), len(m.Params()))
	}
	for j, s := range se {
		if s <= 0 {
			t.Errorf("SE[%d] = %g, must be positive", j, s)
		}
	}
}

// Bootstrap SEs should be in the same ballpark as Hessian-based SEs
// when n is large enough that the asymptotic-normality assumption is
// reasonable. Generous tolerance — bootstrap noise at B=50 is real.
func TestBootstrapInference_LargeNAgreesWithHessianSEs(t *testing.T) {
	if testing.Short() {
		t.Skip("Hessian Summary computation + 50 bootstrap fits — slow")
	}
	rng := rand.New(rand.NewPCG(21, 22))
	const n = 500
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = 0.6*y[i-1] + rng.NormFloat64()
	}
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 60
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	asymptoticSE, err := m.StdErrors()
	if err != nil {
		t.Fatal(err)
	}
	bootSE, err := m.BootstrapStdErrors(80, 13)
	if err != nil {
		t.Fatal(err)
	}
	if len(asymptoticSE) != len(bootSE) {
		t.Fatalf("SE length mismatch: asymp=%d boot=%d", len(asymptoticSE), len(bootSE))
	}
	for j := 0; j < len(asymptoticSE); j++ {
		ratio := bootSE[j] / asymptoticSE[j]
		t.Logf("param[%d]: asymp=%.4f boot=%.4f ratio=%.2f",
			j, asymptoticSE[j], bootSE[j], ratio)
		// Generous: at large n, bootstrap and asymptotic should agree
		// within a factor of ~3 — wider would suggest the bootstrap
		// found a multimodal likelihood or some pathology.
		if ratio < 0.3 || ratio > 3.0 {
			t.Errorf("param[%d]: bootstrap/asymptotic SE ratio = %g; expected in [0.3, 3.0]",
				j, ratio)
		}
	}
}

// BootstrapInference error paths.
func TestBootstrapInference_ErrorPaths(t *testing.T) {
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	if _, err := m.BootstrapInference(BootstrapInferenceOpts{B: 100, Alpha: 0.05}); err == nil {
		t.Error("unfitted model should error")
	}
	rng := rand.New(rand.NewPCG(33, 34))
	y := make([]float64, 100)
	for i := 1; i < 100; i++ {
		y[i] = 0.5*y[i-1] + rng.NormFloat64()
	}
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := m.BootstrapInference(BootstrapInferenceOpts{B: 5, Alpha: 0.05}); err == nil {
		t.Error("B<10 should error")
	}
	if _, err := m.BootstrapInference(BootstrapInferenceOpts{B: 50, Alpha: 0}); err == nil {
		t.Error("Alpha=0 should error")
	}
	if _, err := m.BootstrapInference(BootstrapInferenceOpts{B: 50, Alpha: 1.5}); err == nil {
		t.Error("Alpha>1 should error")
	}
}

// Math sanity: empirical variance from BootstrapInference should equal
// hand-computed sample variance from the same Samples matrix (so
// callers can trust the StdErr field).
func TestBootstrapInference_StdErrMatchesEmpirical(t *testing.T) {
	rng := rand.New(rand.NewPCG(41, 42))
	y := make([]float64, 200)
	for i := 1; i < 200; i++ {
		y[i] = 0.5*y[i-1] + rng.NormFloat64()
	}
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 60
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	res, err := m.BootstrapInference(BootstrapInferenceOpts{
		B: 30, Alpha: 0.05, Seed: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	for j := 0; j < len(res.Params); j++ {
		mean := 0.0
		for _, row := range res.Samples {
			mean += row[j]
		}
		mean /= float64(len(res.Samples))
		ssq := 0.0
		for _, row := range res.Samples {
			d := row[j] - mean
			ssq += d * d
		}
		want := math.Sqrt(ssq / float64(len(res.Samples)-1))
		if math.Abs(res.StdErr[j]-want)/want > 1e-9 {
			t.Errorf("param[%d]: StdErr=%g, want %g", j, res.StdErr[j], want)
		}
	}
}
