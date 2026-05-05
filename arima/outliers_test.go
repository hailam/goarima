package arima

import (
	"math/rand/v2"
	"testing"
)

// G-NEW-3a: detect a single planted AO. The model is ARIMA(0,1,1)
// (random walk + MA1, the default DetectOutliers shape), and the
// outlier is a +20σ impulse at t=50. The detector must return at
// least one outlier of type AO with index within ±1 of 50.
func TestDetectOutliers_PlantedAO(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	n := 200
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = y[i-1] + rng.NormFloat64() // random walk
	}
	// Plant a single, large AO.
	const plantIdx = 50
	const plantMag = 20.0
	y[plantIdx] += plantMag

	outs, _, err := DetectOutliers(y, DetectOutliersOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(outs) == 0 {
		t.Fatal("expected at least one outlier; got none")
	}
	found := false
	for _, o := range outs {
		if o.Type == OutlierAO && o.Index >= plantIdx-1 && o.Index <= plantIdx+1 {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected AO near %d; got %+v", plantIdx, outs)
	}
}

// G-NEW-3a: detect a planted LS. A +15σ step starting at t=120 in
// a random-walk series. Detector must return at least one LS at
// approximately the right index.
func TestDetectOutliers_PlantedLS(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 22))
	n := 250
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = y[i-1] + rng.NormFloat64()
	}
	const plantIdx = 120
	const plantMag = 15.0
	for i := plantIdx; i < n; i++ {
		y[i] += plantMag
	}

	outs, _, err := DetectOutliers(y, DetectOutliersOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(outs) == 0 {
		t.Fatal("expected at least one outlier; got none")
	}
	// The LS should be within a small window of plantIdx (the
	// detector can latch onto the immediate post-step residual which
	// gives an off-by-one shift; tolerate ±2).
	found := false
	for _, o := range outs {
		if o.Type == OutlierLS && o.Index >= plantIdx-2 && o.Index <= plantIdx+2 {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected LS near %d; got %+v", plantIdx, outs)
	}
}

// OUT-TC-IO: detect a planted Temporary Change. A +18σ initial spike
// at t=80 that decays geometrically with δ=0.7. Detector with TC enabled
// must return at least one TC near 80.
func TestDetectOutliers_PlantedTC(t *testing.T) {
	rng := rand.New(rand.NewPCG(101, 102))
	n := 250
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = y[i-1] + rng.NormFloat64()
	}
	const plantIdx = 80
	const plantMag = 18.0
	const delta = 0.7
	v := 1.0
	for i := plantIdx; i < n; i++ {
		y[i] += plantMag * v
		v *= delta
	}

	outs, _, err := DetectOutliers(y, DetectOutliersOpts{
		Types:       []OutlierType{OutlierAO, OutlierLS, OutlierTC, OutlierIO},
		TCDecayRate: 0.7,
	})
	if err != nil {
		t.Fatal(err)
	}
	hasTC := false
	for _, o := range outs {
		if o.Type == OutlierTC && o.Index >= plantIdx-2 && o.Index <= plantIdx+2 {
			hasTC = true
			break
		}
	}
	if !hasTC {
		t.Errorf("expected TC near %d in result; got %+v", plantIdx, outs)
	}
}

// OUT-TC-IO: detect a planted Innovational Outlier. An IO at τ
// perturbs the innovation at time τ — equivalent to adding ω·ψ_(t-τ)
// to y for t ≥ τ. For an ARIMA(0,1,1) random-walk-plus-MA1 default
// model, ψ ≈ [1, 1, 1, …] (cumulative sum of MA), so an IO mimics a
// level shift in shape but is structurally distinct.
func TestDetectOutliers_PlantedIO(t *testing.T) {
	rng := rand.New(rand.NewPCG(111, 112))
	n := 200
	// Generate ARMA(1,0) with phi=0.6 driven by epsilons we control.
	const phi = 0.6
	const plantIdx = 80
	const plantMag = 25.0
	eps := make([]float64, n)
	for i := range eps {
		eps[i] = rng.NormFloat64()
	}
	eps[plantIdx] += plantMag // innovation perturbation
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = phi*y[i-1] + eps[i]
	}

	outs, _, err := DetectOutliers(y, DetectOutliersOpts{
		Order: Order{P: 1, D: 0, Q: 0},
		Types: []OutlierType{OutlierAO, OutlierLS, OutlierTC, OutlierIO},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Got outliers: %+v", outs)
	// Direction check: at the plant index the largest standardised
	// residual should appear. AO and IO are nearly indistinguishable
	// for AR(1) (since IO becomes a single-impulse residual perturbation).
	// Accept either type at the plant index.
	hasAt := false
	for _, o := range outs {
		if o.Index >= plantIdx-1 && o.Index <= plantIdx+1 &&
			(o.Type == OutlierIO || o.Type == OutlierAO) {
			hasAt = true
			break
		}
	}
	if !hasAt {
		t.Errorf("expected IO/AO near %d; got %+v", plantIdx, outs)
	}
}

// String() coverage for new types.
func TestOutlierType_StringCoverage(t *testing.T) {
	for _, tc := range []struct {
		typ  OutlierType
		want string
	}{
		{OutlierAO, "AO"},
		{OutlierLS, "LS"},
		{OutlierTC, "TC"},
		{OutlierIO, "IO"},
	} {
		if got := tc.typ.String(); got != tc.want {
			t.Errorf("(%d).String() = %q, want %q", int(tc.typ), got, tc.want)
		}
	}
}

// G-NEW-3a: A clean random-walk series with no outliers should
// return an empty outlier list — false-positive guard.
func TestDetectOutliers_CleanSeriesIsEmpty(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 32))
	n := 200
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = y[i-1] + rng.NormFloat64()
	}
	outs, _, err := DetectOutliers(y, DetectOutliersOpts{
		CritVal: 4.0, // tighter threshold to suppress sampling-noise hits
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(outs) > 0 {
		t.Errorf("clean series produced %d outliers: %+v", len(outs), outs)
	}
}

// G-NEW-3a: Detect both an AO and an LS in the same series.
// MaxIter=5 leaves headroom; we expect at least one of each type
// to be reported, and indices to land near the plants.
func TestDetectOutliers_PlantedAOAndLS(t *testing.T) {
	rng := rand.New(rand.NewPCG(41, 42))
	n := 250
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = y[i-1] + rng.NormFloat64()
	}
	const aoIdx, aoMag = 60, 20.0
	const lsIdx, lsMag = 150, 12.0
	y[aoIdx] += aoMag
	for i := lsIdx; i < n; i++ {
		y[i] += lsMag
	}

	outs, _, err := DetectOutliers(y, DetectOutliersOpts{})
	if err != nil {
		t.Fatal(err)
	}
	hasAO, hasLS := false, false
	for _, o := range outs {
		if o.Type == OutlierAO && o.Index >= aoIdx-2 && o.Index <= aoIdx+2 {
			hasAO = true
		}
		if o.Type == OutlierLS && o.Index >= lsIdx-2 && o.Index <= lsIdx+2 {
			hasLS = true
		}
	}
	if !hasAO || !hasLS {
		t.Errorf("expected both AO~%d and LS~%d; got %+v", aoIdx, lsIdx, outs)
	}
}

// G-NEW-3a: outlier coefficients in the returned slice must agree
// with the final model's β tail (they are populated from m.Beta()
// after the last refit).
func TestDetectOutliers_CoefMatchesBeta(t *testing.T) {
	rng := rand.New(rand.NewPCG(51, 52))
	n := 200
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = y[i-1] + rng.NormFloat64()
	}
	y[80] += 25

	outs, m, err := DetectOutliers(y, DetectOutliersOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(outs) == 0 {
		t.Fatal("expected at least one outlier")
	}
	betas := m.Beta()
	if len(betas) < len(outs) {
		t.Fatalf("Beta() len=%d < outliers len=%d", len(betas), len(outs))
	}
	offset := len(betas) - len(outs)
	for i, o := range outs {
		if o.Coef != betas[offset+i] {
			t.Errorf("outlier %d Coef=%g, Beta tail=%g", i, o.Coef, betas[offset+i])
		}
	}
}
