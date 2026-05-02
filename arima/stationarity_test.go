package arima

import (
	"math"
	"testing"

	"github.com/hailam/goarima/datasets"
)

// Mirrors test_embedding for KPSS/PP/ADF (all use the same _embed).
func TestEmbedding(t *testing.T) {
	x := []float64{0, 1, 2, 3, 4}
	got, err := embed(x, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float64{
		{1, 2, 3, 4},
		{0, 1, 2, 3},
	}
	if !matEq(got, want) {
		t.Errorf("embed got %v want %v", got, want)
	}

	y := []float64{1, -1, 0, 2, -1, -2, 3}

	got1, _ := embed(y, 1)
	if !matEq(got1, [][]float64{{1, -1, 0, 2, -1, -2, 3}}) {
		t.Errorf("embed k=1 = %v", got1)
	}

	g2, _ := embedT(y, 2)
	want2 := [][]float64{
		{-1, 1}, {0, -1}, {2, 0}, {-1, 2}, {-2, -1}, {3, -2},
	}
	if !matEq(g2, want2) {
		t.Errorf("embedT k=2 mismatch: got %v", g2)
	}

	g3, _ := embedT(y, 3)
	want3 := [][]float64{
		{0, -1, 1}, {2, 0, -1}, {-1, 2, 0}, {-2, -1, 2}, {3, -2, -1},
	}
	if !matEq(g3, want3) {
		t.Errorf("embedT k=3 mismatch: got %v", g3)
	}

	if _, err := embed(y, 8); err == nil {
		t.Error("expected error for k > n")
	}
}

func matEq(a, b [][]float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if math.Abs(a[i][j]-b[i][j]) > 1e-9 {
				return false
			}
		}
	}
	return true
}

// Mirrors test_kpss for level and trend nulls.
//
// Pmdarima clips the p-value at the table extremes (rule=2), so a wrong stat
// can still produce pval=0.01. We additionally check the underlying stat value
// against pmdarima's own intermediate computation.
func TestKPSS(t *testing.T) {
	austres := datasets.LoadAustres()
	wantStats := map[string]float64{
		"level": 2.312205,
		"trend": 0.538032,
	}
	for _, null := range []string{"level", "trend"} {
		res, err := KPSSTest(austres, KPSSTestOpts{Alpha: 0.05, Null: null, LShort: true})
		if err != nil {
			t.Fatalf("%s: %v", null, err)
		}
		if !res.ShouldDiff {
			t.Errorf("%s austres should diff", null)
		}
		if math.Abs(res.PValue-0.01) > 1e-6 {
			t.Errorf("%s austres pval = %v want 0.01", null, res.PValue)
		}
		if math.Abs(res.Stat-wantStats[null]) > 1e-4 {
			t.Errorf("%s austres stat = %v want %v", null, res.Stat, wantStats[null])
		}
	}

	// issue #67 sample
	x := []float64{1, -1, 0, 2, -1, -2, 3}
	for _, null := range []string{"level", "trend"} {
		res, err := KPSSTest(x, KPSSTestOpts{Alpha: 0.05, Null: null, LShort: true})
		if err != nil {
			t.Fatalf("%s small: %v", null, err)
		}
		if null == "level" {
			if res.ShouldDiff {
				t.Errorf("level small expected no diff, got pval=%v", res.PValue)
			}
			if math.Abs(res.PValue-0.1) > 1e-6 {
				t.Errorf("level small pval=%v want 0.1", res.PValue)
			}
		} else {
			if !res.ShouldDiff {
				t.Errorf("trend small expected diff, got pval=%v", res.PValue)
			}
			if math.Abs(res.PValue-0.01) > 1e-6 {
				t.Errorf("trend small pval=%v want 0.01", res.PValue)
			}
		}
	}
}

// Mirrors test_kpss_corner.
func TestKPSSCorner(t *testing.T) {
	x := datasets.LoadAustres()
	if _, err := KPSSTest(x, KPSSTestOpts{Alpha: 0.05, Null: "weird"}); err == nil {
		t.Error("expected error for invalid null")
	}
}

// Mirrors test_adf_p_value (small dataset returns pval = 0.01).
func TestADFPValueSmall(t *testing.T) {
	x := []float64{1, -1, 0, 2, -1, -2, 3}
	res, err := ADFTest(x, ADFTestOpts{Alpha: 0.05})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(res.PValue-0.01) > 1e-6 {
		t.Errorf("pval=%v want 0.01", res.PValue)
	}
	if res.ShouldDiff {
		t.Errorf("should not diff (pval < alpha)")
	}
}

// Mirrors test_adf for austres on k=1, 2, default.
//
// Expected p-values come from running pmdarima.ADFTest on the same input.
// Tighter than R's published rounded values to catch silent numerical drift.
func TestADFAustres(t *testing.T) {
	austres := datasets.LoadAustres()
	cases := []struct {
		k      int
		hasK   bool
		expect float64
	}{
		{1, true, 0.8488036035966902},
		{2, true, 0.7060733181808536},
	}
	for _, c := range cases {
		opts := ADFTestOpts{Alpha: 0.05, K: c.k, HasK: c.hasK}
		res, err := ADFTest(austres, opts)
		if err != nil {
			t.Fatalf("k=%d: %v", c.k, err)
		}
		if math.Abs(res.PValue-c.expect) > 1e-6 {
			t.Errorf("k=%d: pval=%v want %v", c.k, res.PValue, c.expect)
		}
		if !res.ShouldDiff {
			t.Errorf("k=%d should diff (pval > alpha)", c.k)
		}
	}
	// Default k uses trunc((n-1)^(1/3)); should match R's published value too.
	res, err := ADFTest(austres, ADFTestOpts{Alpha: 0.05})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(res.PValue-0.3493465) > 0.005 {
		t.Errorf("default k: pval=%v want ≈0.3493", res.PValue)
	}
}

// Mirrors test_adf_corner.
func TestADFCorner(t *testing.T) {
	if _, err := ADFTest([]float64{1, 2, 3}, ADFTestOpts{Alpha: 0.05, K: -1, HasK: true}); err == nil {
		t.Error("expected error for k=-1")
	}
}

// Mirrors test_pp for austres in lshort=true and lshort=false.
// Expected values come from pmdarima on the same input.
func TestPPAustres(t *testing.T) {
	austres := datasets.LoadAustres()
	cases := []struct {
		lshort bool
		expect float64
	}{
		{true, 0.9786066224866204},
		{false, 0.9514588685512652},
	}
	for _, c := range cases {
		res, err := PPTest(austres, PPTestOpts{Alpha: 0.05, LShort: c.lshort})
		if err != nil {
			t.Fatalf("lshort=%v: %v", c.lshort, err)
		}
		if !res.ShouldDiff {
			t.Errorf("lshort=%v should diff", c.lshort)
		}
		if math.Abs(res.PValue-c.expect) > 1e-6 {
			t.Errorf("lshort=%v: pval=%v want %v", c.lshort, res.PValue, c.expect)
		}
	}
}

// Mirrors test_base_cases — empty/nil should return NaN, false.
func TestStationarityBaseCases(t *testing.T) {
	a, _ := ADFTest(nil, ADFTestOpts{})
	if !math.IsNaN(a.PValue) || a.ShouldDiff {
		t.Errorf("ADF nil case: %v", a)
	}
	k, _ := KPSSTest(nil, KPSSTestOpts{})
	if !math.IsNaN(k.PValue) || k.ShouldDiff {
		t.Errorf("KPSS nil case: %v", k)
	}
	p, _ := PPTest(nil, PPTestOpts{})
	if !math.IsNaN(p.PValue) || p.ShouldDiff {
		t.Errorf("PP nil case: %v", p)
	}
}
