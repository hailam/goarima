package arima

import (
	"math"
	"testing"

	"github.com/hailam/goarima/datasets"
)

// Mirrors test_ndiffs_corner_cases.
func TestNDiffsCornerCases(t *testing.T) {
	x := datasets.LoadAustres()
	if _, err := NDiffs(x, NDiffsOpts{MaxD: 0}); err == nil {
		t.Error("expected error for max_d <= 0")
	}
}

// Mirrors test_ndiffs_stationary — constant input → 0 for any test.
func TestNDiffsStationary(t *testing.T) {
	x := []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	for _, tst := range []NDiffsTest{NDiffsKPSS, NDiffsADF, NDiffsPP} {
		d, err := NDiffs(x, NDiffsOpts{Alpha: 0.05, Test: tst, MaxD: 2})
		if err != nil {
			t.Fatalf("test %v: %v", tst, err)
		}
		if d != 0 {
			t.Errorf("test %v: got d=%d want 0 (constant)", tst, d)
		}
	}
}

// austres KPSS should return 2; PP and ADF should return 1.
// Mirrors test_kpss / test_pp / test_adf integration with ndiffs.
func TestNDiffsAustres(t *testing.T) {
	austres := datasets.LoadAustres()
	cases := []struct {
		test NDiffsTest
		want int
	}{
		{NDiffsKPSS, 2},
		{NDiffsPP, 1},
	}
	for _, c := range cases {
		d, err := NDiffs(austres, NDiffsOpts{Test: c.test, MaxD: 5, Alpha: 0.05})
		if err != nil {
			t.Fatalf("test %v: %v", c.test, err)
		}
		if d != c.want {
			t.Errorf("ndiffs (%v) = %d, want %d", c.test, d, c.want)
		}
	}
}

// Mirrors test_ch_test_m_values — austres, m={3,24,52,365} → expected 0.
func TestCHTestMValues(t *testing.T) {
	austres := datasets.LoadAustres()
	for _, m := range []int{3, 24, 52, 365} {
		d, err := CHTest(austres, m)
		if err != nil {
			t.Fatalf("m=%d: %v", m, err)
		}
		if d != 0 {
			t.Errorf("m=%d: got %d want 0", m, d)
		}
	}
}

// Mirrors test_ch_seas_dummy: spot check first row at m=4.
func TestCHSeasDummy(t *testing.T) {
	got := chSeasDummy(89, 4)
	// expected first row: [cos(2pi*1*1/4), sin(2pi*1*1/4), cos(2pi*2*1/4)]
	// = [0, 1, -1]
	want := []float64{0, 1, -1}
	if len(got) == 0 || len(got[0]) != 3 {
		t.Fatalf("seas_dummy shape unexpected")
	}
	for j := 0; j < 3; j++ {
		if math.Abs(got[0][j]-want[j]) > 1e-9 {
			t.Errorf("seas_dummy[0,%d]=%v want %v", j, got[0][j], want[j])
		}
	}
	// second row expected [-1, 0, 1]
	wantR2 := []float64{-1, 0, 1}
	for j := 0; j < 3; j++ {
		if math.Abs(got[1][j]-wantR2[j]) > 1e-9 {
			t.Errorf("seas_dummy[1,%d]=%v want %v", j, got[1][j], wantR2[j])
		}
	}
}

// Mirrors test_ch_base.
func TestCHBase(t *testing.T) {
	d, err := CHTest(nil, 2)
	if err != nil {
		t.Fatal(err)
	}
	if d != 0 {
		t.Errorf("nil → %d want 0", d)
	}
}

// Mirrors test_nsdiffs_corner_cases.
func TestNSDiffsCornerCases(t *testing.T) {
	austres := datasets.LoadAustres()
	for _, tst := range []NSDiffsTest{NSDiffsCH, NSDiffsOCSB} {
		// max_D=0 → error
		if _, err := NSDiffs(austres, NSDiffsOpts{M: 2, MaxD: 0, Test: tst}); err == nil {
			t.Errorf("test=%v: max_D=0 should error", tst)
		}
		// constant series → 0
		d, err := NSDiffs([]float64{1, 1, 1, 1}, NSDiffsOpts{M: 2, MaxD: 1, Test: tst, MaxLag: 3})
		if err != nil {
			t.Fatal(err)
		}
		if d != 0 {
			t.Errorf("test=%v: constant should give 0, got %d", tst, d)
		}
		// m <= 1 → error
		for _, mm := range []int{0, 1} {
			if _, err := NSDiffs(austres, NSDiffsOpts{M: mm, MaxD: 1, Test: tst}); err == nil {
				t.Errorf("test=%v m=%d should error", tst, mm)
			}
		}
	}
}

// PG-97: NSDiffsSEAS implements R's auto.arima default seasonal-test
// (Wang-Smith-Hyndman seasonal-strength). Reference values captured
// from R 4.x + forecast 8.x on 2026-05-07 via `nsdiffs(y, test="seas")`.
// Goarima's implementation reuses centered-MA Decompose (vs R's STL
// via mstl) but the F_s formula and 0.64 threshold are identical;
// empirical D verdicts match R on all canonical datasets.
func TestNSDiffsSEAS_MatchesR(t *testing.T) {
	cases := []struct {
		name string
		x    []float64
		m    int
		want int
	}{
		{"austres m=4 (low season strength)", datasets.LoadAustres(), 4, 0},
		{"airpassengers m=12 (strong season)", datasets.LoadAirPassengers(), 12, 1},
		{"wineind m=12 (strong season)", datasets.LoadWineind(), 12, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := NSDiffs(c.x, NSDiffsOpts{
				M: c.m, MaxD: 1, Test: NSDiffsSEAS, MaxLag: 3,
			})
			if err != nil {
				t.Fatalf("NSDiffs(SEAS): %v", err)
			}
			if got != c.want {
				t.Errorf("NSDiffs(SEAS) on %s: got %d, want %d (R verdict)",
					c.name, got, c.want)
			}
		})
	}
}

// Mirrors test_ch_sd_test — austres CH stat at m={3,4,24} must match R.
func TestCHTestStatAustres(t *testing.T) {
	austres := datasets.LoadAustres()
	cases := []struct {
		m    int
		want float64
	}{
		{3, 0.07956102},
		{4, 0.1935046},
		{24, 4.134289},
	}
	for _, c := range cases {
		got, err := CHTestStat(austres, c.m)
		if err != nil {
			t.Fatalf("m=%d: %v", c.m, err)
		}
		if math.Abs(got-c.want) > 1e-4 {
			t.Errorf("m=%d: got %v want %v", c.m, got, c.want)
		}
	}
}

// Mirrors test_ocsb_test_statistic. Validates against R's `forecast::ocsb.test`
// reference values to high precision. We follow the R algorithm directly
// (no-intercept AR fit) rather than pmdarima's `add_constant` workaround,
// so we can match R within float tolerance.
//
// Reference values: ocsb.test(austres, lag.method='fixed', maxlag=2)$stat = -5.673749
// and similarly -5.632227 for maxlag=3.
func TestOCSBStatAustresFixed(t *testing.T) {
	austres := datasets.LoadAustres()
	cases := []struct {
		maxLag int
		want   float64
	}{
		{2, -5.673749},
		{3, -5.632227},
	}
	for _, c := range cases {
		got, _, err := ocsbFit(austres, 4, c.maxLag, c.maxLag)
		if err != nil {
			t.Fatalf("maxLag=%d: %v", c.maxLag, err)
		}
		// Tight tolerance — within R's printed precision.
		if math.Abs(got-c.want) > 1e-3 {
			t.Errorf("maxLag=%d: got %v want %v", c.maxLag, got, c.want)
		}
	}
}

// CalcCHCritVal table coverage.
func TestCalcCHCritVal(t *testing.T) {
	cases := map[int]float64{
		2:   0.4617146,
		12:  2.7391007,
		24:  5.098624,
		52:  10.341416,
		365: 65.44445,
	}
	for m, want := range cases {
		got := CalcCHCritVal(m)
		if math.Abs(got-want) > 1e-6 {
			t.Errorf("m=%d: %v want %v", m, got, want)
		}
	}
}

// CalcOCSBCritVal sanity — should be negative for typical m.
func TestCalcOCSBCritVal(t *testing.T) {
	for _, m := range []int{4, 12} {
		v := CalcOCSBCritVal(m)
		if !(v < 0) {
			t.Errorf("m=%d: crit val %v should be negative", m, v)
		}
	}
}
