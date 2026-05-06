package arima

import (
	"math"
	"testing"

	"github.com/hailam/goarima/datasets"
)

// TestSTL_AdditiveDecomposition verifies STL produces a valid
// additive decomposition: y == trend + seasonal + remainder bit-for-bit.
func TestSTL_AdditiveDecomposition(t *testing.T) {
	y := datasets.LoadAirPassengers()
	trend, seasonal, remainder := STL(y, 12, 11, stlDefaultTWindow(12, 11), 2)
	if len(trend) != len(y) || len(seasonal) != len(y) || len(remainder) != len(y) {
		t.Fatalf("length mismatch: trend=%d seasonal=%d remainder=%d y=%d",
			len(trend), len(seasonal), len(remainder), len(y))
	}
	for i := range y {
		got := trend[i] + seasonal[i] + remainder[i]
		if math.Abs(got-y[i]) > 1e-9 {
			t.Errorf("decomposition not additive at i=%d: y=%.6f trend+seasonal+remainder=%.6f",
				i, y[i], got)
		}
	}
}

// TestSTL_ProducesPositiveSeasStrength verifies that STL on a clearly
// seasonal series (airpassengers, wineind) produces a seasonal-strength
// statistic above the 0.64 threshold and a constant series produces ~0.
func TestSTL_ProducesPositiveSeasStrength(t *testing.T) {
	cases := []struct {
		name    string
		y       []float64
		m       int
		wantPos bool // true → expect F_s > 0.64
	}{
		{"airpassengers", datasets.LoadAirPassengers(), 12, true},
		{"wineind", datasets.LoadWineind(), 12, true},
		{"constant", make([]float64, 240), 12, false}, // all zeros — no seasonality
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fs, err := seasStrength(c.y, c.m)
			if err != nil {
				t.Fatal(err)
			}
			if c.wantPos && fs <= 0.64 {
				t.Errorf("F_s on %s: got %.4f, expected > 0.64", c.name, fs)
			}
			if !c.wantPos && fs > 0.64 {
				t.Errorf("F_s on %s: got %.4f, expected ≤ 0.64", c.name, fs)
			}
		})
	}
}

// TestSTL_ShortSeriesHandling ensures STL doesn't crash and seasStrength
// returns 0 (no seasonal evidence) when there are < 2 full cycles.
func TestSTL_ShortSeriesHandling(t *testing.T) {
	y := []float64{1, 2, 3, 4, 5} // length 5, m=12 → less than 1 cycle
	fs, err := seasStrength(y, 12)
	if err != nil {
		t.Fatal(err)
	}
	if fs != 0 {
		t.Errorf("F_s on short series: got %.4f, expected 0", fs)
	}
}

// TestSTL_DefaultTWindow matches R's nextodd(ceil(1.5*m/(1-1.5/sWindow))).
func TestSTL_DefaultTWindow(t *testing.T) {
	cases := []struct {
		m, s, want int
	}{
		// R: tw = nextodd(ceil(1.5*m/(1-1.5/s)))
		// m=12, s=11: 1.5*12 / (1 - 1.5/11) = 18 / 0.8636... = 20.84 → ceil 21 → already odd
		{12, 11, 21},
		// m=7, s=11: 1.5*7 / 0.8636... = 12.16 → ceil 13 → already odd
		{7, 11, 13},
		// m=12, s=7: 1.5*12 / (1 - 1.5/7) = 18 / 0.7857... = 22.91 → ceil 23 → odd
		{12, 7, 23},
	}
	for _, c := range cases {
		got := stlDefaultTWindow(c.m, c.s)
		if got != c.want {
			t.Errorf("stlDefaultTWindow(m=%d, s=%d): got %d, want %d", c.m, c.s, got, c.want)
		}
	}
}
