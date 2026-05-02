package arima

import (
	"math"
	"testing"

	"github.com/hailam/goarima/datasets"
)

// Mirrors test_decompose_happy_path: at indices [m/2 : -m/2] the
// decomposition components must reconstruct the input.
func TestDecomposeHappyPathAdditive(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6}
	d, err := Decompose(x, Additive, 3, nil)
	if err != nil {
		t.Fatal(err)
	}
	first := 3 / 2
	last := len(x) - 3/2
	for i := first; i < last; i++ {
		got := d.Trend[i] + d.Seasonal[i] + d.Random[i]
		if math.IsNaN(got) {
			continue
		}
		if math.Abs(got-x[i]) > 1e-9 {
			t.Errorf("reconstruction at %d: %v vs %v", i, got, x[i])
		}
	}
}

func TestDecomposeMultiplicative(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	d, err := Decompose(ap, Multiplicative, 12, nil)
	if err != nil {
		t.Fatal(err)
	}
	first := 6
	last := len(ap) - 6
	for i := first; i < last; i++ {
		got := d.Trend[i] * d.Seasonal[i] * d.Random[i]
		if math.IsNaN(got) {
			continue
		}
		if math.Abs(got-ap[i])/ap[i] > 1e-6 {
			t.Errorf("reconstruction at %d: %v vs %v", i, got, ap[i])
		}
	}
}

// Mirrors test_decompose_corner_cases.
func TestDecomposeCornerCases(t *testing.T) {
	if _, err := Decompose([]float64{1, 2, 3, 4}, 5, 4, nil); err == nil {
		t.Error("expected error: bad type")
	}
	if _, err := Decompose([]float64{1, 2}, Multiplicative, 1, nil); err == nil {
		t.Error("expected error: m<2")
	}
	if _, err := Decompose([]float64{1}, Multiplicative, 4, nil); err == nil {
		t.Error("expected error: too short")
	}
}
