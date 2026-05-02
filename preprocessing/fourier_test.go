package preprocessing

import (
	"testing"
)

// Mirrors test_r_equivalency from pmdarima — the expected matrix below is the
// one R's forecast::fourier(y, K=2) produces for m=5, repeated 4 times.
func TestFourierREquivalency(t *testing.T) {
	y := make([]float64, 20) // values irrelevant; only length is used
	expected := [][]float64{
		{0.9510565, 0.309017, 0.5877853, -0.809017},
		{0.5877853, -0.809017, -0.9510565, 0.309017},
		{-0.5877853, -0.809017, 0.9510565, 0.309017},
		{-0.9510565, 0.309017, -0.5877853, -0.809017},
		{0.0000000, 1.000000, 0.0000000, 1.000000},
		{0.9510565, 0.309017, 0.5877853, -0.809017},
		{0.5877853, -0.809017, -0.9510565, 0.309017},
		{-0.5877853, -0.809017, 0.9510565, 0.309017},
		{-0.9510565, 0.309017, -0.5877853, -0.809017},
		{0.0000000, 1.000000, 0.0000000, 1.000000},
		{0.9510565, 0.309017, 0.5877853, -0.809017},
		{0.5877853, -0.809017, -0.9510565, 0.309017},
		{-0.5877853, -0.809017, 0.9510565, 0.309017},
		{-0.9510565, 0.309017, -0.5877853, -0.809017},
		{0.0000000, 1.000000, 0.0000000, 1.000000},
		{0.9510565, 0.309017, 0.5877853, -0.809017},
		{0.5877853, -0.809017, -0.9510565, 0.309017},
		{-0.5877853, -0.809017, 0.9510565, 0.309017},
		{-0.9510565, 0.309017, -0.5877853, -0.809017},
		{0.0000000, 1.000000, 0.0000000, 1.000000},
	}

	tr := NewFourierFeaturizer(5, 2)
	mat, err := tr.FitTransform(y, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(mat) != 20 || len(mat[0]) != 4 {
		t.Fatalf("shape: got %dx%d want 20x4", len(mat), len(mat[0]))
	}
	for i, row := range mat {
		if !almostEqual(row, expected[i], 1e-6) {
			t.Errorf("row %d got %v want %v", i, row, expected[i])
		}
	}
}

func TestFourierWithExog(t *testing.T) {
	y := make([]float64, 20)
	x := make([][]float64, 20)
	for i := range x {
		x[i] = []float64{float64(i), float64(i + 1), float64(i + 2)}
	}
	tr := NewFourierFeaturizer(5, 2)
	if err := tr.Fit(y); err != nil {
		t.Fatal(err)
	}
	mat, err := tr.Transform(y, x, 0)
	if err != nil {
		t.Fatal(err)
	}
	// First 3 columns are X
	for i := range mat {
		if !almostEqual(mat[i][:3], x[i], 1e-12) {
			t.Errorf("row %d X-cols got %v want %v", i, mat[i][:3], x[i])
		}
	}
}

func TestFourierBadK(t *testing.T) {
	tr := NewFourierFeaturizer(12, 8) // 2*8 = 16 > 12
	if err := tr.Fit(make([]float64, 5)); err == nil {
		t.Error("expected error for k > m/2")
	}
}

func TestFourierNPeriodsXMismatch(t *testing.T) {
	y := make([]float64, 20)
	tr := NewFourierFeaturizer(5, 2)
	if err := tr.Fit(y); err != nil {
		t.Fatal(err)
	}
	bad := make([][]float64, 5)
	for i := range bad {
		bad[i] = []float64{0, 0, 0}
	}
	if _, err := tr.Transform(y, bad, 2); err == nil {
		t.Error("expected error: nPeriods != len(X)")
	}
}

func TestFourierUpdateAndTransform(t *testing.T) {
	n := 150
	y := make([]float64, n)
	tr := NewFourierFeaturizer(10, 5)
	if err := tr.Fit(y[:100]); err != nil {
		t.Fatal(err)
	}
	if _, err := tr.Transform(y[:100], nil, 0); err != nil {
		t.Fatal(err)
	}
	xt, err := tr.UpdateAndTransform(y[100:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(xt) != 50 || tr.n != 150 {
		t.Errorf("update: %d rows, n=%d", len(xt), tr.n)
	}

	// Vanilla transform of full y should split into [first 100, last 50] matching prior outputs.
	full, err := tr.Transform(y, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(full) != 150 {
		t.Errorf("full transform len = %d", len(full))
	}
	for i := 100; i < 150; i++ {
		if !almostEqual(full[i], xt[i-100], 1e-12) {
			t.Errorf("row %d mismatch", i)
		}
	}
}

func TestFourierFeatureNames(t *testing.T) {
	tr := NewFourierFeaturizer(12, 2)
	if err := tr.Fit(make([]float64, 24)); err != nil {
		t.Fatal(err)
	}
	names := tr.FeatureNames()
	want := []string{"FOURIER_S12-0", "FOURIER_C12-0", "FOURIER_S12-1", "FOURIER_C12-1"}
	if len(names) != len(want) {
		t.Fatalf("names: %v want %v", names, want)
	}
	for i, n := range names {
		if n != want[i] {
			t.Errorf("name[%d] got %s want %s", i, n, want[i])
		}
	}
}
