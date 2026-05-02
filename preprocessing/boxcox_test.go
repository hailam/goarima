package preprocessing

import (
	"math"
	"testing"
)

// helper to compare slices
func almostEqual(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
}

// Mirrors test_invertible — a fitted BoxCox should round trip to original.
func TestBoxCoxInvertible(t *testing.T) {
	// generate loggamma-like positive data deterministically
	y := make([]float64, 200)
	for i := range y {
		y[i] = math.Log(float64(i+1)) + 5.0 + 0.01*float64(i%7)
	}
	tr := NewBoxCoxEndogTransformer()
	yt, err := tr.FitTransform(y)
	if err != nil {
		t.Fatal(err)
	}
	yp, err := tr.InverseTransform(yt)
	if err != nil {
		t.Fatal(err)
	}
	if !almostEqual(y, yp, 1e-6) {
		t.Errorf("BoxCox round-trip failed: max diff %v", maxDiff(y, yp))
	}
}

// Mirrors test_invertible_when_lambda_is_0.
func TestBoxCoxLambdaZero(t *testing.T) {
	zero := 0.0
	tr := &BoxCoxEndogTransformer{Lmbda: &zero, NegAction: NegRaise, Floor: 1e-16}
	y := []float64{1, 2, 3}
	yt, err := tr.FitTransform(y)
	if err != nil {
		t.Fatal(err)
	}
	yp, err := tr.InverseTransform(yt)
	if err != nil {
		t.Fatal(err)
	}
	if !almostEqual(y, yp, 1e-12) {
		t.Errorf("lambda=0 round trip failed: %v -> %v -> %v", y, yt, yp)
	}
}

// Mirrors test_value_error_on_neg_lambda.
func TestBoxCoxNegLambda2(t *testing.T) {
	tr := NewBoxCoxEndogTransformer()
	tr.Lmbda2 = -4.0
	if _, err := tr.FitTransform([]float64{1, 2, 3}); err == nil {
		t.Error("expected error for lmbda2 < 0")
	}
}

// Mirrors TestNonInvertibleBC.test_expected_error.
func TestBoxCoxNegRaise(t *testing.T) {
	two := 2.0
	tr := &BoxCoxEndogTransformer{Lmbda: &two, NegAction: NegRaise, Floor: 1e-16}
	if _, err := tr.FitTransform([]float64{-1, 0, 1}); err == nil {
		t.Error("expected error for negatives in y")
	}
}

// Mirrors TestNonInvertibleBC.test_expected_warning.
func TestBoxCoxNegWarn(t *testing.T) {
	two := 2.0
	called := false
	tr := &BoxCoxEndogTransformer{
		Lmbda:     &two,
		NegAction: NegWarn,
		Floor:     1e-16,
		OnWarn:    func(string) { called = true },
	}
	y := []float64{-1, 0, 1}
	yt, err := tr.FitTransform(y)
	if err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Error("expected warn callback")
	}
	yp, _ := tr.InverseTransform(yt)
	if almostEqual(yp, y, 1e-6) {
		t.Error("non-invertible case should not round-trip exactly")
	}
}

// Mirrors TestNonInvertibleBC.test_invertible_when_lam2.
func TestBoxCoxInvertibleWhenLam2(t *testing.T) {
	two := 2.0
	tr := &BoxCoxEndogTransformer{Lmbda: &two, Lmbda2: 2.0, NegAction: NegRaise, Floor: 1e-16}
	y := []float64{-1, 0, 1}
	yt, err := tr.FitTransform(y)
	if err != nil {
		t.Fatal(err)
	}
	yp, err := tr.InverseTransform(yt)
	if err != nil {
		t.Fatal(err)
	}
	if !almostEqual(y, yp, 1e-9) {
		t.Errorf("lam2=2 should preserve invertibility: %v -> %v -> %v", y, yt, yp)
	}
}

// Log transformer.
func TestLogTransformer(t *testing.T) {
	tr := NewLogEndogTransformer(0, NegRaise, 1e-16)
	y := []float64{1, math.E, math.E * math.E}
	yt, err := tr.FitTransform(y)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{0, 1, 2}
	if !almostEqual(yt, want, 1e-12) {
		t.Errorf("log got %v want %v", yt, want)
	}
	yp, _ := tr.InverseTransform(yt)
	if !almostEqual(yp, y, 1e-9) {
		t.Errorf("log inverse got %v want %v", yp, y)
	}
}

func maxDiff(a, b []float64) float64 {
	m := 0.0
	for i := range a {
		d := math.Abs(a[i] - b[i])
		if d > m {
			m = d
		}
	}
	return m
}
