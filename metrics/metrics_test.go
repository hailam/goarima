package metrics

import (
	"math"
	"testing"
)

func TestSMAPE(t *testing.T) {
	yTrue := []float64{0.07533, 0.07533, 0.07533, 0.07533, 0.07533, 0.07533, 0.0672, 0.0672}
	yPred := []float64{0.102, 0.107, 0.047, 0.1, 0.032, 0.047, 0.108, 0.089}
	want := 42.60306631890196
	got, err := SMAPE(yTrue, yPred)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-want) > 1e-8 {
		t.Errorf("SMAPE got %.12f want %.12f", got, want)
	}

	// perfect match
	got, err = SMAPE(yTrue, yTrue)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Errorf("SMAPE perfect got %v, want 0", got)
	}
}

func TestMAE(t *testing.T) {
	got, err := MAE([]float64{1, 2, 3}, []float64{2, 4, 6})
	if err != nil {
		t.Fatal(err)
	}
	want := (1.0 + 2.0 + 3.0) / 3.0
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("MAE got %v want %v", got, want)
	}
}

func TestMSE(t *testing.T) {
	got, err := MSE([]float64{1, 2, 3}, []float64{2, 4, 6})
	if err != nil {
		t.Fatal(err)
	}
	want := (1.0 + 4.0 + 9.0) / 3.0
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("MSE got %v want %v", got, want)
	}
}

func TestLengthMismatch(t *testing.T) {
	if _, err := SMAPE([]float64{1, 2}, []float64{1}); err == nil {
		t.Error("expected length mismatch error")
	}
}
