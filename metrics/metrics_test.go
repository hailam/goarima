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

func TestRMSE(t *testing.T) {
	got, err := RMSE([]float64{1, 2, 3}, []float64{2, 4, 6})
	if err != nil {
		t.Fatal(err)
	}
	want := math.Sqrt((1.0 + 4.0 + 9.0) / 3.0)
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("RMSE got %v want %v", got, want)
	}
}

func TestMASE(t *testing.T) {
	// Linear-trend training set: scale = 1 (lag-1 absolute difference).
	train := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	yTrue := []float64{9, 10, 11}
	yPred := []float64{9.5, 10.5, 11.5}
	got, err := MASE(yTrue, yPred, train, 1)
	if err != nil {
		t.Fatal(err)
	}
	// MAE = 0.5; scale = 1; MASE = 0.5
	if math.Abs(got-0.5) > 1e-12 {
		t.Errorf("MASE got %v want 0.5", got)
	}

	// Constant training → zero scale → error.
	if _, err := MASE(yTrue, yPred, []float64{5, 5, 5, 5, 5}, 1); err == nil {
		t.Error("expected zero-scale error on constant train")
	}
	// Train shorter than seasonal step.
	if _, err := MASE(yTrue, yPred, []float64{1, 2}, 12); err == nil {
		t.Error("expected error when train < season+1")
	}
}

func TestMASEScoring(t *testing.T) {
	train := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	score := MASEScoring(train, 1)
	got, err := score([]float64{9, 10, 11}, []float64{9.5, 10.5, 11.5})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-0.5) > 1e-12 {
		t.Errorf("MASEScoring got %v want 0.5", got)
	}
	// Same answer as the static MASE call.
	staticGot, _ := MASE([]float64{9, 10, 11}, []float64{9.5, 10.5, 11.5}, train, 1)
	if math.Abs(got-staticGot) > 1e-12 {
		t.Errorf("MASEScoring %v != static MASE %v", got, staticGot)
	}
}
