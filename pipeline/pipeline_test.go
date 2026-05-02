package pipeline

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/hailam/goarima/arima"
	"github.com/hailam/goarima/preprocessing"
)

// Pipeline with a Log transformer + ARIMA must round-trip onto the original scale.
func TestPipelineLogPlusARIMA(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 43))
	n := 100
	y := make([]float64, n)
	for i := range y {
		y[i] = math.Exp(0.5 + 0.01*float64(i) + 0.1*rng.NormFloat64())
	}
	logT := preprocessing.NewLogEndogTransformer(0, preprocessing.NegRaise, 1e-16)
	model := arima.NewARIMA(arima.Order{P: 1, D: 0, Q: 0})
	model.MaxIter = 60
	pl, err := NewPipeline([]Step{
		{Name: "log", Endog: logT},
	}, model)
	if err != nil {
		t.Fatal(err)
	}
	if err := pl.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	fc, _, _, err := pl.Predict(5, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(fc) != 5 {
		t.Errorf("forecast len=%d", len(fc))
	}
	// All forecasts must be positive (since we exp'd them back).
	for i, v := range fc {
		if v <= 0 {
			t.Errorf("fc[%d]=%v should be positive", i, v)
		}
	}
}

// Validate that names must be unique and non-empty.
func TestPipelineValidation(t *testing.T) {
	model := arima.NewARIMA(arima.Order{P: 1, D: 0, Q: 0})
	if _, err := NewPipeline(nil, nil); err == nil {
		t.Error("expected error for nil model")
	}
	bc := preprocessing.NewBoxCoxEndogTransformer()
	steps := []Step{
		{Name: "a", Endog: bc},
		{Name: "a", Endog: bc},
	}
	if _, err := NewPipeline(steps, model); err == nil {
		t.Error("expected duplicate-name error")
	}
	if _, err := NewPipeline([]Step{{Name: ""}}, model); err == nil {
		t.Error("expected empty-name error")
	}
	if _, err := NewPipeline([]Step{{Name: "x"}}, model); err == nil {
		t.Error("expected error: must set Endog or Exog")
	}
}
