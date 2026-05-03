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
	fc, _, _, err := pl.Predict(5, 0, nil)
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

// A pipeline with no exog featurizer must still pass user-supplied
// futureExog through to the underlying model. Pre-fix the predict path
// silently dropped futureExog when len(p.exogChain) == 0, even when the
// model was fitted with raw exog.
func TestPipelineNoExogFeaturizer_FutureExogPasses(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 99))
	n := 120
	y := make([]float64, n)
	x := make([][]float64, n)
	for i := 0; i < n; i++ {
		x[i] = []float64{rng.Float64()}
		var prev float64
		if i > 0 {
			prev = y[i-1]
		}
		y[i] = 0.5*prev + 1.5*x[i][0] + rng.NormFloat64()
	}
	model := arima.NewARIMA(arima.Order{P: 1, D: 0, Q: 0})
	model.WithIntercept = true
	pl, err := NewPipeline(nil, model) // no transformer steps
	if err != nil {
		t.Fatal(err)
	}
	if err := pl.Fit(y, x); err != nil {
		t.Fatal(err)
	}
	futureX := make([][]float64, 5)
	for i := range futureX {
		futureX[i] = []float64{rng.Float64()}
	}
	fc, _, _, err := pl.Predict(5, 0, futureX)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if len(fc) != 5 {
		t.Errorf("got %d forecasts, want 5", len(fc))
	}
	for i, v := range fc {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("fc[%d] = %v (NaN/Inf)", i, v)
		}
	}
	// Sanity: passing nil futureExog should now error (model was fit with
	// exog, so raw exog must be supplied).
	if _, _, _, err := pl.Predict(5, 0, nil); err == nil {
		t.Error("expected error when fitted-with-exog model gets no futureExog")
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
