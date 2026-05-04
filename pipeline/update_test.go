package pipeline

import (
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/hailam/goarima/arima"
	"github.com/hailam/goarima/preprocessing"
)

// CD-F1: Pipeline.Update appends new data, applies the transformer chain
// without re-fitting transformer state, and warm-starts the underlying
// model. This test covers a Box-Cox-only pipeline (endog transformer).
func TestPipelineUpdate_LogPlusARIMA(t *testing.T) {
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
	if err := pl.Fit(y[:80], nil); err != nil {
		t.Fatal(err)
	}
	beforeParams := pl.Model.Params()
	beforePipelineLen := len(pl.yTrain)

	// Append the last 20 observations.
	if err := pl.Update(y[80:], nil); err != nil {
		t.Fatalf("Update: %v", err)
	}
	// Pipeline's own original-scale yTrain cache must be extended by exactly 20.
	if len(pl.yTrain) != beforePipelineLen+20 {
		t.Errorf("pipeline yTrain len = %d, want %d", len(pl.yTrain), beforePipelineLen+20)
	}
	// Predict must still work and produce positive forecasts (log inverted).
	fc, _, _, err := pl.Predict(5, 0, nil)
	if err != nil {
		t.Fatalf("Predict after Update: %v", err)
	}
	for i, v := range fc {
		if v <= 0 {
			t.Errorf("fc[%d] = %g, want positive", i, v)
		}
	}
	// Param values should have shifted slightly from warm-start refresh.
	afterParams := pl.Model.Params()
	if len(beforeParams) != len(afterParams) {
		t.Fatal("params changed shape across Update")
	}
	moved := false
	for i := range beforeParams {
		if math.Abs(afterParams[i]-beforeParams[i]) > 1e-6 {
			moved = true
			break
		}
	}
	if !moved {
		t.Error("params identical pre/post Update — warm-start didn't refresh")
	}
}

// CD-F1: Pipeline.Refit on the same pipeline must produce results within
// optimizer tolerance of a fresh Fit on the combined series.
func TestPipelineRefit_MatchesFreshFit(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	n := 100
	y := make([]float64, n)
	for i := range y {
		y[i] = math.Exp(0.4 + 0.005*float64(i) + 0.1*rng.NormFloat64())
	}

	// Pipeline 1: Fit on first 60, Refit on remaining 40.
	logT1 := preprocessing.NewLogEndogTransformer(0, preprocessing.NegRaise, 1e-16)
	m1 := arima.NewARIMA(arima.Order{P: 1, D: 0, Q: 0})
	m1.MaxIter = 80
	pl1, err := NewPipeline([]Step{{Name: "log", Endog: logT1}}, m1)
	if err != nil {
		t.Fatal(err)
	}
	if err := pl1.Fit(y[:60], nil); err != nil {
		t.Fatal(err)
	}
	if err := pl1.Refit(y[60:], nil); err != nil {
		t.Fatal(err)
	}

	// Pipeline 2: Fit on full 100 from scratch.
	logT2 := preprocessing.NewLogEndogTransformer(0, preprocessing.NegRaise, 1e-16)
	m2 := arima.NewARIMA(arima.Order{P: 1, D: 0, Q: 0})
	m2.MaxIter = 80
	pl2, err := NewPipeline([]Step{{Name: "log", Endog: logT2}}, m2)
	if err != nil {
		t.Fatal(err)
	}
	if err := pl2.Fit(y, nil); err != nil {
		t.Fatal(err)
	}

	p1 := pl1.Model.Params()
	p2 := pl2.Model.Params()
	for i := range p1 {
		if math.Abs(p1[i]-p2[i]) > 1e-3 {
			t.Errorf("param[%d]: Refit-after-Fit=%g vs fresh Fit=%g (diff %g)",
				i, p1[i], p2[i], p1[i]-p2[i])
		}
	}
}

// CD-F1: Update through an exog featurizer (PipelineDateFeaturizer)
// extends the date index and re-emits features for the combined range.
func TestPipelineUpdate_WithDateFeaturizer(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 32))
	n := 80
	start := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	dates := make([]time.Time, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		dates[i] = start.AddDate(0, 0, i)
		var prev float64
		if i > 0 {
			prev = y[i-1]
		}
		y[i] = 0.4*prev + rng.NormFloat64()
	}

	feat := preprocessing.NewPipelineDateFeaturizer(dates[:60], preprocessing.DailyStep)
	model := arima.NewARIMA(arima.Order{P: 1, D: 0, Q: 0})
	model.WithIntercept = true
	pl, err := NewPipeline([]Step{{Name: "dates", Exog: feat}}, model)
	if err != nil {
		t.Fatal(err)
	}
	if err := pl.Fit(y[:60], nil); err != nil {
		t.Fatal(err)
	}
	if err := pl.Update(y[60:], nil); err != nil {
		t.Fatalf("Update: %v", err)
	}
	// Featurizer's date index should have grown by 20.
	if len(feat.FitDates) != 80 {
		t.Errorf("FitDates len = %d, want 80", len(feat.FitDates))
	}
	// Predict 7 days ahead — featurizer should auto-extend dates beyond
	// the new tail.
	fc, _, _, err := pl.Predict(7, 0, nil)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if len(fc) != 7 {
		t.Errorf("len(fc) = %d, want 7", len(fc))
	}
	for i, v := range fc {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("fc[%d] = %v", i, v)
		}
	}
}

// CD-F1: Update on an unfitted pipeline must error.
func TestPipelineUpdate_Unfitted(t *testing.T) {
	model := arima.NewARIMA(arima.Order{P: 1, D: 0, Q: 0})
	pl, _ := NewPipeline(nil, model)
	if err := pl.Update([]float64{1, 2, 3}, nil); err == nil {
		t.Error("Update on unfitted pipeline should error")
	}
	if err := pl.Refit([]float64{1, 2, 3}, nil); err == nil {
		t.Error("Refit on unfitted pipeline should error")
	}
}

// CD-F1: empty newY must error.
func TestPipelineUpdate_EmptyNewY(t *testing.T) {
	rng := rand.New(rand.NewPCG(99, 100))
	n := 50
	y := make([]float64, n)
	for i := range y {
		y[i] = math.Exp(rng.NormFloat64())
	}
	logT := preprocessing.NewLogEndogTransformer(0, preprocessing.NegRaise, 1e-16)
	model := arima.NewARIMA(arima.Order{P: 1, D: 0, Q: 0})
	pl, _ := NewPipeline([]Step{{Name: "log", Endog: logT}}, model)
	if err := pl.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	if err := pl.Update([]float64{}, nil); err == nil {
		t.Error("Update with empty newY should error")
	}
	if err := pl.Refit([]float64{}, nil); err == nil {
		t.Error("Refit with empty newY should error")
	}
}
