package modelselection

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/hailam/goarima/arima"
)

// CrossValScore on AR(1) data: rolling CV should produce finite SMAPE scores.
func TestCrossValScoreAR1(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	y := make([]float64, 200)
	for i := 1; i < len(y); i++ {
		y[i] = 0.6*y[i-1] + rng.NormFloat64()
	}
	cv, _ := NewRollingForecastCV(1, 5, 100)
	mk := func() *arima.ARIMA {
		m := arima.NewARIMA(arima.Order{P: 1, D: 0, Q: 0})
		m.MaxIter = 30
		return m
	}
	scores, err := CrossValScore(y, nil, cv, mk, MAEScorer)
	if err != nil {
		t.Fatal(err)
	}
	if len(scores) == 0 {
		t.Fatal("no scores")
	}
	for i, s := range scores {
		if math.IsNaN(s) || math.IsInf(s, 0) || s < 0 {
			t.Errorf("score[%d] = %v", i, s)
		}
	}
}

func TestCrossValidateAggregates(t *testing.T) {
	y := make([]float64, 100)
	for i := range y {
		y[i] = float64(i) // monotone trend
	}
	cv, _ := NewSlidingWindowForecastCV(1, 5, 30)
	mk := func() *arima.ARIMA {
		m := arima.NewARIMA(arima.Order{P: 0, D: 1, Q: 0})
		return m
	}
	res, err := CrossValidate(y, nil, cv, mk, MSEScorer)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(res.Mean) {
		t.Error("mean is NaN")
	}
	if len(res.Scores) == 0 {
		t.Error("no scores")
	}
	// for a perfect linear trend with d=1 model, MSE should be tiny
	if res.Mean > 1.0 {
		t.Errorf("mean MSE=%v unexpectedly large", res.Mean)
	}
}

// CrossValPredict: every test index should be filled, others NaN.
func TestCrossValPredict(t *testing.T) {
	y := make([]float64, 80)
	for i := range y {
		y[i] = float64(i)
	}
	cv, _ := NewRollingForecastCV(1, 1, 60)
	mk := func() *arima.ARIMA {
		return arima.NewARIMA(arima.Order{P: 0, D: 1, Q: 0})
	}
	pred, err := CrossValPredict(y, nil, cv, mk)
	if err != nil {
		t.Fatal(err)
	}
	if len(pred) != 80 {
		t.Errorf("len=%d", len(pred))
	}
	// Indices 0..59 untouched (NaN); 60..79 should have predictions.
	for i := 0; i < 60; i++ {
		if !math.IsNaN(pred[i]) {
			t.Errorf("pred[%d]=%v expected NaN", i, pred[i])
		}
	}
	for i := 60; i < 80; i++ {
		if math.IsNaN(pred[i]) {
			t.Errorf("pred[%d] is NaN", i)
		}
	}
}

// CD-N4: CrossValPredict must guard against nil ModelFactory and
// length-mismatched exog (matching CrossValScoreConcurrent).
func TestCrossValPredictGuards(t *testing.T) {
	y := make([]float64, 50)
	for i := range y {
		y[i] = float64(i)
	}
	cv, _ := NewRollingForecastCV(1, 1, 30)
	mk := func() *arima.ARIMA { return arima.NewARIMA(arima.Order{P: 0, D: 1, Q: 0}) }

	// nil cv → error
	if _, err := CrossValPredict(y, nil, nil, mk); err == nil {
		t.Error("expected error for nil cv")
	}
	// nil mk → error
	if _, err := CrossValPredict(y, nil, cv, nil); err == nil {
		t.Error("expected error for nil ModelFactory")
	}
	// short exog → error (was: panic in pickRows)
	short := make([][]float64, 10)
	for i := range short {
		short[i] = []float64{1}
	}
	if _, err := CrossValPredict(y, short, cv, mk); err == nil {
		t.Error("expected error for length-mismatched exog")
	}
}
