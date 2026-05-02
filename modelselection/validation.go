package modelselection

import (
	"errors"
	"fmt"
	"math"

	"github.com/hailam/goarima/arima"
	"github.com/hailam/goarima/metrics"
)

// Scorer computes a single scalar score from (yTrue, yPred).
// Lower is "better" by convention, so wrap negation around metrics like R^2.
type Scorer func(yTrue, yPred []float64) (float64, error)

// SMAPEScorer is a Scorer wrapping metrics.SMAPE.
func SMAPEScorer(yTrue, yPred []float64) (float64, error) {
	return metrics.SMAPE(yTrue, yPred)
}

// MAEScorer is a Scorer wrapping metrics.MAE.
func MAEScorer(yTrue, yPred []float64) (float64, error) {
	return metrics.MAE(yTrue, yPred)
}

// MSEScorer is a Scorer wrapping metrics.MSE.
func MSEScorer(yTrue, yPred []float64) (float64, error) {
	return metrics.MSE(yTrue, yPred)
}

// ModelFactory produces a fresh, untrained ARIMA model on each call.
// Used by cross-validators to keep folds independent.
type ModelFactory func() *arima.ARIMA

// CrossValScore runs the splitter, fits a fresh model per fold, scores each
// fold's forecast vs the held-out portion, and returns per-fold scores.
//
// Mirrors pmdarima.model_selection.cross_val_score.
//
//	y    = full series
//	exog = optional exogenous matrix; pass nil if not used
//	cv   = the CV splitter (RollingForecastCV or SlidingWindowForecastCV)
//	mk   = model factory (builds a fresh ARIMA per fold)
//	score = scoring function on (yTrue, yPred)
func CrossValScore(y []float64, exog [][]float64, cv CrossValidator, mk ModelFactory, score Scorer) ([]float64, error) {
	if cv == nil {
		return nil, errors.New("cv must be non-nil")
	}
	if mk == nil {
		return nil, errors.New("model factory must be non-nil")
	}
	if score == nil {
		score = SMAPEScorer
	}
	if exog != nil && len(exog) != len(y) {
		return nil, fmt.Errorf("exog rows (%d) != len(y) (%d)", len(exog), len(y))
	}
	splits, err := cv.Split(len(y))
	if err != nil {
		return nil, err
	}
	if len(splits) == 0 {
		return nil, errors.New("cv produced no splits")
	}
	out := make([]float64, len(splits))
	for i, s := range splits {
		yTrain := pickIdx(y, s.Train)
		yTest := pickIdx(y, s.Test)
		var xTrain, xTest [][]float64
		if exog != nil {
			xTrain = pickRows(exog, s.Train)
			xTest = pickRows(exog, s.Test)
		}
		mdl := mk()
		if mdl == nil {
			return nil, fmt.Errorf("fold %d: nil model from factory", i)
		}
		if err := mdl.Fit(yTrain, xTrain); err != nil {
			return nil, fmt.Errorf("fold %d fit: %w", i, err)
		}
		fc, _, _, err := mdl.Predict(len(yTest), 0, xTest)
		if err != nil {
			return nil, fmt.Errorf("fold %d predict: %w", i, err)
		}
		s, err := score(yTest, fc)
		if err != nil {
			return nil, fmt.Errorf("fold %d score: %w", i, err)
		}
		out[i] = s
	}
	return out, nil
}

// CrossValidateResult holds aggregate CV statistics.
type CrossValidateResult struct {
	Scores []float64
	Mean   float64
	Std    float64
}

// CrossValidate is a convenience wrapper around CrossValScore that returns
// per-fold scores plus their mean and standard deviation.
//
// Mirrors pmdarima.model_selection.cross_validate (simplified).
func CrossValidate(y []float64, exog [][]float64, cv CrossValidator, mk ModelFactory, score Scorer) (*CrossValidateResult, error) {
	scores, err := CrossValScore(y, exog, cv, mk, score)
	if err != nil {
		return nil, err
	}
	mean := 0.0
	for _, v := range scores {
		mean += v
	}
	mean /= float64(len(scores))
	variance := 0.0
	for _, v := range scores {
		variance += (v - mean) * (v - mean)
	}
	if len(scores) > 1 {
		variance /= float64(len(scores) - 1)
	}
	return &CrossValidateResult{
		Scores: scores,
		Mean:   mean,
		Std:    math.Sqrt(variance),
	}, nil
}

// CrossValPredict produces concatenated out-of-fold predictions, aligned to
// the original series indices. Indices not covered by any test fold are
// returned as NaN.
//
// Mirrors pmdarima.model_selection.cross_val_predict.
func CrossValPredict(y []float64, exog [][]float64, cv CrossValidator, mk ModelFactory) ([]float64, error) {
	if cv == nil {
		return nil, errors.New("cv must be non-nil")
	}
	splits, err := cv.Split(len(y))
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(y))
	for i := range out {
		out[i] = math.NaN()
	}
	for i, s := range splits {
		yTrain := pickIdx(y, s.Train)
		var xTrain, xTest [][]float64
		if exog != nil {
			xTrain = pickRows(exog, s.Train)
			xTest = pickRows(exog, s.Test)
		}
		mdl := mk()
		if err := mdl.Fit(yTrain, xTrain); err != nil {
			return nil, fmt.Errorf("fold %d fit: %w", i, err)
		}
		fc, _, _, err := mdl.Predict(len(s.Test), 0, xTest)
		if err != nil {
			return nil, fmt.Errorf("fold %d predict: %w", i, err)
		}
		for k, idx := range s.Test {
			out[idx] = fc[k]
		}
	}
	return out, nil
}

func pickIdx(y []float64, idx []int) []float64 {
	out := make([]float64, len(idx))
	for i, k := range idx {
		out[i] = y[k]
	}
	return out
}

func pickRows(x [][]float64, idx []int) [][]float64 {
	out := make([][]float64, len(idx))
	for i, k := range idx {
		out[i] = append([]float64{}, x[k]...)
	}
	return out
}
