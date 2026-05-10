// Package metrics provides forecasting accuracy metrics.
package metrics

import (
	"errors"
	"math"
)

// SMAPE returns the symmetric mean absolute percentage error
// 100/n * sum(2*|p-a| / (|p|+|a|)).
//
// Mirrors pmdarima.metrics.smape — formula uses 200 numerator and
// (|p|+|a|) denominator; the typical "/2" in the denominator cancels with the 200.
func SMAPE(yTrue, yPred []float64) (float64, error) {
	if len(yTrue) != len(yPred) {
		return 0, errors.New("yTrue and yPred lengths differ")
	}
	if len(yTrue) == 0 {
		return 0, errors.New("empty input")
	}
	sum := 0.0
	n := float64(len(yTrue))
	for i, a := range yTrue {
		p := yPred[i]
		denom := math.Abs(p) + math.Abs(a)
		if denom == 0 {
			// matches numpy nan-behavior in original; we treat as zero error
			continue
		}
		sum += math.Abs(p-a) * 200.0 / denom
	}
	return sum / n, nil
}

// MAE — mean absolute error.
func MAE(yTrue, yPred []float64) (float64, error) {
	if len(yTrue) != len(yPred) {
		return 0, errors.New("yTrue and yPred lengths differ")
	}
	if len(yTrue) == 0 {
		return 0, errors.New("empty input")
	}
	s := 0.0
	for i, a := range yTrue {
		s += math.Abs(yPred[i] - a)
	}
	return s / float64(len(yTrue)), nil
}

// MSE — mean squared error.
func MSE(yTrue, yPred []float64) (float64, error) {
	if len(yTrue) != len(yPred) {
		return 0, errors.New("yTrue and yPred lengths differ")
	}
	if len(yTrue) == 0 {
		return 0, errors.New("empty input")
	}
	s := 0.0
	for i, a := range yTrue {
		d := yPred[i] - a
		s += d * d
	}
	return s / float64(len(yTrue)), nil
}

// RMSE — root mean squared error.
func RMSE(yTrue, yPred []float64) (float64, error) {
	mse, err := MSE(yTrue, yPred)
	if err != nil {
		return 0, err
	}
	return math.Sqrt(mse), nil
}

// MASE — Mean Absolute Scaled Error. Hyndman-Koehler 2006.
//
// MASE = MAE(forecast, actual) / MAE(naive_seasonal, train)
//
// where naive_seasonal is the period-`season` lag of the training set
// (lag-1 differences for season ≤ 1, period-`season` differences for
// seasonal data). Scale-independent — meaningful across series — and
// the standard metric used by the M-competition family.
//
// Returns an error if `train` has fewer than `season+1` observations
// (cannot form the baseline) or if the baseline scale is zero
// (constant training set).
func MASE(yTrue, yPred, train []float64, season int) (float64, error) {
	if len(yTrue) != len(yPred) {
		return 0, errors.New("yTrue and yPred lengths differ")
	}
	if len(yTrue) == 0 {
		return 0, errors.New("empty input")
	}
	step := season
	if step < 1 {
		step = 1
	}
	if len(train) <= step {
		return 0, errors.New("train shorter than seasonal step")
	}
	scale := 0.0
	for i := step; i < len(train); i++ {
		scale += math.Abs(train[i] - train[i-step])
	}
	scale /= float64(len(train) - step)
	if scale == 0 {
		return 0, errors.New("zero baseline scale (constant training set)")
	}
	mae := 0.0
	for i, a := range yTrue {
		mae += math.Abs(yPred[i] - a)
	}
	mae /= float64(len(yTrue))
	return mae / scale, nil
}

// MASEScoring returns a Scoring-compatible closure with the training
// set and seasonality baked in. Designed for AutoArimaOpts.Scoring
// when OutOfSampleSize > 0:
//
//	opts.Scoring = metrics.MASEScoring(y[:len(y)-opts.OutOfSampleSize], 12)
func MASEScoring(train []float64, season int) func(yTrue, yPred []float64) (float64, error) {
	// Pre-compute the baseline scale so each candidate evaluation only
	// pays for the MAE on the holdout.
	step := season
	if step < 1 {
		step = 1
	}
	scale := 0.0
	if len(train) > step {
		for i := step; i < len(train); i++ {
			scale += math.Abs(train[i] - train[i-step])
		}
		scale /= float64(len(train) - step)
	}
	return func(yTrue, yPred []float64) (float64, error) {
		if len(yTrue) != len(yPred) {
			return 0, errors.New("yTrue and yPred lengths differ")
		}
		if len(yTrue) == 0 {
			return 0, errors.New("empty input")
		}
		if scale == 0 {
			return 0, errors.New("zero baseline scale (constant training set)")
		}
		mae := 0.0
		for i, a := range yTrue {
			mae += math.Abs(yPred[i] - a)
		}
		return (mae / float64(len(yTrue))) / scale, nil
	}
}
