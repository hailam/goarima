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
