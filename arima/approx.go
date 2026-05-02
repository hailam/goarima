package arima

import (
	"errors"
	"math"
	"sort"
)

// ApproxRule controls extrapolation behavior.
type ApproxRule int

const (
	// RuleNaN returns NaN outside [min(x), max(x)].
	RuleNaN ApproxRule = 1
	// RuleClip extrapolates as the closest endpoint value.
	RuleClip ApproxRule = 2
)

// ApproxMethod selects the interpolation method.
type ApproxMethod int

const (
	// Linear interpolation.
	Linear ApproxMethod = iota
	// Constant interpolation.
	Constant
)

// Approx performs R-style linear (or constant) interpolation.
//
// xRef and yRef are the reference points. xOut is the locations to interpolate at.
// Mirrors pmdarima.arima.approx.approx (with ties='mean' regularization).
func Approx(xRef, yRef, xOut []float64, method ApproxMethod, rule ApproxRule) ([]float64, error) {
	if len(xRef) != len(yRef) {
		return nil, errors.New("xRef and yRef must have equal length")
	}
	if len(xRef) == 0 {
		return nil, errors.New("xRef must be non-empty")
	}
	xs, ys := regularizeMean(xRef, yRef)
	yLeft := math.NaN()
	yRight := math.NaN()
	if rule == RuleClip {
		yLeft = ys[0]
		yRight = ys[len(ys)-1]
	}
	out := make([]float64, len(xOut))
	for i, v := range xOut {
		if math.IsNaN(v) {
			out[i] = math.NaN()
			continue
		}
		out[i] = approx1(v, xs, ys, yLeft, yRight, method, 1, 0)
	}
	return out, nil
}

func regularizeMean(x, y []float64) ([]float64, []float64) {
	n := len(x)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool { return x[idx[i]] < x[idx[j]] })
	xs := make([]float64, n)
	ys := make([]float64, n)
	for i, k := range idx {
		xs[i] = x[k]
		ys[i] = y[k]
	}
	// collapse duplicates by mean
	i := 0
	out := 0
	for i < n {
		j := i
		s := 0.0
		c := 0
		for j < n && xs[j] == xs[i] {
			s += ys[j]
			c++
			j++
		}
		xs[out] = xs[i]
		ys[out] = s / float64(c)
		out++
		i = j
	}
	return xs[:out], ys[:out]
}

// approx1 mirrors the C reference implementation.
func approx1(v float64, x, y []float64, yLow, yHigh float64, method ApproxMethod, f1, f2 float64) float64 {
	n := len(x)
	if n == 0 {
		return math.NaN()
	}
	if v < x[0] {
		return yLow
	}
	if v > x[n-1] {
		return yHigh
	}
	i := 0
	j := n - 1
	for i < j-1 {
		ij := (i + j) / 2
		if v < x[ij] {
			j = ij
		} else {
			i = ij
		}
	}
	if v == x[j] {
		return y[j]
	}
	if v == x[i] {
		return y[i]
	}
	if method == Linear {
		return y[i] + (y[j]-y[i])*((v-x[i])/(x[j]-x[i]))
	}
	left := 0.0
	if f1 != 0 {
		left = y[i] * f1
	}
	right := 0.0
	if f2 != 0 {
		right = y[j] * f2
	}
	return left + right
}
