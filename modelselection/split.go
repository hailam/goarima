// Package modelselection provides time series cross-validation splitters and
// utilities ported from pmdarima.model_selection.
package modelselection

import (
	"errors"
	"fmt"
)

// Split is a (train, test) pair of integer indices into the original series.
type Split struct {
	Train []int
	Test  []int
}

// CrossValidator yields train/test index folds for a series of length n.
type CrossValidator interface {
	Split(n int) ([]Split, error)
	Horizon() int
	Step() int
}

// TrainTestSplit splits indices [0,n) sequentially into train/test.
//
// If testSize > 0, that many samples go to test (or testFrac if 0<frac<1).
// If trainSize > 0, that many go to train.
// If both zero, defaults to 0.25 test fraction (sklearn convention).
func TrainTestSplit(n int, testSize int, testFrac float64, trainSize int) (train, test []int, err error) {
	if n <= 0 {
		return nil, nil, errors.New("n must be > 0")
	}
	var k int
	switch {
	case testSize > 0:
		k = testSize
	case testFrac > 0 && testFrac < 1:
		k = int(float64(n) * testFrac)
	case trainSize > 0:
		k = n - trainSize
	default:
		k = n / 4 // 0.25
	}
	if k <= 0 || k >= n {
		return nil, nil, fmt.Errorf("invalid test size %d for n=%d", k, n)
	}
	train = make([]int, n-k)
	test = make([]int, k)
	for i := 0; i < n-k; i++ {
		train[i] = i
	}
	for i := 0; i < k; i++ {
		test[i] = n - k + i
	}
	return
}

// RollingForecastCV grows the training set by step each fold while predicting
// h periods ahead.
type RollingForecastCV struct {
	H       int // forecast horizon
	StepSz  int // training-set growth per fold
	Initial int // initial training size; 0 = max(1, n/3)
}

// NewRollingForecastCV returns a CV with default h=1, step=1, initial=auto.
func NewRollingForecastCV(h, step, initial int) (*RollingForecastCV, error) {
	if h < 1 {
		return nil, errors.New("h must be a positive value")
	}
	if step < 1 {
		return nil, errors.New("step must be a positive value")
	}
	return &RollingForecastCV{H: h, StepSz: step, Initial: initial}, nil
}

// Horizon returns h.
func (r *RollingForecastCV) Horizon() int { return r.H }

// Step returns step.
func (r *RollingForecastCV) Step() int { return r.StepSz }

// Split yields rolling-origin folds.
func (r *RollingForecastCV) Split(n int) ([]Split, error) {
	initial := r.Initial
	if initial > 0 {
		if initial+r.H > n {
			return nil, errors.New("initial training size + h exceeds series length")
		}
	} else {
		initial = n / 3
		if initial < 1 {
			initial = 1
		}
	}
	var splits []Split
	windowEnd := initial
	for windowEnd+r.H <= n {
		train := makeRange(0, windowEnd)
		test := makeRange(windowEnd, windowEnd+r.H)
		splits = append(splits, Split{Train: train, Test: test})
		windowEnd += r.StepSz
	}
	return splits, nil
}

// SlidingWindowForecastCV slides a fixed-size window of training observations.
type SlidingWindowForecastCV struct {
	H          int
	StepSz     int
	WindowSize int // 0 = max(3, n/5)
}

// NewSlidingWindowForecastCV constructs a sliding-window CV.
func NewSlidingWindowForecastCV(h, step, windowSize int) (*SlidingWindowForecastCV, error) {
	if h < 1 {
		return nil, errors.New("h must be a positive value")
	}
	if step < 1 {
		return nil, errors.New("step must be a positive value")
	}
	return &SlidingWindowForecastCV{H: h, StepSz: step, WindowSize: windowSize}, nil
}

// Horizon returns h.
func (s *SlidingWindowForecastCV) Horizon() int { return s.H }

// Step returns step.
func (s *SlidingWindowForecastCV) Step() int { return s.StepSz }

// Split yields sliding-window folds.
func (s *SlidingWindowForecastCV) Split(n int) ([]Split, error) {
	w := s.WindowSize
	if w > 0 {
		if w+s.H > n {
			return nil, errors.New("window_size + h would exceed series length")
		}
	} else {
		w = n / 5
		if w < 3 {
			w = 3
		}
	}
	if w < 3 {
		return nil, errors.New("window_size must be > 2")
	}
	var splits []Split
	start := 0
	for {
		end := start + w
		if end+s.H > n {
			break
		}
		splits = append(splits, Split{
			Train: makeRange(start, end),
			Test:  makeRange(end, end+s.H),
		})
		start += s.StepSz
	}
	return splits, nil
}

func makeRange(lo, hi int) []int {
	if hi <= lo {
		return []int{}
	}
	out := make([]int, hi-lo)
	for i := range out {
		out[i] = lo + i
	}
	return out
}
