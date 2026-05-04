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

// ApproxTable holds a sorted, deduplicated reference grid (`xs`, `ys`)
// suitable for many subsequent interpolation calls. Constructed once via
// NewApproxTable; the table is immutable and safe to share concurrently
// across goroutines.
//
// Use ApproxTable when the same reference points feed multiple Approx
// calls — it amortises the O(n log n) sort + dedupe across the calls
// instead of paying it on every Approx invocation.
type ApproxTable struct {
	xs []float64
	ys []float64
}

// NewApproxTable validates the input and builds the sorted/deduplicated
// reference grid. Returns an error for empty or length-mismatched inputs.
//
// The returned table is immutable: subsequent Approx calls only read
// from it, so a single ApproxTable can be safely reused by many
// concurrent goroutines. The mirrors-pmdarima `ties='mean'` semantics
// from the package-level Approx are preserved.
func NewApproxTable(xRef, yRef []float64) (*ApproxTable, error) {
	if len(xRef) != len(yRef) {
		return nil, errors.New("xRef and yRef must have equal length")
	}
	if len(xRef) == 0 {
		return nil, errors.New("xRef must be non-empty")
	}
	xs, ys := regularizeMean(xRef, yRef)
	return &ApproxTable{xs: xs, ys: ys}, nil
}

// Approx interpolates this table at the requested xOut locations using
// `method` (Linear or Constant). `rule` chooses extrapolation behavior:
// RuleNaN returns NaN outside the table's x-range; RuleClip returns the
// nearest endpoint value.
//
// Concurrency-safe: only reads the table.
func (t *ApproxTable) Approx(xOut []float64, method ApproxMethod, rule ApproxRule) []float64 {
	yLeft := math.NaN()
	yRight := math.NaN()
	if rule == RuleClip {
		yLeft = t.ys[0]
		yRight = t.ys[len(t.ys)-1]
	}
	out := make([]float64, len(xOut))
	for i, v := range xOut {
		if math.IsNaN(v) {
			out[i] = math.NaN()
			continue
		}
		out[i] = approx1(v, t.xs, t.ys, yLeft, yRight, method, 1, 0)
	}
	return out
}

// Len returns the number of (x, y) reference points after deduplication.
// Useful for callers that want to know if many duplicates were collapsed.
func (t *ApproxTable) Len() int { return len(t.xs) }

// Approx performs R-style linear (or constant) interpolation.
//
// xRef and yRef are the reference points. xOut is the locations to interpolate at.
// Mirrors pmdarima.arima.approx.approx (with ties='mean' regularization).
//
// One-shot wrapper around NewApproxTable + (*ApproxTable).Approx. For
// repeated interpolation against the same reference grid, build an
// ApproxTable once and reuse it.
func Approx(xRef, yRef, xOut []float64, method ApproxMethod, rule ApproxRule) ([]float64, error) {
	t, err := NewApproxTable(xRef, yRef)
	if err != nil {
		return nil, err
	}
	return t.Approx(xOut, method, rule), nil
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
