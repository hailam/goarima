package preprocessing

import (
	"errors"
	"fmt"
	"math"
)

// FourierFeaturizer creates exogenous Fourier-series terms for modeling
// seasonality without seasonal ARIMA components.
//
// Mirrors pmdarima.preprocessing.FourierFeaturizer.
type FourierFeaturizer struct {
	M      int    // seasonal period
	K      int    // number of sin/cos pairs (default m/2)
	Prefix string // optional column prefix; defaults to "FOURIER"

	p      []float64 // periods = (1..K)/m
	n      int       // training set length
	fitted bool
}

// NewFourierFeaturizer creates a featurizer; if k <= 0, defaults to m/2 at fit time.
func NewFourierFeaturizer(m, k int) *FourierFeaturizer {
	return &FourierFeaturizer{M: m, K: k}
}

// Fit computes the Fourier periods. y is used only for length.
func (f *FourierFeaturizer) Fit(y []float64) error {
	m := f.M
	k := f.K
	if k <= 0 {
		k = m / 2
	}
	if 2*k > m || k < 1 {
		return errors.New("k must be a positive integer not greater than m//2")
	}
	if len(y) == 0 {
		return errors.New("y must be non-empty")
	}
	p := make([]float64, k)
	for i := 0; i < k; i++ {
		p[i] = float64(i+1) / float64(m)
	}
	f.p = p
	f.K = k
	f.n = len(y)
	f.fitted = true
	return nil
}

// Transform builds the n-by-2K Fourier matrix. If nPeriods > 0, returns the
// last nPeriods rows for forecasting. If x is provided, the matrix is column-
// concatenated to its right.
func (f *FourierFeaturizer) Transform(y []float64, x [][]float64, nPeriods int) ([][]float64, error) {
	if !f.fitted {
		return nil, errors.New("transformer not fitted")
	}
	if nPeriods > 0 && x != nil && len(x) != nPeriods {
		return nil, fmt.Errorf("n_periods (%d) must match X rows (%d)", nPeriods, len(x))
	}
	total := f.n + nPeriods
	full := fourierTerms(f.p, total)
	mat := full
	if nPeriods > 0 {
		mat = full[len(full)-nPeriods:]
	}
	if x == nil {
		return mat, nil
	}
	// hstack X | mat
	if len(x) != len(mat) {
		return nil, fmt.Errorf("x rows (%d) != mat rows (%d)", len(x), len(mat))
	}
	out := make([][]float64, len(mat))
	for i := range mat {
		row := make([]float64, len(x[i])+len(mat[i]))
		copy(row, x[i])
		copy(row[len(x[i]):], mat[i])
		out[i] = row
	}
	return out, nil
}

// FitTransform fits and transforms in one step.
func (f *FourierFeaturizer) FitTransform(y []float64, x [][]float64) ([][]float64, error) {
	if err := f.Fit(y); err != nil {
		return nil, err
	}
	return f.Transform(y, x, 0)
}

// UpdateAndTransform extends n_ by len(y) and returns the transformed exog
// matrix for the new y rows only.
func (f *FourierFeaturizer) UpdateAndTransform(y []float64, x [][]float64) ([][]float64, error) {
	if !f.fitted {
		return nil, errors.New("transformer not fitted")
	}
	out, err := f.Transform(y, x, len(y))
	if err != nil {
		return nil, err
	}
	f.n += len(y)
	return out, nil
}

// FeatureNames returns column labels like "FOURIER_S12-0", "FOURIER_C12-0", ...
func (f *FourierFeaturizer) FeatureNames() []string {
	pfx := f.Prefix
	if pfx == "" {
		pfx = "FOURIER"
	}
	cols := 2 * f.K
	out := make([]string, cols)
	for i := 0; i < cols; i++ {
		ch := "S"
		if i%2 != 0 {
			ch = "C"
		}
		out[i] = fmt.Sprintf("%s_%s%d-%d", pfx, ch, f.M, i/2)
	}
	return out
}

// fourierTerms builds the n-by-(2*len(p)) matrix where columns alternate sin/cos
// of (2*pi * p[j] * t) for t = 1..n. Mirrors pmdarima._fourier.C_fourier_terms.
func fourierTerms(p []float64, n int) [][]float64 {
	cols := 2 * len(p)
	out := make([][]float64, n)
	for t := 1; t <= n; t++ {
		row := make([]float64, cols)
		for j, pj := range p {
			arg := math.Pi * 2 * pj * float64(t)
			row[2*j] = math.Sin(arg)
			row[2*j+1] = math.Cos(arg)
		}
		out[t-1] = row
	}
	return out
}
