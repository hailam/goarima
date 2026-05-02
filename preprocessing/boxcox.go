// Package preprocessing provides endogenous and exogenous transformers
// ported from pmdarima.preprocessing.
package preprocessing

import (
	"errors"
	"fmt"
	"math"
)

// NegAction controls how non-positive values are treated during BoxCox.
type NegAction int

const (
	// NegRaise raises an error if any (y + lmbda2) <= 0.
	NegRaise NegAction = iota
	// NegWarn truncates non-positive values to floor and emits a warning callback.
	NegWarn
	// NegIgnore truncates non-positive values to floor silently.
	NegIgnore
)

// BoxCoxEndogTransformer applies the Box-Cox power transformation.
//
//	z = ((y + lam2)^lam1 - 1) / lam1   if lam1 != 0
//	z = log(y + lam2)                  if lam1 == 0
//
// Mirrors pmdarima.preprocessing.BoxCoxEndogTransformer.
type BoxCoxEndogTransformer struct {
	// User-set parameters
	Lmbda     *float64 // if nil, estimate via MLE
	Lmbda2    float64
	NegAction NegAction
	Floor     float64
	OnWarn    func(string) // optional warn hook for NegWarn

	// Fitted state
	lam1   float64
	lam2   float64
	fitted bool
}

// NewBoxCoxEndogTransformer creates a transformer with default parameters.
func NewBoxCoxEndogTransformer() *BoxCoxEndogTransformer {
	return &BoxCoxEndogTransformer{
		Lmbda2:    0,
		NegAction: NegRaise,
		Floor:     1e-16,
	}
}

// Fit estimates lambda if not provided.
func (t *BoxCoxEndogTransformer) Fit(y []float64) error {
	if t.Floor == 0 {
		t.Floor = 1e-16
	}
	if t.Lmbda2 < 0 {
		return errors.New("lmbda2 must be a non-negative scalar value")
	}
	t.lam2 = t.Lmbda2
	if t.Lmbda != nil {
		t.lam1 = *t.Lmbda
		t.fitted = true
		return nil
	}
	if len(y) == 0 {
		return errors.New("cannot fit BoxCox on empty array")
	}
	// scipy.stats.boxcox MLE: lambda must produce strictly positive shifted y.
	shifted := make([]float64, len(y))
	for i, v := range y {
		s := v + t.lam2
		if s <= 0 {
			return fmt.Errorf("BoxCox MLE requires y + lmbda2 > 0; got %v", s)
		}
		shifted[i] = s
	}
	t.lam1 = boxcoxNormMLE(shifted)
	t.fitted = true
	return nil
}

// boxcoxNormMLE finds lambda that maximizes the Box-Cox log-likelihood
// (per Box & Cox, 1964; matches scipy.stats.boxcox without alpha).
func boxcoxNormMLE(y []float64) float64 {
	// Profile log-likelihood:
	//   L(lambda) = -n/2 * log(var(z(lambda))) + (lambda-1) * sum(log y)
	negLL := func(lambda float64) float64 {
		z := make([]float64, len(y))
		if math.Abs(lambda) < 1e-12 {
			for i, v := range y {
				z[i] = math.Log(v)
			}
		} else {
			for i, v := range y {
				z[i] = (math.Pow(v, lambda) - 1) / lambda
			}
		}
		mean := 0.0
		for _, v := range z {
			mean += v
		}
		mean /= float64(len(z))
		varZ := 0.0
		for _, v := range z {
			varZ += (v - mean) * (v - mean)
		}
		varZ /= float64(len(z)) // MLE-style (no bias correction; matches scipy)
		if varZ <= 0 {
			return math.Inf(1)
		}
		sumLog := 0.0
		for _, v := range y {
			sumLog += math.Log(v)
		}
		ll := -float64(len(z))/2*math.Log(varZ) + (lambda-1)*sumLog
		return -ll
	}
	// Bracket lambda in [-2, 2] and use golden-section search.
	return goldenMin(negLL, -2.0, 2.0, 1e-7)
}

// goldenMin minimizes f on [a,b] via golden-section search.
func goldenMin(f func(float64) float64, a, b, tol float64) float64 {
	const phi = 1.6180339887498949
	gr := (math.Sqrt(5) - 1) / 2 // 1/phi
	c := b - gr*(b-a)
	d := a + gr*(b-a)
	for math.Abs(b-a) > tol {
		fc := f(c)
		fd := f(d)
		if fc < fd {
			b = d
		} else {
			a = c
		}
		c = b - gr*(b-a)
		d = a + gr*(b-a)
	}
	_ = phi
	return (a + b) / 2
}

// Lambda returns the fitted lambda1 (power) and lambda2 (shift).
func (t *BoxCoxEndogTransformer) Lambda() (lam1, lam2 float64) {
	return t.lam1, t.lam2
}

// Transform applies the Box-Cox transform to y, returning a new slice.
func (t *BoxCoxEndogTransformer) Transform(y []float64) ([]float64, error) {
	if !t.fitted {
		return nil, errors.New("transformer not fitted")
	}
	out := make([]float64, len(y))
	hasNeg := false
	for i, v := range y {
		s := v + t.lam2
		if s <= 0 {
			hasNeg = true
			s = t.Floor
		}
		out[i] = s
	}
	if hasNeg {
		switch t.NegAction {
		case NegRaise:
			return nil, errors.New("negative or zero values present in y")
		case NegWarn:
			if t.OnWarn != nil {
				t.OnWarn("Negative or zero values present in y")
			}
		}
	}
	if math.Abs(t.lam1) < 1e-12 {
		for i, s := range out {
			out[i] = math.Log(s)
		}
	} else {
		for i, s := range out {
			out[i] = (math.Pow(s, t.lam1) - 1) / t.lam1
		}
	}
	return out, nil
}

// InverseTransform reverses the Box-Cox transform.
func (t *BoxCoxEndogTransformer) InverseTransform(y []float64) ([]float64, error) {
	if !t.fitted {
		return nil, errors.New("transformer not fitted")
	}
	out := make([]float64, len(y))
	if math.Abs(t.lam1) < 1e-12 {
		for i, v := range y {
			out[i] = math.Exp(v) - t.lam2
		}
		return out, nil
	}
	for i, v := range y {
		num := v*t.lam1 + 1
		out[i] = math.Pow(num, 1/t.lam1) - t.lam2
	}
	return out, nil
}

// FitTransform combines fit + transform.
func (t *BoxCoxEndogTransformer) FitTransform(y []float64) ([]float64, error) {
	if err := t.Fit(y); err != nil {
		return nil, err
	}
	return t.Transform(y)
}

// LogEndogTransformer is a BoxCox with lam1=0; lmbda parameter shifts by lam2.
type LogEndogTransformer struct {
	*BoxCoxEndogTransformer
}

// NewLogEndogTransformer mirrors pmdarima.LogEndogTransformer(lmbda=0).
func NewLogEndogTransformer(lmbda float64, neg NegAction, floor float64) *LogEndogTransformer {
	zero := 0.0
	bc := &BoxCoxEndogTransformer{
		Lmbda:     &zero,
		Lmbda2:    lmbda,
		NegAction: neg,
		Floor:     floor,
	}
	if floor == 0 {
		bc.Floor = 1e-16
	}
	return &LogEndogTransformer{BoxCoxEndogTransformer: bc}
}
