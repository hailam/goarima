package arima

import (
	"errors"
	"math"
)

// boxCoxApply transforms each y_i to ((y_i + lam2)^lam1 - 1) / lam1, or
// log(y_i + lam2) when lam1 == 0. Errors on non-positive (y + lam2).
func boxCoxApply(y []float64, lam1, lam2 float64) ([]float64, error) {
	out := make([]float64, len(y))
	if math.Abs(lam1) < 1e-12 {
		for i, v := range y {
			s := v + lam2
			if s <= 0 {
				return nil, errors.New("Box-Cox: y + lambda2 must be positive")
			}
			out[i] = math.Log(s)
		}
		return out, nil
	}
	for i, v := range y {
		s := v + lam2
		if s <= 0 {
			return nil, errors.New("Box-Cox: y + lambda2 must be positive")
		}
		out[i] = (math.Pow(s, lam1) - 1) / lam1
	}
	return out, nil
}

// boxCoxInvert inverts boxCoxApply.
func boxCoxInvert(y []float64, lam1, lam2 float64) []float64 {
	out := make([]float64, len(y))
	boxCoxInvertInto(out, y, lam1, lam2)
	return out
}

// boxCoxInvertInto is the in-place variant: writes inverse-Box-Cox of
// y into dst. dst and y may alias (dst === y is safe since each output
// element depends only on the corresponding input element).
//
// BOXCOX-INV-1 (2026-06-12): mirrors R's `forecast::InvBoxCox` signed-
// power convention, `sign(xx)·|xx|^(1/λ)` with xx = λ·y+1. The previous
// plain `Pow(xx, 1/λ)` diverged whenever xx < 0 — which happens on the
// LOWER prediction-interval bound of a Box-Cox model: NaN for
// fractional 1/λ, silently wrong SIGN when 1/λ lands on an even
// integer (λ=0.5: Pow(-0.5, 2) = +0.25 vs R's -0.25). For λ < 0, R
// additionally marks y > -1/λ as NA (the inverse is undefined there);
// we emit NaN for that region.
func boxCoxInvertInto(dst, y []float64, lam1, lam2 float64) {
	if math.Abs(lam1) < 1e-12 {
		for i, v := range y {
			dst[i] = math.Exp(v) - lam2
		}
		return
	}
	invLam := 1 / lam1
	for i, v := range y {
		if lam1 < 0 && v > -invLam {
			dst[i] = math.NaN()
			continue
		}
		xx := v*lam1 + 1
		dst[i] = math.Copysign(math.Pow(math.Abs(xx), invLam), xx) - lam2
	}
}
