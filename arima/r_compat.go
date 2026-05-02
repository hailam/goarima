package arima

import (
	"errors"
	"fmt"
)

// RArimaOpts provides an R-compatible argument shape for ARIMA model
// construction, mirroring `forecast::Arima` / `stats::arima` in R.
//
// Field names map to R as follows:
//
//	Order        → order = c(p, d, q)
//	Seasonal     → seasonal = list(order = c(P, D, Q), period = m)
//	IncludeMean  → include.mean (TRUE adds intercept; only effective when d+D=0)
//	IncludeDrift → include.drift (TRUE adds linear time index as xreg)
//	Xreg         → xreg
//	Lambda       → lambda (Box-Cox; nil = no transform; *Lambda=0 → log)
//	Method       → method ("CSS-ML" → MethodCSSML; "CSS" → MethodCSS; "ML" → MethodML)
//	MaxIter      → optim.control$maxit
type RArimaOpts struct {
	Order        Order
	Seasonal     SeasonalOrder
	IncludeMean  bool
	IncludeDrift bool
	Xreg         [][]float64
	Lambda       *float64
	Lambda2      float64
	Method       string // "CSS-ML" (default), "CSS", or "ML"
	MaxIter      int
}

// RArima fits an ARIMA model with R-compatible argument shape and returns the
// fitted goarima.ARIMA struct. The fit is on the same Kalman state-space form
// as R's stats::arima, so coefficient values match R within optimizer tolerance.
//
// include.drift=TRUE prepends a linear `tt = 1..n` column to xreg, matching
// forecast::Arima's behavior.
func RArima(y []float64, opts RArimaOpts) (*ARIMA, error) {
	method := MethodCSSML
	switch opts.Method {
	case "", "CSS-ML":
		method = MethodCSSML
	case "CSS":
		method = MethodCSS
	case "ML":
		method = MethodML
	default:
		return nil, fmt.Errorf("invalid method %q (want CSS-ML, CSS, or ML)", opts.Method)
	}
	// IncludeMean only takes effect when both d and D are zero (R behaviour).
	d := opts.Order.D
	D := 0
	if opts.Seasonal.Active() {
		D = opts.Seasonal.D
	}
	withIntercept := opts.IncludeMean && (d+D == 0)
	if d+D > 1 && opts.IncludeDrift {
		return nil, errors.New("include.drift cannot be used when total differencing > 1 (matches R warning)")
	}

	// Build xreg: drift prefix + user xreg.
	xreg := opts.Xreg
	if opts.IncludeDrift {
		drift := make([][]float64, len(y))
		for i := range drift {
			drift[i] = []float64{float64(i + 1)}
		}
		if xreg == nil {
			xreg = drift
		} else {
			combined := make([][]float64, len(y))
			for i := range combined {
				combined[i] = append([]float64{float64(i + 1)}, xreg[i]...)
			}
			xreg = combined
		}
	}

	maxIter := opts.MaxIter
	if maxIter == 0 {
		maxIter = 100
	}
	m := &ARIMA{
		Order:         opts.Order,
		Seasonal:      opts.Seasonal,
		WithIntercept: withIntercept,
		Method:        method,
		MaxIter:       maxIter,
		Lambda:        opts.Lambda,
		Lambda2:       opts.Lambda2,
	}
	if err := m.Fit(y, xreg); err != nil {
		return nil, err
	}
	return m, nil
}
