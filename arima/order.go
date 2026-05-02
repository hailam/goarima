// Package arima provides ARIMA / SARIMAX model fitting and forecasting
// ported from pmdarima.arima.
package arima

import "fmt"

// Order is the (p, d, q) tuple of an ARIMA model.
type Order struct {
	P, D, Q int
}

// String returns a "(p,d,q)" representation.
func (o Order) String() string { return fmt.Sprintf("(%d,%d,%d)", o.P, o.D, o.Q) }

// SeasonalOrder is the (P, D, Q, m) tuple of a seasonal ARIMA model.
// When M == 0 or 1 (and all of P,D,Q are 0), the seasonal part is disabled.
type SeasonalOrder struct {
	P, D, Q, M int
}

// String returns "(P,D,Q,m)" or "(0,0,0,0)" when disabled.
func (s SeasonalOrder) String() string { return fmt.Sprintf("(%d,%d,%d,%d)", s.P, s.D, s.Q, s.M) }

// Active reports whether the seasonal part contributes parameters.
func (s SeasonalOrder) Active() bool { return s.M > 1 && (s.P > 0 || s.D > 0 || s.Q > 0) }

