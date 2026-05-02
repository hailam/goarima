package arima

import (
	"fmt"
)

// AutoArimaOpts configures the auto_arima search.
type AutoArimaOpts struct {
	// Seasonal period; if 0 or 1, only non-seasonal models are considered.
	M int

	// Optional fixed differencing terms; -1 means "estimate".
	D  int
	Dd int // seasonal D
	HasD  bool
	HasDd bool

	// Search ranges.
	StartP, MaxP int
	StartQ, MaxQ int
	StartCapP, MaxCapP int // seasonal P
	StartCapQ, MaxCapQ int // seasonal Q

	MaxOrder int // p+q+P+Q upper bound (default 5)
	MaxD     int // upper bound for non-seasonal d (default 2)
	MaxCapD  int // upper bound for seasonal D (default 1)

	Alpha float64 // alpha for diff tests
	Test  NDiffsTest // unit-root test (default KPSS)

	WithIntercept *bool // explicit override; nil = auto

	// Information criterion to minimize.
	IC InfoCriterion

	// Maximum optimizer iterations per fit.
	MaxIter int
}

// AutoArima runs a stepwise model selection over ARIMA(p,d,q)(P,D,Q,m).
// exog is optional (nil for none); if provided, every fitted candidate
// includes the same regressors and the chosen IC compares like-for-like.
//
// Mirrors a simplified subset of pmdarima.auto_arima: chooses d/D via unit-root
// tests, then iteratively explores neighbors of the current order, keeping
// the model with the lowest IC.
func AutoArima(y []float64, exog [][]float64, opts AutoArimaOpts) (*ARIMA, error) {
	if len(y) < 10 {
		return nil, fmt.Errorf("series too short: %d", len(y))
	}
	if opts.MaxOrder == 0 {
		opts.MaxOrder = 5
	}
	if opts.MaxD == 0 {
		opts.MaxD = 2
	}
	if opts.MaxCapD == 0 {
		opts.MaxCapD = 1
	}
	if opts.Alpha == 0 {
		opts.Alpha = 0.05
	}
	if opts.MaxP == 0 {
		opts.MaxP = 5
	}
	if opts.MaxQ == 0 {
		opts.MaxQ = 5
	}
	if opts.MaxCapP == 0 {
		opts.MaxCapP = 2
	}
	if opts.MaxCapQ == 0 {
		opts.MaxCapQ = 2
	}
	if opts.MaxIter == 0 {
		opts.MaxIter = 100
	}

	// Determine d if not provided.
	d := opts.D
	if !opts.HasD {
		nd, err := NDiffs(y, NDiffsOpts{
			Alpha: opts.Alpha, Test: opts.Test, MaxD: opts.MaxD,
		})
		if err != nil {
			return nil, fmt.Errorf("ndiffs: %w", err)
		}
		d = nd
	}

	// Determine D if seasonal.
	D := 0
	if opts.M > 1 {
		if opts.HasDd {
			D = opts.Dd
		} else {
			Dx, err := NSDiffs(y, NSDiffsOpts{
				M: opts.M, MaxD: opts.MaxCapD, Test: NSDiffsCH, MaxLag: 3,
			})
			if err == nil {
				D = Dx
			}
		}
	}

	// Compute initial seed.
	p := minInt(opts.StartP, opts.MaxP)
	q := minInt(opts.StartQ, opts.MaxQ)
	if p == 0 && opts.StartP == 0 {
		p = minInt(2, opts.MaxP)
	}
	if q == 0 && opts.StartQ == 0 {
		q = minInt(2, opts.MaxQ)
	}
	P := minInt(opts.StartCapP, opts.MaxCapP)
	Q := minInt(opts.StartCapQ, opts.MaxCapQ)
	if opts.M <= 1 {
		P, Q = 0, 0
	}

	withIntercept := false
	if opts.WithIntercept != nil {
		withIntercept = *opts.WithIntercept
	} else {
		withIntercept = (d + D) == 0
	}

	// Cache fit results by order.
	type orderKey struct{ p, q, P, Q int }
	cache := map[orderKey]*ARIMA{}

	tryOrder := func(p, q, capP, capQ int) (*ARIMA, error) {
		if p < 0 || q < 0 || capP < 0 || capQ < 0 {
			return nil, fmt.Errorf("negative orders")
		}
		if p > opts.MaxP || q > opts.MaxQ || capP > opts.MaxCapP || capQ > opts.MaxCapQ {
			return nil, fmt.Errorf("exceeds max order")
		}
		if p+q+capP+capQ > opts.MaxOrder {
			return nil, fmt.Errorf("exceeds max_order sum")
		}
		k := orderKey{p, q, capP, capQ}
		if cached, ok := cache[k]; ok {
			return cached, nil
		}
		ord := Order{P: p, D: d, Q: q}
		var ssn SeasonalOrder
		if opts.M > 1 && (capP+capQ+D > 0) {
			ssn = SeasonalOrder{P: capP, D: D, Q: capQ, M: opts.M}
		}
		mdl := &ARIMA{
			Order:         ord,
			Seasonal:      ssn,
			WithIntercept: withIntercept,
			Method:        MethodCSSML,
			MaxIter:       opts.MaxIter,
		}
		if err := mdl.Fit(y, exog); err != nil {
			cache[k] = nil
			return nil, err
		}
		cache[k] = mdl
		return mdl, nil
	}

	best, err := tryOrder(p, q, P, Q)
	if err != nil {
		return nil, fmt.Errorf("initial fit failed: %w", err)
	}
	bestIC := best.IC(opts.IC)
	bestKey := orderKey{p, q, P, Q}

	improved := true
	for iter := 0; iter < 50 && improved; iter++ {
		improved = false
		// neighbours: ±1 in each of p, q, P, Q.
		neighbors := []orderKey{
			{bestKey.p - 1, bestKey.q, bestKey.P, bestKey.Q},
			{bestKey.p + 1, bestKey.q, bestKey.P, bestKey.Q},
			{bestKey.p, bestKey.q - 1, bestKey.P, bestKey.Q},
			{bestKey.p, bestKey.q + 1, bestKey.P, bestKey.Q},
		}
		if opts.M > 1 {
			neighbors = append(neighbors,
				orderKey{bestKey.p, bestKey.q, bestKey.P - 1, bestKey.Q},
				orderKey{bestKey.p, bestKey.q, bestKey.P + 1, bestKey.Q},
				orderKey{bestKey.p, bestKey.q, bestKey.P, bestKey.Q - 1},
				orderKey{bestKey.p, bestKey.q, bestKey.P, bestKey.Q + 1},
			)
		}
		// Also try toggling intercept once, only when d+D <= 1.
		for _, n := range neighbors {
			cand, err := tryOrder(n.p, n.q, n.P, n.Q)
			if err != nil || cand == nil {
				continue
			}
			ic := cand.IC(opts.IC)
			if ic < bestIC-1e-6 {
				best = cand
				bestIC = ic
				bestKey = n
				improved = true
			}
		}
	}
	return best, nil
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
