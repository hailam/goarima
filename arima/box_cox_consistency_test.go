package arima

import (
	"math"
	"testing"

	"github.com/hailam/goarima/datasets"
)

// FittedValues should produce results in the user's original units when
// Box-Cox was used. Pre-fix the function subtracted model-scale residuals
// from original-scale yTrain — meaningless arithmetic.
//
// For Lambda=0 (log) on positive data, fitted values must be positive (since
// they're produced by exp() of model-scale predictions). They should also
// be in the same magnitude band as yTrain — not several orders of magnitude
// off.
func TestFittedValues_BoxCoxConsistency(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	lambda := 0.0
	m := NewARIMA(Order{P: 0, D: 1, Q: 1})
	m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
	m.Lambda = &lambda
	m.MaxIter = 100
	if err := m.Fit(ap, nil); err != nil {
		t.Fatal(err)
	}
	fitted := m.FittedValues()
	if len(fitted) == 0 {
		t.Fatal("FittedValues returned empty slice")
	}

	// Every fitted value must be positive (Box-Cox lambda=0 inverts to exp).
	for i, v := range fitted {
		if v <= 0 {
			t.Errorf("fitted[%d] = %g, want positive (lambda=0 inverse is exp)", i, v)
		}
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("fitted[%d] = %v (NaN/Inf)", i, v)
		}
	}

	// Fitted values should track yTrain reasonably closely (one-step-ahead
	// predictions on a well-fit model). Median absolute relative error on a
	// known-good airline fit is well under 10%.
	dHead := m.Order.D + m.Seasonal.D*m.Seasonal.M
	maxRelErr := 0.0
	for i, v := range fitted {
		actual := ap[dHead+i]
		relErr := math.Abs(v-actual) / actual
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
	}
	if maxRelErr > 0.20 {
		t.Errorf("max relative error %g — fitted values way off, likely scale-mixing", maxRelErr)
	}
}

// PredictBoot under Lambda=0 must produce strictly positive paths/quantiles
// (since the original-scale data is positive and Box-Cox lambda=0 inverts to
// exp). Pre-fix the function used original-scale heads with model-scale
// simulations and never inverse-transformed the output, easily producing
// negative values.
func TestPredictBoot_BoxCoxPositiveOutput(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	lambda := 0.0
	m := NewARIMA(Order{P: 0, D: 1, Q: 1})
	m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
	m.Lambda = &lambda
	m.MaxIter = 100
	if err := m.Fit(ap, nil); err != nil {
		t.Fatal(err)
	}
	res, err := m.PredictBoot(12, 0.05, 500, 7, nil)
	if err != nil {
		t.Fatal(err)
	}
	for h := 0; h < 12; h++ {
		if res.Mean[h] <= 0 {
			t.Errorf("mean[%d] = %g, want positive", h, res.Mean[h])
		}
		if res.Lower[h] <= 0 {
			t.Errorf("lower[%d] = %g, want positive (Box-Cox inverse should always be > 0)", h, res.Lower[h])
		}
		if res.Upper[h] <= res.Mean[h] {
			t.Errorf("upper[%d] = %g <= mean = %g", h, res.Upper[h], res.Mean[h])
		}
	}
	for s, path := range res.Paths {
		for h, v := range path {
			if v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("path[%d][%d] = %g (must be positive finite)", s, h, v)
				return
			}
		}
	}

	// Magnitude check: bootstrap median should be in the same ballpark as
	// the parametric Predict() (both forecast the same quantity).
	pf, _, _, err := m.Predict(12, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	for h := 0; h < 12; h++ {
		ratio := res.Mean[h] / pf[h]
		if ratio < 0.5 || ratio > 2.0 {
			t.Errorf("h=%d: bootstrap mean %g and parametric forecast %g differ by >2× (ratio %.2f)",
				h, res.Mean[h], pf[h], ratio)
		}
	}
}

// Sanity: with no Box-Cox, FittedValues should produce the same numbers
// pre- and post-fix (the new code path collapses to the old behaviour when
// yMSCache == yTrain and Lambda is nil).
func TestFittedValues_NoBoxCox_Unchanged(t *testing.T) {
	y := simulateAR1(200, 0.5, 1.0, 1)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.WithIntercept = true
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	fitted := m.FittedValues()
	for i, v := range fitted {
		want := y[i] - m.resids[i]
		if math.Abs(v-want) > 1e-12 {
			t.Errorf("fitted[%d] = %g, want %g (diff %g)", i, v, want, v-want)
		}
	}
}
