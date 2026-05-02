package arima

import (
	"math/rand/v2"
	"testing"

	"github.com/hailam/goarima/datasets"
)

// auto_arima must produce a reasonable model on austres (a non-stationary trend).
// Mirrors test_auto_arima_with_arima — auto-selected model should fit and
// produce forecasts.
func TestAutoArimaAustres(t *testing.T) {
	austres := datasets.LoadAustres()
	mdl, err := AutoArima(austres, AutoArimaOpts{
		M:        4,
		MaxP:     3,
		MaxQ:     3,
		MaxCapP:  1,
		MaxCapQ:  1,
		MaxOrder: 5,
		MaxD:     2,
		Alpha:    0.05,
		Test:     NDiffsKPSS,
		IC:       AICc,
		MaxIter:  60,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mdl == nil {
		t.Fatal("auto_arima returned nil")
	}
	// austres requires at least one diff
	if mdl.Order.D == 0 && mdl.Seasonal.D == 0 {
		t.Errorf("austres should have d or D > 0; order=%v seasonal=%v",
			mdl.Order, mdl.Seasonal)
	}
	// Forecast should not error.
	fc, _, _, err := mdl.Predict(4, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	if len(fc) != 4 {
		t.Errorf("forecast len=%d", len(fc))
	}
}

// auto_arima on a known AR(1) should pick up p>=1.
func TestAutoArimaAR1(t *testing.T) {
	rng := rand.New(rand.NewPCG(123, 124))
	n := 300
	phi := 0.6
	y := make([]float64, n)
	for i := 1; i < n; i++ {
		y[i] = phi*y[i-1] + rng.NormFloat64()
	}
	mdl, err := AutoArima(y, AutoArimaOpts{
		M:        0,
		MaxP:     3,
		MaxQ:     3,
		MaxOrder: 5,
		MaxD:     2,
		Alpha:    0.05,
		IC:       AICc,
		MaxIter:  80,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mdl.Order.P == 0 && mdl.Order.Q == 0 && mdl.Order.D == 0 {
		t.Errorf("expected non-trivial model, got %v", mdl.Order)
	}
}

// auto_arima must accept a too-short series only with an error.
func TestAutoArimaTooShort(t *testing.T) {
	y := []float64{1, 2, 3}
	if _, err := AutoArima(y, AutoArimaOpts{}); err == nil {
		t.Error("expected error for very short series")
	}
}
