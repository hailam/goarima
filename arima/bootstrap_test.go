package arima

import (
	"math"
	"testing"
)

func TestPredictBootBasic(t *testing.T) {
	y := simulateAR1(300, 0.6, 1.0, 1)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 60
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	res, err := m.PredictBoot(10, 0.05, 500, 42, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Mean) != 10 || len(res.Lower) != 10 || len(res.Upper) != 10 {
		t.Errorf("lengths: mean=%d lo=%d hi=%d", len(res.Mean), len(res.Lower), len(res.Upper))
	}
	for i := range res.Mean {
		if !(res.Lower[i] <= res.Mean[i] && res.Mean[i] <= res.Upper[i]) {
			t.Errorf("CI ordering violated at h=%d: lo=%v mean=%v hi=%v",
				i, res.Lower[i], res.Mean[i], res.Upper[i])
		}
		if math.IsNaN(res.Mean[i]) || math.IsNaN(res.Lower[i]) || math.IsNaN(res.Upper[i]) {
			t.Errorf("NaN at h=%d", i)
		}
	}
	// Width should grow with horizon.
	w0 := res.Upper[0] - res.Lower[0]
	wEnd := res.Upper[9] - res.Lower[9]
	if wEnd < w0 {
		t.Errorf("CI shrunk over horizon: %.3f → %.3f", w0, wEnd)
	}
}

func TestPredictBootShortSeed(t *testing.T) {
	y := simulateAR1(80, 0.4, 1.0, 99)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 30
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	r1, err := m.PredictBoot(5, 0.1, 200, 7, nil)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := m.PredictBoot(5, 0.1, 200, 7, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range r1.Mean {
		if r1.Mean[i] != r2.Mean[i] {
			t.Errorf("not deterministic at h=%d: %v vs %v", i, r1.Mean[i], r2.Mean[i])
		}
	}
}

func TestPredictBootErrors(t *testing.T) {
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	if _, err := m.PredictBoot(5, 0.05, 100, 0, nil); err == nil {
		t.Error("expected error: not fitted")
	}
}
