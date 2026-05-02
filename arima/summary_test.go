package arima

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

func TestSummaryAR1(t *testing.T) {
	y := simulateAR1(500, 0.7, 1.0, 1)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 100
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	s, err := m.Summary()
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Coefs) != 1 {
		t.Fatalf("expected 1 coef, got %d", len(s.Coefs))
	}
	c := s.Coefs[0]
	if c.Name != "ar.L1" {
		t.Errorf("name=%q want ar.L1", c.Name)
	}
	if math.Abs(c.Value-0.7) > 0.07 {
		t.Errorf("AR(1) coef=%v want ~0.7", c.Value)
	}
	if math.IsNaN(c.StdErr) || c.StdErr <= 0 {
		t.Errorf("stderr=%v expected positive", c.StdErr)
	}
	// For AR(1) with phi=0.7 and n=500, |z| should be very large.
	if math.Abs(c.Z) < 5 {
		t.Errorf("z=%v expected large magnitude", c.Z)
	}
	str := s.String()
	if !strings.Contains(str, "ar.L1") {
		t.Errorf("summary string missing coef name:\n%s", str)
	}
	if !strings.Contains(str, "AIC") {
		t.Errorf("summary missing AIC line:\n%s", str)
	}
}

func TestSummaryWithExog(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	n := 300
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x := rng.NormFloat64()
		X[i] = []float64{x}
		y[i] = 2 + 3*x + rng.NormFloat64()
	}
	m := NewARIMA(Order{P: 0, D: 0, Q: 0})
	m.WithIntercept = true
	m.MaxIter = 100
	if err := m.Fit(y, X); err != nil {
		t.Fatal(err)
	}
	s, err := m.Summary()
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Coefs) != 2 {
		t.Fatalf("expected 2 coefs, got %d", len(s.Coefs))
	}
	str := s.String()
	if !strings.Contains(str, "intercept") || !strings.Contains(str, "x1") {
		t.Errorf("summary missing rows:\n%s", str)
	}
}

func TestSummaryNotFitted(t *testing.T) {
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	if _, err := m.Summary(); err == nil {
		t.Error("expected error: not fitted")
	}
}
