package arima

import (
	"math"
	"math/cmplx"
	"testing"
)

// AR(1) with phi=0.5: characteristic poly 1 - 0.5z, root at z=2.
func TestARRootsAR1(t *testing.T) {
	y := simulateAR1(500, 0.5, 1.0, 1)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 80
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	roots := m.ARRoots()
	if len(roots) != 1 {
		t.Fatalf("expected 1 root, got %d", len(roots))
	}
	r := roots[0]
	// root should be ~ 1/phi = 2 (real, positive). Allow ±0.4 due to noise.
	if math.Abs(real(r)-2) > 0.4 {
		t.Errorf("AR(1) root real=%v want ~2", real(r))
	}
	if math.Abs(imag(r)) > 1e-6 {
		t.Errorf("AR(1) root should be real, imag=%v", imag(r))
	}
	if !m.IsStationary() {
		t.Errorf("phi=0.5 should be stationary")
	}
}

// MA(1) with theta=0.5: poly 1 + 0.5z, root at z=-2.
func TestMARootsMA1(t *testing.T) {
	y := simulateMA1(500, 0.5, 1.0, 1)
	m := NewARIMA(Order{P: 0, D: 0, Q: 1})
	m.MaxIter = 80
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	roots := m.MARoots()
	if len(roots) != 1 {
		t.Fatalf("got %d roots", len(roots))
	}
	if cmplx.Abs(roots[0]) <= 1 {
		t.Errorf("MA root inside unit circle: %v", roots[0])
	}
	if !m.IsInvertible() {
		t.Errorf("theta=0.5 should be invertible")
	}
}

func TestRootsNoARNoMA(t *testing.T) {
	y := []float64{0, 0, 0, 0, 1, 2, 3, 4, 5, 6}
	m := NewARIMA(Order{P: 0, D: 1, Q: 0})
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	if len(m.ARRoots()) != 0 || len(m.MARoots()) != 0 {
		t.Error("expected no roots for (0,1,0)")
	}
	if !m.IsStationary() || !m.IsInvertible() {
		t.Error("vacuous case must be stationary/invertible")
	}
}

func TestMinRootAbs(t *testing.T) {
	y := simulateAR1(300, 0.5, 1.0, 1)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.MaxIter = 60
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	if m.MinRootAbs() <= 1 {
		t.Errorf("min root abs %v should be > 1 for stationary AR(1)", m.MinRootAbs())
	}
}
