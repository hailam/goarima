package utils

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestACFBasics(t *testing.T) {
	// White noise: ACF[1..k] should be near 0; ACF[0] = 1.
	y := []float64{1, -1, 1, -1, 1, -1, 1, -1, 1, -1}
	acf, err := ACF(y, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(acf) != 4 {
		t.Fatalf("len=%d", len(acf))
	}
	if math.Abs(acf[0]-1) > 1e-12 {
		t.Errorf("acf[0]=%v want 1", acf[0])
	}
	// y is exactly periodic with period 2: acf[1] should be -0.9, acf[2] should be 0.8.
	if !(acf[1] < 0) {
		t.Errorf("acf[1]=%v expected negative", acf[1])
	}
	if !(acf[2] > 0) {
		t.Errorf("acf[2]=%v expected positive", acf[2])
	}
}

// AR(1) with phi=0.7: theoretical ACF[k] = 0.7^k. Sample ACF should be close.
func TestACFAR1(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 43))
	y := make([]float64, 2000)
	for i := 1; i < len(y); i++ {
		y[i] = 0.7*y[i-1] + rng.NormFloat64()
	}
	acf, err := ACF(y, 5)
	if err != nil {
		t.Fatal(err)
	}
	for k := 1; k <= 3; k++ {
		want := math.Pow(0.7, float64(k))
		if math.Abs(acf[k]-want) > 0.1 {
			t.Errorf("acf[%d]=%v want ~%v", k, acf[k], want)
		}
	}
}

// PACF of AR(1): pacf[1] should be phi, pacf[k] = 0 for k > 1.
func TestPACFAR1(t *testing.T) {
	rng := rand.New(rand.NewPCG(123, 124))
	y := make([]float64, 2000)
	for i := 1; i < len(y); i++ {
		y[i] = 0.7*y[i-1] + rng.NormFloat64()
	}
	pacf, err := PACF(y, 5)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(pacf[0]-1) > 1e-12 {
		t.Errorf("pacf[0]=%v", pacf[0])
	}
	if math.Abs(pacf[1]-0.7) > 0.1 {
		t.Errorf("pacf[1]=%v want ~0.7", pacf[1])
	}
	for k := 2; k <= 5; k++ {
		if math.Abs(pacf[k]) > 0.1 {
			t.Errorf("pacf[%d]=%v want ~0", k, pacf[k])
		}
	}
}

func TestACFErrors(t *testing.T) {
	if _, err := ACF([]float64{1, 2}, 5); err == nil {
		t.Error("expected error: nLags >= n")
	}
	if _, err := ACF(nil, 1); err == nil {
		t.Error("expected error: empty y")
	}
	if _, err := ACF([]float64{1, 1, 1, 1}, 2); err == nil {
		t.Error("expected error: zero variance")
	}
}
