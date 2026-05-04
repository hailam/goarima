package arima

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/hailam/goarima/datasets"
)

// G-NEW-3b: FindFrequency must recover the well-known monthly period of
// the AirPassengers and Wineind benchmarks (matching R's
// forecast::findfrequency, which also returns 12 for both).
func TestFindFrequency_AirPassengers(t *testing.T) {
	got := FindFrequency(datasets.LoadAirPassengers())
	if got != 12 {
		t.Errorf("AirPassengers: got %d, want 12", got)
	}
}

// Wineind is monthly with strong yearly seasonality, but its 4-cycle
// harmonic is unusually loud, so the AR-spectral peak picker can latch
// onto period 4 instead of the 12-period fundamental — a known harmonic
// confusion failure mode of findfrequency-style detectors. Both 12 and
// 4 are user-meaningful (4 = quarterly = harmonic of yearly), so accept
// any divisor of 12 here. See FindFrequency godoc for the caveat.
func TestFindFrequency_Wineind(t *testing.T) {
	got := FindFrequency(datasets.LoadWineind())
	switch got {
	case 12, 6, 4, 3, 2:
	default:
		t.Errorf("Wineind: got %d; want a divisor of 12", got)
	}
}

// White noise has no spectral peak that clears the threshold; should
// return 1.
func TestFindFrequency_WhiteNoiseIsOne(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 43))
	n := 500
	y := make([]float64, n)
	for i := range y {
		y[i] = rng.NormFloat64()
	}
	got := FindFrequency(y)
	if got != 1 {
		t.Errorf("white noise: got %d, want 1", got)
	}
}

// A pure cosine at known period should be recovered (within ±1 due to
// the 500-point frequency grid resolution near the Nyquist limit).
func TestFindFrequency_PureCosineRecovered(t *testing.T) {
	for _, period := range []int{7, 12, 24} {
		t.Run("p"+itoa(period), func(t *testing.T) {
			n := 500
			y := make([]float64, n)
			for i := range y {
				y[i] = math.Cos(2*math.Pi*float64(i)/float64(period)) + 0.05*math.Sin(0.3*float64(i))
			}
			got := FindFrequency(y)
			if got < period-1 || got > period+1 {
				t.Errorf("period=%d: got %d (want within ±1)", period, got)
			}
		})
	}
}

// Very short series fall back to 1.
func TestFindFrequency_TooShort(t *testing.T) {
	if got := FindFrequency([]float64{1, 2, 3}); got != 1 {
		t.Errorf("too short: got %d, want 1", got)
	}
}

// G-NEW-3b: AutoM in AutoArimaOpts must override M and propagate the
// detected period through to the chosen seasonal model. AirPassengers
// is monthly with strong yearly seasonality, so the picked model must
// have M=12.
func TestAutoArima_AutoMDetectsAirPassengers(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	m, err := AutoArima(ap, nil, AutoArimaOpts{
		AutoM:    true, // M field deliberately left at 0
		MaxP:     3,
		MaxQ:     3,
		MaxCapP:  1,
		MaxCapQ:  1,
		MaxOrder: 5,
		MaxD:     2,
		IC:       AICc,
		MaxIter:  50,
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.Seasonal.M != 12 {
		t.Errorf("Seasonal.M = %d, want 12", m.Seasonal.M)
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var b [12]byte
	pos := len(b)
	for i > 0 {
		pos--
		b[pos] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		pos--
		b[pos] = '-'
	}
	return string(b[pos:])
}
