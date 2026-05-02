package arima

import (
	"math"
	"testing"

	"github.com/hailam/goarima/datasets"
)

// BenchmarkFitARIMA011_Airline measures fitting an ARIMA(0,1,1)(0,1,1)[12] —
// the canonical Box-Jenkins airline model — on log(AirPassengers).
func BenchmarkFitARIMA011_Airline(b *testing.B) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := NewARIMA(Order{P: 0, D: 1, Q: 1})
		m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
		m.MaxIter = 100
		if err := m.Fit(logAp, nil); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkFitNonSimple measures the exact-diffuse Kalman path.
func BenchmarkFitNonSimple_Airline(b *testing.B) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := NewARIMA(Order{P: 0, D: 1, Q: 1})
		m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
		m.NonSimpleDifferencing = true
		m.MaxIter = 100
		if err := m.Fit(logAp, nil); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkAutoArima_AirPassengers measures stepwise auto-selection on AirPassengers.
func BenchmarkAutoArima_AirPassengers(b *testing.B) {
	ap := datasets.LoadAirPassengers()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := AutoArima(ap, nil, AutoArimaOpts{
			M: 12, MaxP: 3, MaxQ: 3, MaxCapP: 1, MaxCapQ: 1,
			MaxOrder: 5, MaxD: 2, IC: AICc, MaxIter: 50,
		})
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkAutoArimaFull_Wineind measures full-search auto-selection.
func BenchmarkAutoArimaFull_Wineind(b *testing.B) {
	wi := datasets.LoadWineind()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := AutoArima(wi, nil, AutoArimaOpts{
			M: 12, MaxP: 2, MaxQ: 2, MaxCapP: 1, MaxCapQ: 1,
			MaxOrder: 4, MaxD: 2, IC: AICc, MaxIter: 30,
			FullSearch: true,
		})
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkKalmanARMALikelihood measures one likelihood evaluation.
func BenchmarkKalmanARMALikelihood(b *testing.B) {
	y := simulateAR1(500, 0.5, 1.0, 1)
	phi := []float64{0.5}
	theta := []float64{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = kalmanARMALikelihood(y, phi, theta)
	}
}

// BenchmarkKalmanARIMAFull measures one exact-diffuse likelihood evaluation.
func BenchmarkKalmanARIMAFull(b *testing.B) {
	wi := datasets.LoadWineind()
	phi := []float64{0.13}
	theta := []float64{-0.72}
	sTheta := []float64{-0.39}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = kalmanARIMAFull(wi, 1, 12, 1, phi, theta, nil, sTheta, 1e6)
	}
}
