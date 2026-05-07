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

// BenchmarkFitARIMA011_AirlineWithExog measures ARIMA(0,1,1) Airline fit
// with a non-trivial exog matrix (k=5 random regressors). This is where
// the SIMD path matters — residOf's inner k-loop is the dominant
// vectorizable hot loop. Without exog (k=0) the SIMD path is never
// entered, so this bench is the "pays-off-when-it-matters" check.
func BenchmarkFitARIMA011_AirlineWithExog(b *testing.B) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	const k = 5
	exog := make([][]float64, len(logAp))
	for i := range exog {
		row := make([]float64, k)
		for j := 0; j < k; j++ {
			row[j] = math.Sin(float64(i*7+j*13) * 0.1)
		}
		exog[i] = row
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := NewARIMA(Order{P: 0, D: 1, Q: 1})
		m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
		m.MaxIter = 100
		if err := m.Fit(logAp, exog); err != nil {
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

// L-2 verification: stepwise AutoArima with NJobs=1 forces every
// neighbor candidate fit to run sequentially, holding the cache mutex
// uncontended. Compared to BenchmarkAutoArima_AirPassengers (which uses
// the parallel default), the speedup tells us how much wall-clock the
// parallel path actually saves — and therefore how much room there is
// for cache-mutex contention to matter in the first place. If the lock
// were a bottleneck, parallel speedup would be far below the per-iter
// neighbor count (4-8). If it isn't, speedup approaches min(K, GOMAXPROCS).
func BenchmarkAutoArima_AirPassengers_Sequential(b *testing.B) {
	ap := datasets.LoadAirPassengers()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := AutoArima(ap, nil, AutoArimaOpts{
			M: 12, MaxP: 3, MaxQ: 3, MaxCapP: 1, MaxCapQ: 1,
			MaxOrder: 5, MaxD: 2, IC: AICc, MaxIter: 50,
			NJobs: 1, // force sequential — no goroutine dispatch, mutex uncontended
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

// BenchmarkKalmanARMA_SeasMA_r27 is the PG-113 target case — Kalman
// likelihood for an expanded SARIMA(1,0,2)(0,0,2)[12] (state r=27)
// at n=3310, mirroring the sunspot_month seasonal-MA refit profile.
// This is the inner kernel that the audit measured at 62% flat.
func BenchmarkKalmanARMA_SeasMA_r27(b *testing.B) {
	y := simulateAR1(3310, 0.5, 1.0, 7)
	phi := expandSARMA([]float64{0.4}, nil, 12)
	theta := expandSMA([]float64{-0.3, -0.1}, []float64{-0.5, -0.2}, 12)
	var ws kalmanWorkspace
	_, _ = kalmanARMALikelihoodInto(y, phi, theta, &ws)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = kalmanARMALikelihoodInto(y, phi, theta, &ws)
	}
}

// BenchmarkKalmanARMA_Airline_r13 is the smaller airline-shape case
// (0,1,1)(0,1,1)[12] giving expanded q=13 → r=14 — the typical monthly
// SARIMA. Used to verify PG-113 changes don't regress the common case.
func BenchmarkKalmanARMA_Airline_r13(b *testing.B) {
	y := simulateAR1(132, 0.0, 1.0, 7)
	phi := []float64{}
	theta := expandSMA([]float64{-0.4}, []float64{-0.6}, 12)
	var ws kalmanWorkspace
	_, _ = kalmanARMALikelihoodInto(y, phi, theta, &ws)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = kalmanARMALikelihoodInto(y, phi, theta, &ws)
	}
}

// BenchmarkPredictBoot measures bootstrap CIs on a fitted SARIMA model.
// This path is dominated by the per-simulation forecast loop; it benefits
// strongly from the cache-reuse + window-only history optimization.
func BenchmarkPredictBoot(b *testing.B) {
	wi := datasets.LoadWineind()
	m := NewARIMA(Order{P: 1, D: 1, Q: 1})
	m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
	m.MaxIter = 100
	if err := m.Fit(wi, nil); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := m.PredictBoot(12, 0.05, 1000, 1, nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkPredictBootLarge stresses the L-7 parallel path with a longer
// horizon and more simulations — where per-path work is large enough to
// amortise goroutine overhead.
func BenchmarkPredictBootLarge(b *testing.B) {
	wi := datasets.LoadWineind()
	m := NewARIMA(Order{P: 1, D: 1, Q: 1})
	m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
	m.MaxIter = 100
	if err := m.Fit(wi, nil); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := m.PredictBoot(60, 0.05, 5000, 1, nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkPredictRepeated measures repeat-Predict on a fitted SARIMA model
// (e.g., as in cross-validation or simulation studies). Exercises the cached
// wsCenteredCache / yMSCache path.
func BenchmarkPredictRepeated(b *testing.B) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	m := NewARIMA(Order{P: 0, D: 1, Q: 1})
	m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
	m.MaxIter = 100
	if err := m.Fit(logAp, nil); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _, _ = m.Predict(12, 0.05, nil)
	}
}

func BenchmarkFitARIMA011_Airline_MethodCSS(b *testing.B) {
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
		m.Method = MethodCSS
		if err := m.Fit(logAp, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFitARIMA011_Airline_MethodML(b *testing.B) {
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
		m.Method = MethodML
		if err := m.Fit(logAp, nil); err != nil {
			b.Fatal(err)
		}
	}
}

// Worst-case realistic scenario: AutoArima with NonSimpleDifferencing on
// a long monthly series. Pre-CSS-1 this would compound 100ms+ NonSimple
// fits across ~30 candidates → multi-second wall.
func BenchmarkAutoArima_NonSimple_Wineind(b *testing.B) {
	wi := datasets.LoadWineind()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := AutoArima(wi, nil, AutoArimaOpts{
			M: 12, MaxP: 2, MaxQ: 2, MaxCapP: 1, MaxCapQ: 1,
			MaxOrder: 4, MaxD: 2, IC: AICc, MaxIter: 50,
			NonSimpleDifferencing: true,
		})
		if err != nil {
			b.Fatal(err)
		}
	}
}

// GAP-2: bench Approximation=true (CSS search + CSSML refit) vs full
// CSSML candidate fits. Per the gap audit, R uses approximation=TRUE
// by default for n>150 or m>12; goarima users now have the same lever.
func BenchmarkAutoArima_AirPassengers_Approximation(b *testing.B) {
	ap := datasets.LoadAirPassengers()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := AutoArima(ap, nil, AutoArimaOpts{
			M: 12, MaxP: 3, MaxQ: 3, MaxCapP: 1, MaxCapQ: 1,
			MaxOrder:      5,
			MaxD:          2,
			IC:            AICc,
			MaxIter:       50,
			Approximation: true,
		})
		if err != nil {
			b.Fatal(err)
		}
	}
}
