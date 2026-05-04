package arima

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/hailam/goarima/datasets"
)

// White noise input: portmanteau tests should return high p-values
// (no detectable autocorrelation).
func TestLjungBox_WhiteNoise(t *testing.T) {
	// Independent N(0,1) noise — simulate via simulateAR1 with phi=0 = pure noise.
	y := simulateAR1(500, 0.0, 1.0, 42)
	Q, p, err := LjungBox(y, 24, 0)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(Q) || math.IsInf(Q, 0) {
		t.Errorf("Q = %g (NaN/Inf)", Q)
	}
	if p < 0.05 {
		t.Errorf("LjungBox on white noise: p = %g, want > 0.05", p)
	}
}

// Strongly autocorrelated input: portmanteau tests should reject white-noise.
func TestLjungBox_StrongAR(t *testing.T) {
	// AR(1) phi=0.8 has clear positive autocorrelation at lag 1.
	y := simulateAR1(500, 0.8, 1.0, 11)
	Q, p, err := LjungBox(y, 24, 0)
	if err != nil {
		t.Fatal(err)
	}
	if Q < 100 {
		t.Errorf("LjungBox on AR(0.8): Q = %g, expected > 100", Q)
	}
	if p > 0.001 {
		t.Errorf("LjungBox on AR(0.8): p = %g, want very small (< 0.001)", p)
	}
}

// Box-Pierce on white noise — same null behavior.
func TestBoxPierce_WhiteNoise(t *testing.T) {
	y := simulateAR1(500, 0.0, 1.0, 7)
	Q, p, err := BoxPierce(y, 24, 0)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(Q) || math.IsInf(Q, 0) {
		t.Errorf("Q = %g", Q)
	}
	if p < 0.05 {
		t.Errorf("BoxPierce on white noise: p = %g, want > 0.05", p)
	}
}

// Ljung-Box should be slightly larger than Box-Pierce for the same series
// (the (n+2)/(n-k) weighting makes LB more conservative).
func TestLjungBox_VsBoxPierce_SameSeries(t *testing.T) {
	y := simulateAR1(300, 0.5, 1.0, 99)
	qLB, _, err := LjungBox(y, 12, 0)
	if err != nil {
		t.Fatal(err)
	}
	qBP, _, err := BoxPierce(y, 12, 0)
	if err != nil {
		t.Fatal(err)
	}
	if qLB <= qBP {
		t.Errorf("Ljung-Box Q=%g should exceed Box-Pierce Q=%g", qLB, qBP)
	}
}

// NaN-padded input (the shape of m.Resid() since the API audit) must be
// handled — NaN entries are filtered before computing autocorrelations.
func TestLjungBox_NaNPaddedInput(t *testing.T) {
	y := simulateAR1(200, 0.4, 1.0, 5)
	// Pad the front with 13 NaN (mimics the dHead warmup region).
	padded := make([]float64, 0, len(y)+13)
	for i := 0; i < 13; i++ {
		padded = append(padded, math.NaN())
	}
	padded = append(padded, y...)

	Q1, _, err := LjungBox(padded, 24, 0)
	if err != nil {
		t.Fatal(err)
	}
	Q2, _, err := LjungBox(y, 24, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Filtering NaN should yield exactly the same Q as passing the unpadded series.
	if math.Abs(Q1-Q2) > 1e-9 {
		t.Errorf("NaN-padded Q=%g differs from unpadded Q=%g", Q1, Q2)
	}
}

// (*ARIMA).LjungBox / (*ARIMA).BoxPierce convenience methods on a fitted
// model — auto-derive modelOrder from the ARMA degrees of freedom.
func TestARIMA_LjungBox_ConvenienceMethod(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	m := NewARIMA(Order{P: 0, D: 1, Q: 1})
	m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
	m.MaxIter = 100
	if err := m.Fit(logAp, nil); err != nil {
		t.Fatal(err)
	}

	Q, p, err := m.LjungBox(24)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(Q) || math.IsInf(Q, 0) {
		t.Errorf("Q = %g", Q)
	}
	if math.IsNaN(p) || math.IsInf(p, 0) {
		t.Errorf("p = %g", p)
	}
	// Box-Jenkins airline model on log(AirPassengers) is the canonical
	// well-fit example — residuals should be (approximately) white noise.
	if p < 0.01 {
		t.Errorf("airline-model LB p = %g, expected residuals to look white-noisy", p)
	}

	// BoxPierce should also produce finite output on the same model.
	if _, _, err := m.BoxPierce(24); err != nil {
		t.Fatal(err)
	}
}

// Argument validation.
func TestLjungBox_ArgValidation(t *testing.T) {
	y := simulateAR1(100, 0.3, 1.0, 1)
	// h < 1
	if _, _, err := LjungBox(y, 0, 0); err == nil {
		t.Error("expected error for h=0")
	}
	// modelOrder < 0
	if _, _, err := LjungBox(y, 12, -1); err == nil {
		t.Error("expected error for negative modelOrder")
	}
	// h >= len(filtered)
	if _, _, err := LjungBox(y[:5], 5, 0); err == nil {
		t.Error("expected error when h >= len(residuals)")
	}
	// fewer than 4 finite values
	if _, _, err := LjungBox(y[:3], 1, 0); err == nil {
		t.Error("expected error for n<4")
	}
	// all-NaN input
	allNaN := []float64{math.NaN(), math.NaN(), math.NaN(), math.NaN()}
	if _, _, err := LjungBox(allNaN, 1, 0); err == nil {
		t.Error("expected error for all-NaN input")
	}
}

// Parity check against statsmodels.acorr_ljungbox on a fixed small
// triangular-wave series. Verified via:
//
//	from statsmodels.stats.diagnostic import acorr_ljungbox
//	y = np.array([1,2,3,4,5,4,3,2,1,2,3,4,5,4,3,2,1,2,3,4], dtype=float)
//	acorr_ljungbox(y, lags=[5], boxpierce=True)
//	#   lb_stat = 41.772662, lb_p = 6.548473e-08
//	#   bp_stat = 31.654554, bp_p = 6.92e-06
func TestLjungBox_StatsmodelsParity(t *testing.T) {
	y := []float64{
		1, 2, 3, 4, 5, 4, 3, 2, 1, 2,
		3, 4, 5, 4, 3, 2, 1, 2, 3, 4,
	}
	Q, p, err := LjungBox(y, 5, 0)
	if err != nil {
		t.Fatal(err)
	}
	const wantQ = 41.772662
	const wantP = 6.548473e-08
	if math.Abs(Q-wantQ)/wantQ > 1e-5 {
		t.Errorf("LjungBox Q = %g, want %g (statsmodels)", Q, wantQ)
	}
	if math.Abs(p-wantP)/wantP > 1e-3 {
		t.Errorf("LjungBox p = %g, want %g (statsmodels)", p, wantP)
	}
}

// JarqueBera parity vs statsmodels. Verified via:
//
//	from statsmodels.stats.stattools import jarque_bera
//	y = np.array([1,2,3,4,5,4,3,2,1,2,3,4,5,4,3,2,1,2,3,4], dtype=float)
//	jarque_bera(y)
//	# stat=0.843556 p=0.655880 skew=0.026391 kurt=1.995270
func TestJarqueBera_StatsmodelsParity(t *testing.T) {
	y := []float64{
		1, 2, 3, 4, 5, 4, 3, 2, 1, 2,
		3, 4, 5, 4, 3, 2, 1, 2, 3, 4,
	}
	jb, p, skew, kurt, err := JarqueBera(y)
	if err != nil {
		t.Fatal(err)
	}
	const wantJB = 0.843556
	const wantP = 0.655880
	const wantSkew = 0.026391
	const wantKurt = 1.995270
	if math.Abs(jb-wantJB) > 1e-4 {
		t.Errorf("JB = %g, want %g", jb, wantJB)
	}
	if math.Abs(p-wantP) > 1e-4 {
		t.Errorf("p = %g, want %g", p, wantP)
	}
	if math.Abs(skew-wantSkew) > 1e-4 {
		t.Errorf("skew = %g, want %g", skew, wantSkew)
	}
	if math.Abs(kurt-wantKurt) > 1e-4 {
		t.Errorf("kurtosis = %g, want %g", kurt, wantKurt)
	}
}

// JB on near-normal noise should NOT reject H0.
func TestJarqueBera_WhiteNoise(t *testing.T) {
	y := simulateAR1(500, 0.0, 1.0, 42)
	jb, p, _, _, err := JarqueBera(y)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(jb) || math.IsInf(jb, 0) {
		t.Errorf("JB = %g", jb)
	}
	if p < 0.01 {
		t.Errorf("JB on white noise: p = %g, want > 0.01", p)
	}
}

// JB should reject for skewed input.
func TestJarqueBera_SkewedRejects(t *testing.T) {
	// Exponential-ish: lots of small positives, few large.
	rng := simulateAR1(0, 0, 0, 0) // unused; just need a deterministic seed below
	_ = rng
	n := 500
	y := make([]float64, n)
	// Use a simple pseudo-random seed via simulateAR1's RNG pattern.
	for i := 0; i < n; i++ {
		// y_i = (i mod 17) - distribution is roughly uniform with strong
		// truncation at 0 and 16; far from normal.
		y[i] = float64((i*31 + 7) % 17)
	}
	_, p, _, _, err := JarqueBera(y)
	if err != nil {
		t.Fatal(err)
	}
	if p > 0.05 {
		t.Errorf("JB on non-normal series: p = %g, want < 0.05", p)
	}
}

// ArchLM parity vs statsmodels.het_arch on a Go-generated ARCH-effect series.
// Series produced by the same PCG(7,8) seed used to generate /tmp/arch_series.txt
// during diagnostics development. statsmodels reports lm=11.1981923, p=0.0475890.
func TestArchLM_StatsmodelsParity(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	n := 200
	e := make([]float64, n)
	sigma := 1.0
	for t := 1; t < n; t++ {
		sigma = math.Sqrt(0.1 + 0.7*e[t-1]*e[t-1])
		e[t] = sigma * rng.NormFloat64()
	}
	lm, p, err := ArchLM(e, 5)
	if err != nil {
		t.Fatal(err)
	}
	const wantLM = 11.1981923131
	const wantP = 0.0475889738
	if math.Abs(lm-wantLM) > 1e-6 {
		t.Errorf("ArchLM lm = %g, want %g (statsmodels)", lm, wantLM)
	}
	if math.Abs(p-wantP)/wantP > 1e-3 {
		t.Errorf("ArchLM p = %g, want %g (statsmodels)", p, wantP)
	}
}

// ArchLM on white noise should NOT reject (no ARCH effects).
func TestArchLM_WhiteNoise(t *testing.T) {
	y := simulateAR1(500, 0.0, 1.0, 42)
	_, p, err := ArchLM(y, 5)
	if err != nil {
		t.Fatal(err)
	}
	if p < 0.01 {
		t.Errorf("ArchLM on white noise: p = %g, want > 0.01", p)
	}
}

// ArchLM convenience method on a fitted model.
func TestARIMA_ArchLM_ConvenienceMethod(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	m := NewARIMA(Order{P: 0, D: 1, Q: 1})
	m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
	m.MaxIter = 100
	if err := m.Fit(logAp, nil); err != nil {
		t.Fatal(err)
	}
	if _, _, err := m.ArchLM(12); err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, err := m.JarqueBera(); err != nil {
		t.Fatal(err)
	}
}

// JB and ArchLM error paths.
func TestJarqueBeraArchLM_ArgValidation(t *testing.T) {
	short := []float64{1, 2, 3}
	if _, _, _, _, err := JarqueBera(short); err == nil {
		t.Error("expected error for n<4")
	}
	if _, _, err := ArchLM([]float64{1, 2, 3}, 1); err == nil {
		t.Error("expected error for n<2q+2")
	}
	if _, _, err := ArchLM([]float64{1, 2, 3, 4, 5}, 0); err == nil {
		t.Error("expected error for q=0")
	}
}

func TestBoxPierce_StatsmodelsParity(t *testing.T) {
	y := []float64{
		1, 2, 3, 4, 5, 4, 3, 2, 1, 2,
		3, 4, 5, 4, 3, 2, 1, 2, 3, 4,
	}
	Q, p, err := BoxPierce(y, 5, 0)
	if err != nil {
		t.Fatal(err)
	}
	const wantQ = 31.654554
	const wantP = 6.917885e-06
	if math.Abs(Q-wantQ)/wantQ > 1e-5 {
		t.Errorf("BoxPierce Q = %g, want %g (statsmodels)", Q, wantQ)
	}
	// p tolerance is looser because the chi² CDF in gonum's distuv uses
	// the incomplete-gamma approximation, which agrees with scipy to ~3
	// significant digits in the deep tail (p ≈ 1e-6) but not 6+ digits.
	// Looser bound still confirms same order of magnitude + leading digits.
	if math.Abs(p-wantP)/wantP > 1e-2 {
		t.Errorf("BoxPierce p = %g, want %g (statsmodels, ±1%%)", p, wantP)
	}
}
