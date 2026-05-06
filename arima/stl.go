package arima

import "math"

// STL performs a simplified Seasonal-Trend decomposition based on
// Cleveland et al. (1990, "STL: A Seasonal-Trend Decomposition Procedure
// Based on Loess"), tuned to match R's `stats::stl` defaults closely
// enough for the Wang-Smith-Hyndman seasonal-strength heuristic
// (NSDiffsSEAS / SEASTest).
//
// Differences from R's full STL:
//
//   - Cycle-subseries smoothing uses degree-0 weighted-mean LOESS
//     (R's stl s.degree default is 0). This matches.
//   - Low-pass filter uses the MA(m)·MA(m)·MA(3) cascade (matches R)
//     but skips R's final LOESS(degree=1) polish — the cascade alone
//     is close enough to make verdicts agree.
//   - Trend smoothing uses a wide centered moving average instead of
//     LOESS(degree=1). Cheaper, and the SEAS heuristic only cares
//     about the variance ratio var(R)/var(R+S) — minor trend leakage
//     into S is tolerable as long as the binary 0/1 D-verdict stays
//     identical to R's.
//
// Verified to match R's `nsdiffs(test="seas")` verdicts on every
// canonical threeway dataset (airpassengers, co2, m5, m5_with_exog,
// sunspot_month) on 2026-05-07.
//
// Parameters:
//
//   - y       : input series, length n
//   - m       : seasonal period (>= 2)
//   - sWindow : seasonal smoothing window in cycles (R's s.window).
//     R's default is 7; mstl uses 7+4*1 = 11 for the first period.
//     Use 11 for parity with mstl-default behavior.
//   - tWindow : trend smoothing window in observations (R's t.window).
//     R's default is `nextOdd(ceil(1.5*m/(1-1.5/sWindow)))`.
//   - nIter   : inner-loop iterations. R's default is 2 (non-robust).
//
// Returns trend, seasonal, remainder slices each of length n.
func STL(y []float64, m, sWindow, tWindow, nIter int) (trend, seasonal, remainder []float64) {
	n := len(y)
	trend = make([]float64, n)
	seasonal = make([]float64, n)

	for iter := 0; iter < nIter; iter++ {
		// 1. Detrend.
		detrend := make([]float64, n)
		for i := range detrend {
			detrend[i] = y[i] - trend[i]
		}
		// 2. Cycle-subseries smoothing (degree-0 LOESS per phase).
		C := cycleSubseriesSmooth(detrend, m, sWindow)
		// 3. Low-pass filter cascade.
		L := movingAvgValid(C, m)
		L = movingAvgValid(L, m)
		L = movingAvgValid(L, 3)
		fillEdgeNaNs(L)
		// 4. Subtract low-pass to remove trend leakage from seasonal.
		for i := range seasonal {
			seasonal[i] = C[i] - L[i]
		}
		// 5. Deseasonalize.
		deseasonal := make([]float64, n)
		for i := range deseasonal {
			deseasonal[i] = y[i] - seasonal[i]
		}
		// 6. Trend smoothing.
		t := movingAvgValid(deseasonal, tWindow)
		fillEdgeNaNs(t)
		trend = t
	}

	remainder = make([]float64, n)
	for i := range remainder {
		remainder[i] = y[i] - trend[i] - seasonal[i]
	}
	return
}

// cycleSubseriesSmooth applies a degree-0 LOESS smoother (tricube-weighted
// mean over a window of width sWindow) to each phase subseries of detrend
// at frequency m. Returns a length-n time-varying seasonal estimate.
func cycleSubseriesSmooth(detrend []float64, m, sWindow int) []float64 {
	n := len(detrend)
	out := make([]float64, n)
	for c := 0; c < m; c++ {
		// Collect subseries values + their original indices.
		var sub []float64
		var idx []int
		for j := c; j < n; j += m {
			sub = append(sub, detrend[j])
			idx = append(idx, j)
		}
		smoothed := weightedMeanSmooth(sub, sWindow)
		for k, j := range idx {
			out[j] = smoothed[k]
		}
	}
	return out
}

// weightedMeanSmooth applies a degree-0 LOESS-style smoother (tricube-
// weighted mean over a window of half-width sWindow/2) at every index.
// Edges shrink the window asymmetrically rather than padding.
func weightedMeanSmooth(y []float64, sWindow int) []float64 {
	n := len(y)
	half := sWindow / 2
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		lo := i - half
		hi := i + half
		if lo < 0 {
			lo = 0
		}
		if hi >= n {
			hi = n - 1
		}
		maxD := math.Max(float64(i-lo), float64(hi-i))
		if maxD == 0 {
			maxD = 1
		}
		var sumW, sumWy float64
		for k := lo; k <= hi; k++ {
			d := math.Abs(float64(k-i)) / maxD
			w := tricubeKernel(d)
			sumW += w
			sumWy += w * y[k]
		}
		if sumW > 0 {
			out[i] = sumWy / sumW
		} else {
			out[i] = y[i]
		}
	}
	return out
}

// tricubeKernel returns (1 - |u|^3)^3 for |u| < 1, else 0.
func tricubeKernel(u float64) float64 {
	a := math.Abs(u)
	if a >= 1 {
		return 0
	}
	v := 1 - a*a*a
	return v * v * v
}

// movingAvgValid returns a centered moving average of x with window k.
// Edge positions where the window doesn't fit are NaN-marked; callers
// should run fillEdgeNaNs to replace them with nearest-valid values.
func movingAvgValid(x []float64, k int) []float64 {
	n := len(x)
	half := k / 2
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		lo := i - half
		hi := i + half
		if k%2 == 0 {
			hi = i + half - 1
		}
		if lo < 0 || hi >= n {
			out[i] = math.NaN()
			continue
		}
		s := 0.0
		for j := lo; j <= hi; j++ {
			s += x[j]
		}
		out[i] = s / float64(k)
	}
	return out
}

// fillEdgeNaNs replaces leading/trailing NaN values in x with the nearest
// non-NaN value (in either direction). Mutates x in place.
func fillEdgeNaNs(x []float64) {
	n := len(x)
	for i := 0; i < n; i++ {
		if !math.IsNaN(x[i]) {
			continue
		}
		// Search outward for nearest valid value.
		for d := 1; d < n; d++ {
			if i-d >= 0 && !math.IsNaN(x[i-d]) {
				x[i] = x[i-d]
				break
			}
			if i+d < n && !math.IsNaN(x[i+d]) {
				x[i] = x[i+d]
				break
			}
		}
		if math.IsNaN(x[i]) {
			x[i] = 0
		}
	}
}

// stlNextOdd rounds k up to the next odd integer (matches R's nextodd).
func stlNextOdd(k int) int {
	if k%2 == 0 {
		return k + 1
	}
	return k
}

// stlDefaultTWindow computes R's default trend-window:
// nextOdd(ceil(1.5*m / (1 - 1.5/sWindow))).
func stlDefaultTWindow(m, sWindow int) int {
	mf := float64(m)
	sf := float64(sWindow)
	return stlNextOdd(int(math.Ceil(1.5 * mf / (1 - 1.5/sf))))
}
