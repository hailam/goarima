package arima

import "math"

// FindFrequency estimates the dominant seasonal period of y by searching
// for a peak in the AR-spectrum, matching R's forecast::findfrequency().
//
// Algorithm:
//  1. Demean y.
//  2. Fit AR(p) by Yule-Walker (Levinson-Durbin), with order p ∈ [1, 10·log10(n)]
//     selected by AIC.
//  3. Evaluate the AR spectral density at 500 equispaced frequencies in [0, 0.5].
//  4. If the peak exceeds 10 (R's magic threshold) and is not at f=0,
//     return round(1 / f_peak). If the peak is at f=0 (DC due to trend),
//     skip past the first up-tick and look for the next peak.
//  5. Otherwise return 1 (no seasonality detected).
//
// Returns 1 for very short series or when no peak clears the threshold.
// Use this to suggest a seasonal period for AutoArima when the user
// doesn't already know whether the series is daily/weekly/monthly/etc.
//
// Caveat: like R's findfrequency, this can latch onto a strong harmonic
// instead of the fundamental period when the series has sharp recurring
// pulses (e.g. monthly data dominated by a quarterly subharmonic). On
// ambiguous series users should treat the result as a hint and verify
// against domain knowledge or ACF inspection.
func FindFrequency(y []float64) int {
	n := len(y)
	if n < 4 {
		return 1
	}

	pMax := int(math.Floor(10 * math.Log10(float64(n))))
	if pMax > n-1 {
		pMax = n - 1
	}
	if pMax < 1 {
		return 1
	}

	phi, sigma2 := yuleWalkerAIC(y, pMax)
	if sigma2 <= 0 || math.IsNaN(sigma2) || math.IsInf(sigma2, 0) {
		return 1
	}

	const nFreq = 500
	spec := make([]float64, nFreq)
	for i := 0; i < nFreq; i++ {
		f := 0.5 * float64(i) / float64(nFreq-1)
		// Compute |1 - sum_{k=1..p} phi_k · exp(-2πi·k·f)|^2.
		re, im := 1.0, 0.0
		for k := 1; k <= len(phi); k++ {
			angle := -2 * math.Pi * float64(k) * f
			re -= phi[k-1] * math.Cos(angle)
			im -= phi[k-1] * math.Sin(angle)
		}
		denom := re*re + im*im
		if denom <= 0 {
			spec[i] = math.Inf(1)
			continue
		}
		spec[i] = sigma2 / denom
	}

	// Find global peak.
	maxIdx := 0
	maxVal := spec[0]
	for i, v := range spec {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	if maxVal <= 10 {
		return 1
	}

	indexToPeriod := func(idx int) int {
		if idx <= 0 {
			return 0 // signals "Inf" in R
		}
		f := 0.5 * float64(idx) / float64(nFreq-1)
		p := math.Floor(1/f + 0.5)
		if math.IsInf(p, 0) || p >= float64(n) {
			return 0
		}
		return int(p)
	}

	if period := indexToPeriod(maxIdx); period > 0 {
		return period
	}

	// Period == Inf path: trend dominates spec[0]. Skip to first up-tick,
	// then find the next peak in (firstUp, nFreq).
	firstUp := -1
	for i := 1; i < nFreq; i++ {
		if spec[i]-spec[i-1] > 0 {
			firstUp = i
			break
		}
	}
	if firstUp < 0 || firstUp >= nFreq-1 {
		return 1
	}

	nextMaxIdx := firstUp + 1
	nextMaxVal := spec[nextMaxIdx]
	for i := firstUp + 2; i < nFreq; i++ {
		if spec[i] > nextMaxVal {
			nextMaxVal = spec[i]
			nextMaxIdx = i
		}
	}
	if period := indexToPeriod(nextMaxIdx); period > 0 {
		return period
	}
	return 1
}

// yuleWalkerAIC fits AR(m) for m=1..pMax by Levinson-Durbin recursion on
// the demeaned series and returns the (phi, sigma²) for the m that
// minimises AIC = log(σ²) + 2(m+1)/n. AR(0) is included as the m=0
// baseline (no coefficients, σ² = γ(0)). Returns the empty AR if AR(0)
// wins.
func yuleWalkerAIC(y []float64, pMax int) ([]float64, float64) {
	n := len(y)

	mean := 0.0
	for _, v := range y {
		mean += v
	}
	mean /= float64(n)
	yc := make([]float64, n)
	for i, v := range y {
		yc[i] = v - mean
	}

	// Autocovariances γ(0..pMax) using the divisor-N convention (R's default).
	g := make([]float64, pMax+1)
	for k := 0; k <= pMax; k++ {
		s := 0.0
		for t := k; t < n; t++ {
			s += yc[t] * yc[t-k]
		}
		g[k] = s / float64(n)
	}
	if g[0] <= 0 {
		return nil, g[0]
	}

	// AR(0) baseline.
	bestM := 0
	bestSigma2 := g[0]
	bestAIC := math.Log(g[0]) + 2.0/float64(n)
	var bestPhi []float64

	a := make([]float64, pMax+1)    // a[1..m] = current AR coefs
	aTmp := make([]float64, pMax+1) // scratch for Levinson update
	P := g[0]

	for m := 1; m <= pMax; m++ {
		// Reflection coefficient k_m.
		s := g[m]
		for j := 1; j < m; j++ {
			s -= a[j] * g[m-j]
		}
		if P <= 0 {
			break
		}
		k := s / P
		// Update coefficients: a_new[j] = a[j] - k·a[m-j] for j=1..m-1, a_new[m] = k.
		for j := 1; j < m; j++ {
			aTmp[j] = a[j] - k*a[m-j]
		}
		aTmp[m] = k
		for j := 1; j <= m; j++ {
			a[j] = aTmp[j]
		}
		P *= 1 - k*k
		if P <= 0 || math.IsNaN(P) {
			break
		}

		aic := math.Log(P) + 2.0*float64(m+1)/float64(n)
		if aic < bestAIC {
			bestAIC = aic
			bestM = m
			bestSigma2 = P
			bestPhi = append(bestPhi[:0], a[1:m+1]...)
		}
	}
	_ = bestM
	return bestPhi, bestSigma2
}
