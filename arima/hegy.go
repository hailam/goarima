package arima

import (
	"errors"
	"math"
)

// ErrHEGYNotSupportedForM is preserved as an exported sentinel for
// backward compatibility, but post-PG-106b is no longer returned by
// HEGYTest — the response-surface p-value table covers arbitrary m ≥ 2.
// Pre-PG-106b this fired on m ∉ {4, 12}.
var ErrHEGYNotSupportedForM = errors.New(
	"HEGYTest no longer rejects any m ≥ 2 (PG-106b shipped " +
		"response-surface p-values); m < 2 is rejected with a different error")

// HEGYTest implements the Hylleberg-Engle-Granger-Yoo seasonal
// unit-root test (HEGY 1990 quarterly, Beaulieu-Miron 1993 monthly,
// general-m via response-surface coefficients per uroot 2.x). Returns
// 1 if seasonal unit roots are detected (apply seasonal differencing),
// 0 otherwise. Verdict logic mirrors R's `forecast::nsdiffs(test="hegy")`
// which uses `uroot::hegy.test(deterministic=c(1,1,0), maxlag=3,
// lag.method="AIC")` and returns D=1 if the joint F_{2:m} p-value > α.
//
// Supports any m ≥ 2 via the response-surface p-value table from
// uroot's `.HEGY.CM.tabs[["CFs_ct_AIC"]]` — see hegy_rs_table.go.
// The response-surface regression treats m as a feature alongside the
// lag order and sample size, so verdicts match R for arbitrary m
// (verified 5/5 on m∈{7, 12} canonical threeway datasets).
//
// **Acceptance (2026-05-08):** verdicts match R's
// `nsdiffs(test="hegy")` 5/5 on the threeway canonical grid
// (airpassengers/co2/sunspot m=12, m5/m5_with_exog m=7).
//
// For the per-frequency statistical breakdown (zero-frequency t-stat,
// Nyquist t-stat, harmonic-pair F-stats), use HEGYTestFull.
func HEGYTest(x []float64, m int) (int, error) {
	res, err := HEGYTestFull(x, m)
	if err != nil {
		return 0, err
	}
	return res.Verdict, nil
}

// HEGYResult holds the per-frequency unit-root test statistics from a
// HEGY auxiliary regression, returned by HEGYTestFull. The structure
// mirrors uroot::hegy.test's return for inspection of *which*
// seasonal frequencies have unit roots.
//
// Frequency layout (zero-indexed regressor cols in the regression):
//   - π_1:       zero frequency, root at +1     → TStatZero
//   - π_2:       Nyquist, root at -1            → TStatNyquist (m even only)
//   - pairs (cosine, sine) at frequency 2π·j/M for j=1..K → PairFStats[j-1]
//     where K = (M-2)/2 for even M, (M-1)/2 for odd M.
//
// Per-frequency p-values (PG-114) use uroot's individual-statistic
// response-surface tables; computed by HEGYTestFull alongside the
// joint-F p-value.
type HEGYResult struct {
	// Verdict is 1 if seasonal unit roots are detected (fail to reject
	// H0 → apply seasonal differencing), 0 otherwise. Matches HEGYTest.
	Verdict int

	// M is the seasonal period.
	M int

	// BestLag is the AIC/BIC-selected lag-augmentation order (or the
	// fixed Lag if HEGYLagFixed was passed).
	BestLag int

	// AIC is the regression's information criterion at BestLag.
	AIC float64

	// TStatZero is the t-statistic for π_1 (root at +1, zero frequency).
	// Negative & large in magnitude rejects the unit root at that frequency.
	TStatZero float64

	// TStatZeroPValue is the response-surface p-value for TStatZero
	// (uroot Ct1 family). Small p rejects the unit root at frequency 0.
	TStatZeroPValue float64

	// TStatNyquist is the t-statistic for π_2 (root at -1, Nyquist
	// frequency). Only set when M is even; nil for odd M.
	TStatNyquist *float64

	// TStatNyquistPValue is the p-value for TStatNyquist (uroot Ct2);
	// nil for odd M (parallel to TStatNyquist).
	TStatNyquistPValue *float64

	// PairFStats[k] is the pair-F statistic for the k-th harmonic pair
	// (cos, sin) at PairFrequencies[k]. Length is (M-2)/2 for even M,
	// (M-1)/2 for odd M. Large F rejects H0 of unit roots at that pair.
	PairFStats []float64

	// PairFrequencies[k] = 2π·(k+1)/M (radians) — the angular frequency
	// of the k-th pair, parallel to PairFStats.
	PairFrequencies []float64

	// PairPValues[k] is the response-surface p-value for PairFStats[k]
	// (uroot CF). Same length as PairFStats.
	PairPValues []float64

	// JointSeasonalF is the joint F-statistic for (π_2..π_M) — the
	// overall seasonal-unit-root statistic that drives Verdict.
	JointSeasonalF float64

	// JointSeasonalPValue is the p-value of JointSeasonalF via uroot's
	// response-surface table (PG-106b / CFs).
	JointSeasonalPValue float64

	// JointAllF is the joint F for (π_1..π_M) — the overall unit-root
	// test (CFt). Differs from JointSeasonalF by including the
	// zero-frequency coefficient.
	JointAllF float64

	// JointAllPValue is the p-value of JointAllF (CFt).
	JointAllPValue float64
}

// HEGYOpts configures the HEGY auxiliary regression for HEGYTestFull.
//
// Defaults (zero-value) match forecast::nsdiffs(test="hegy"):
// Deterministic = HEGYDetConstantTrend, LagMethod = HEGYLagAIC,
// MaxLag = 3.
type HEGYOpts struct {
	// Deterministic chooses which deterministic terms enter the
	// regression. Defaults to HEGYDetConstantTrend (uroot c(1,1,0)).
	// Pass one of HEGYDetConstant, HEGYDetConstantTrend,
	// HEGYDetConstantSeasDummies, HEGYDetConstantTrendSeasDummies.
	Deterministic HEGYDeterministic

	// LagMethod chooses how the lag-augmentation order is selected.
	// Defaults to HEGYLagAIC.
	LagMethod HEGYLagMethod

	// MaxLag is the upper bound for the lag search (HEGYLagAIC /
	// HEGYLagBIC). Defaults to 3 (uroot/forecast::nsdiffs default).
	MaxLag int

	// Lag is the fixed lag order used when LagMethod = HEGYLagFixed.
	Lag int
}

// HEGYTestFull runs the same auxiliary regression as HEGYTest and
// returns the verdict together with per-frequency raw statistics and
// p-values. The variadic opts argument allows non-default
// (deterministic, lag.method) configurations matching uroot::hegy.test.
//
// See HEGYResult for the field documentation. The Verdict field is
// computed from JointSeasonalPValue at the conventional 0.05 level —
// matching forecast::nsdiffs(test="hegy") for the default opts.
func HEGYTestFull(x []float64, m int, opts ...HEGYOpts) (*HEGYResult, error) {
	o := HEGYOpts{}
	if len(opts) > 0 {
		o = opts[0]
	}
	// Apply defaults.
	if o.Deterministic == (HEGYDeterministic{}) {
		o.Deterministic = HEGYDetConstantTrend
	}
	if _, err := o.Deterministic.code(); err != nil {
		return nil, err
	}
	if _, err := o.LagMethod.suffix(); err != nil {
		return nil, err
	}
	if o.MaxLag <= 0 {
		o.MaxLag = 3
	}
	if o.LagMethod == HEGYLagFixed && o.Lag < 0 {
		return nil, errors.New("HEGY: HEGYLagFixed requires Lag >= 0")
	}

	if m < 2 {
		return nil, errors.New("HEGY requires m >= 2")
	}
	if len(x) < 4*m {
		return nil, errors.New("HEGY requires at least 4 full seasonal cycles")
	}

	auxRows, err := hegyAuxRegressors(x, m)
	if err != nil {
		return nil, err
	}
	dmy := applyMSeasonalDiff(x, m)
	if len(dmy) != len(auxRows) {
		return nil, errors.New("HEGY: aux rows / dmy length mismatch")
	}

	bestStats, bestLag, err := hegySelectLag(dmy, auxRows, m, o)
	if err != nil {
		return nil, err
	}

	detailed, err := hegyOLSDetailed2(dmy, auxRows, m, bestLag, o.Deterministic)
	if err != nil {
		return nil, err
	}

	// uroot uses dfResid (residual df) as the "n" arg into the
	// response-surface tables — see hegy.test source. The table
	// features 1/n, 1/n², lag/n, etc. are calibrated against the
	// simulation residual df, not the raw sample size.
	n := bestStats.dfResid
	jointSeasTab, err := hegyTableID(hegyStatJointSeasonal, o.Deterministic, o.LagMethod)
	if err != nil {
		return nil, err
	}
	jointSeasPval := hegyRSpvalueFromTable(bestStats.fJointSeasonal, n, m, bestLag, jointSeasTab, true)

	jointAllTab, err := hegyTableID(hegyStatJointAll, o.Deterministic, o.LagMethod)
	if err != nil {
		return nil, err
	}
	jointAllPval := hegyRSpvalueFromTable(detailed.fJointAll, n, m, bestLag, jointAllTab, true)

	zeroTab, err := hegyTableID(hegyStatZero, o.Deterministic, o.LagMethod)
	if err != nil {
		return nil, err
	}
	zeroPval := hegyRSpvalueFromTable(detailed.tStatZero, n, m, bestLag, zeroTab, false)

	pairTab, err := hegyTableID(hegyStatPairF, o.Deterministic, o.LagMethod)
	if err != nil {
		return nil, err
	}
	pairPvals := make([]float64, len(detailed.pairFStats))
	for i, f := range detailed.pairFStats {
		pairPvals[i] = hegyRSpvalueFromTable(f, n, m, bestLag, pairTab, true)
	}

	verdict := 0
	if jointSeasPval > 0.05 {
		verdict = 1
	}

	res := &HEGYResult{
		Verdict:             verdict,
		M:                   m,
		BestLag:             bestLag,
		AIC:                 bestStats.aic,
		TStatZero:           detailed.tStatZero,
		TStatZeroPValue:     zeroPval,
		PairFStats:          detailed.pairFStats,
		PairFrequencies:     detailed.pairFreqs,
		PairPValues:         pairPvals,
		JointSeasonalF:      bestStats.fJointSeasonal,
		JointSeasonalPValue: jointSeasPval,
		JointAllF:           detailed.fJointAll,
		JointAllPValue:      jointAllPval,
	}
	if m%2 == 0 {
		nyq := detailed.tStatNyquist
		res.TStatNyquist = &nyq
		nyqTab, err := hegyTableID(hegyStatNyquist, o.Deterministic, o.LagMethod)
		if err != nil {
			return nil, err
		}
		nyqPval := hegyRSpvalueFromTable(nyq, n, m, bestLag, nyqTab, false)
		res.TStatNyquistPValue = &nyqPval
	}
	return res, nil
}

// hegyResult holds the test statistics from one HEGY auxiliary regression.
type hegyResult struct {
	fJointSeasonal float64 // joint F on (π_2 .. π_m), used for D verdict
	aic            float64 // AIC of the regression (for lag selection)
	bic            float64 // BIC of the regression (for HEGYLagBIC)
	dfResid        int     // residual degrees of freedom (rows - cols).
	// uroot's hegy.rs.pvalue receives dfResid as its "n" arg (NOT raw
	// sample size) — it parameterises the response-surface features
	// `1/n`, `1/n²`, etc. against simulation residual df.
}

// hegyDetailedResult holds the full per-frequency breakdown — populated
// once at the (AIC-)best lag for HEGYTestFull. Pair F-stats and t-stats
// share the regression's (X'X)⁻¹ so they're cheap once the OLS solve
// has run.
type hegyDetailedResult struct {
	tStatZero    float64
	tStatNyquist float64 // valid only when m is even
	pairFStats   []float64
	pairFreqs    []float64
	fJointAll    float64 // joint F on (π_1..π_m), uroot CFt
}

// hegyAuxRegressors constructs the m-column auxiliary-regressor matrix
// per uroot's `hegy.regressors`. Returns rows of shape (n - m) × m,
// already lagged by one (so row k contains Y_{i,m+k} for the regressor
// at time t = m+k+1). Direct port of R uroot's `hegy.regressors`:
//
//	ML[t, k]      = y_{t - k}           (k = 0..m-1)
//	ypi[t, 1]     = Σ_k ML[t, k]                              (frequency 0)
//	ypi[t, 2]     = Σ_k (-1)^k · ML[t, k]    if m even        (frequency π)
//	ypi[t, i]     = Σ_k cos(2π·j·(k+1)/m) · ML[t, k]          (cosine)
//	ypi[t, i+1]   = sinesign · Σ_k sin(2π·j·(k+1)/m) · ML[t, k] (sine)
//
// The cosine/sine pairs cycle through j=1..⌊(m-1)/2⌋ (or floor-isEven),
// flipping `sinesign` every ⌈m/4⌉ steps.
func hegyAuxRegressors(x []float64, m int) ([][]float64, error) {
	n := len(x)
	if n <= m {
		return nil, errors.New("HEGY: series too short for aux construction")
	}

	// Build ypi at full length n. Rows 0..m-1 will be invalid (need m
	// past observations); we drop them at the end.
	ypi := make([][]float64, n)
	for i := range ypi {
		ypi[i] = make([]float64, m)
	}
	isEven := m%2 == 0
	isEvenP2 := 2
	if isEven {
		isEvenP2 = 3
	}

	// ML[t, k] = x[t - k] for k = 0..m-1, valid when t >= m-1.
	// We fill only valid rows.
	for t := m - 1; t < n; t++ {
		// Y_1 (frequency 0): sum.
		s := 0.0
		for k := 0; k < m; k++ {
			s += x[t-k]
		}
		ypi[t][0] = s
		// Y_2 (frequency π): alternating sum, only when m is even.
		// uroot uses pattern rep(c(-1, 1), len=S) which is [-1, 1, -1, 1, ...].
		if isEven {
			alt := 0.0
			for k := 0; k < m; k++ {
				if k%2 == 0 {
					alt -= x[t-k]
				} else {
					alt += x[t-k]
				}
			}
			ypi[t][1] = alt
		}
	}

	// Complex-conjugate pairs: j = 1, 2, ..., flipping sinesign every
	// ⌈m/4⌉ steps. Indices i = isEvenP2, isEvenP2+2, ..., m-1 (step 2),
	// each pair filling columns i and i+1.
	j := 0
	sinesign := -1.0
	ref := int(math.Ceil(float64(m) / 4.0))
	for i := isEvenP2; i < m+1; i += 2 {
		j++
		// Pre-compute cos/sin weights.
		cosW := make([]float64, m)
		sinW := make([]float64, m)
		for k := 0; k < m; k++ {
			angle := 2.0 * math.Pi * float64(j) * float64(k+1) / float64(m)
			cosW[k] = math.Cos(angle)
			sinW[k] = math.Sin(angle)
		}
		// Apply to each valid row.
		for t := m - 1; t < n; t++ {
			cs := 0.0
			ss := 0.0
			for k := 0; k < m; k++ {
				cs += cosW[k] * x[t-k]
				ss += sinW[k] * x[t-k]
			}
			ypi[t][i-1] = cs
			if i < m {
				ypi[t][i] = sinesign * ss
			}
		}
		if j == ref {
			sinesign = -1.0 * sinesign
		}
	}

	// uroot does `ypi <- rbind(NA, ypi[-n, ])` — i.e., shift the whole
	// matrix down by 1 (everyone's lagged by 1 timestep). Then drop the
	// first m rows. So the output corresponds to auxiliary variables at
	// time t-1, indexed by t = m+1 .. n-1 → row count = n - m.
	out := make([][]float64, n-m)
	for k := range out {
		// Output row k corresponds to t = m+k (1-indexed in R, 0-indexed
		// here). The lag-1 means we read ypi[t-1] = ypi[m+k-1].
		idx := m + k - 1
		if idx < m-1 {
			// Shouldn't happen given k >= 0 and m >= 4.
			return nil, errors.New("HEGY: internal aux-row indexing error")
		}
		out[k] = make([]float64, m)
		copy(out[k], ypi[idx])
	}
	return out, nil
}

// applyMSeasonalDiff is defined in stl.go-adjacent space; redefine here
// since the existing `applyDiff` in goarima takes lag-and-differences as
// separate args. Δ^m y_t = y_{t+m} - y_t with output indexed at t.
func applyMSeasonalDiff(y []float64, m int) []float64 {
	if len(y) <= m {
		return nil
	}
	out := make([]float64, len(y)-m)
	for i := range out {
		out[i] = y[i+m] - y[i]
	}
	return out
}

// hegySelectLag picks the lag order according to opts.LagMethod and
// returns the regression statistics at the chosen lag, for the given
// deterministic configuration. AIC/BIC scan p ∈ [0, opts.MaxLag];
// HEGYLagFixed runs a single OLS at opts.Lag.
func hegySelectLag(dmy []float64, auxRows [][]float64, m int, opts HEGYOpts) (hegyResult, int, error) {
	if opts.LagMethod == HEGYLagFixed {
		stats, err := hegyOLSConfig(dmy, auxRows, m, opts.Lag, opts.Deterministic)
		if err != nil {
			return hegyResult{}, 0, err
		}
		return stats, opts.Lag, nil
	}
	bestScore := math.Inf(1)
	var bestStats hegyResult
	bestP := 0
	var lastErr error
	for p := 0; p <= opts.MaxLag; p++ {
		stats, err := hegyOLSConfig(dmy, auxRows, m, p, opts.Deterministic)
		if err != nil {
			lastErr = err
			continue
		}
		score := stats.aic
		if opts.LagMethod == HEGYLagBIC {
			score = stats.bic
		}
		if score < bestScore {
			bestScore = score
			bestStats = stats
			bestP = p
		}
	}
	if math.IsInf(bestScore, 1) {
		return hegyResult{}, 0, errors.New("HEGY: no lag selection succeeded: " + lastErr.Error())
	}
	return bestStats, bestP, nil
}

// hegyBuildXY constructs the design matrix and dependent vector for
// the HEGY auxiliary regression at lag p with the given deterministic
// configuration. Layout:
//
//	cols 0..ndet-1:        deterministic (constant, trend, seasonal dummies)
//	cols ndet..ndet+m-1:   π_1..π_m  (the seasonal regressors)
//	cols ndet+m..end:      p lag-augmentation terms Δ^m y_{t-j}
//
// Returns piStart = ndet (column index of π_1), useful for downstream
// joint-F / per-coefficient stats.
func hegyBuildXY(dmy []float64, auxRows [][]float64, m, p int, det HEGYDeterministic) (X [][]float64, yt []float64, piStart int, err error) {
	nDmy := len(dmy)
	if nDmy <= 0 {
		err = errors.New("HEGY: dmy is empty")
		return
	}
	startDmy := p
	rows := nDmy - p

	ndet := 0
	if det.hasConstant() {
		ndet++
	}
	if det.hasTrend() {
		ndet++
	}
	if det.hasSeasonalDummies() {
		ndet += m - 1
	}
	cols := ndet + m + p
	if rows <= cols+1 {
		err = errors.New("HEGY: not enough observations after lag augmentation")
		return
	}

	piStart = ndet
	X = make([][]float64, rows)
	yt = make([]float64, rows)
	for i := 0; i < rows; i++ {
		row := make([]float64, cols)
		c := 0
		if det.hasConstant() {
			row[c] = 1
			c++
		}
		if det.hasTrend() {
			row[c] = float64(startDmy + i + 1)
			c++
		}
		if det.hasSeasonalDummies() {
			// (m-1) seasonal dummies: indicator I[(t mod m) == s] for
			// s = 0..m-2. The s = m-1 dummy is dropped to avoid
			// collinearity with the constant.
			t1 := startDmy + i + 1
			season := (t1 - 1) % m
			for s := 0; s < m-1; s++ {
				if season == s {
					row[c] = 1
				}
				c++
			}
		}
		for j := 0; j < m; j++ {
			row[c] = auxRows[startDmy+i][j]
			c++
		}
		for j := 0; j < p; j++ {
			row[c] = dmy[startDmy+i-1-j]
			c++
		}
		X[i] = row
		yt[i] = dmy[startDmy+i]
	}
	return
}

// hegyOLSConfig is the deterministic-config-aware variant of hegyOLS.
// Returns the joint F on (π_2..π_m), AIC, and BIC.
func hegyOLSConfig(dmy []float64, auxRows [][]float64, m, p int, det HEGYDeterministic) (hegyResult, error) {
	X, yt, piStart, err := hegyBuildXY(dmy, auxRows, m, p, det)
	if err != nil {
		return hegyResult{}, err
	}
	rows := len(X)
	cols := len(X[0])
	beta, err := olsFit(X, yt, false)
	if err != nil {
		return hegyResult{}, err
	}
	resid := make([]float64, rows)
	for i := 0; i < rows; i++ {
		pred := 0.0
		for j, b := range beta {
			pred += X[i][j] * b
		}
		resid[i] = yt[i] - pred
	}
	rss := 0.0
	for _, r := range resid {
		rss += r * r
	}
	if rss <= 0 || math.IsNaN(rss) {
		return hegyResult{}, errors.New("non-positive RSS in HEGY regression")
	}
	sigma2 := rss / float64(rows-cols)
	nf := float64(rows)
	aic := nf*math.Log(rss/nf) + 2*float64(cols)
	bic := nf*math.Log(rss/nf) + math.Log(nf)*float64(cols)

	// Joint F on (π_2..π_m): indices piStart+1 .. piStart+m-1.
	seasonalIdx := make([]int, m-1)
	for i := range seasonalIdx {
		seasonalIdx[i] = piStart + 1 + i
	}
	fSeason, err := hegyJointFStat(X, beta, sigma2, seasonalIdx)
	if err != nil {
		return hegyResult{}, err
	}
	return hegyResult{fJointSeasonal: fSeason, aic: aic, bic: bic, dfResid: rows - cols}, nil
}


// hegyOLSDetailed2 re-runs the regression at the chosen lag p with the
// given deterministic config and returns the per-frequency breakdown.
// PG-114: generalized over deterministic; PG-115a: per-frequency stats.
//
// Frequency layout (cf. hegyAuxRegressors), where piStart = column of π_1:
//   - even m: cols piStart, piStart+1 are π_1, π_2; pairs start at piStart+2
//   - odd m:  col piStart is π_1; pairs start at piStart+1
func hegyOLSDetailed2(dmy []float64, auxRows [][]float64, m, p int, det HEGYDeterministic) (hegyDetailedResult, error) {
	X, yt, piStart, err := hegyBuildXY(dmy, auxRows, m, p, det)
	if err != nil {
		return hegyDetailedResult{}, err
	}
	rows := len(X)
	cols := len(X[0])
	beta, err := olsFit(X, yt, false)
	if err != nil {
		return hegyDetailedResult{}, err
	}
	resid := make([]float64, rows)
	for i := 0; i < rows; i++ {
		pred := 0.0
		for j, b := range beta {
			pred += X[i][j] * b
		}
		resid[i] = yt[i] - pred
	}
	rss := 0.0
	for _, r := range resid {
		rss += r * r
	}
	if rss <= 0 || math.IsNaN(rss) {
		return hegyDetailedResult{}, errors.New("HEGY: non-positive RSS in detailed regression")
	}
	sigma2 := rss / float64(rows-cols)

	xtxInv, err := hegyXtxInverse(X)
	if err != nil {
		return hegyDetailedResult{}, err
	}

	seAt := func(k int) float64 {
		return math.Sqrt(sigma2 * xtxInv[k][k])
	}

	out := hegyDetailedResult{}
	// π_1 t-stat at col piStart.
	out.tStatZero = beta[piStart] / seAt(piStart)

	isEven := m%2 == 0
	var pairCount int
	var pairStartCol int
	if isEven {
		out.tStatNyquist = beta[piStart+1] / seAt(piStart + 1)
		pairCount = (m - 2) / 2
		pairStartCol = piStart + 2
	} else {
		pairCount = (m - 1) / 2
		pairStartCol = piStart + 1
	}

	out.pairFStats = make([]float64, pairCount)
	out.pairFreqs = make([]float64, pairCount)
	for k := 0; k < pairCount; k++ {
		col := pairStartCol + 2*k
		fStat, err := hegyJointFStatFromInv(xtxInv, beta, sigma2, []int{col, col + 1})
		if err != nil {
			return hegyDetailedResult{}, err
		}
		out.pairFStats[k] = fStat
		out.pairFreqs[k] = 2 * math.Pi * float64(k+1) / float64(m)
	}

	// Joint F on all m seasonal coefficients (CFt — π_1..π_m).
	allIdx := make([]int, m)
	for i := range allIdx {
		allIdx[i] = piStart + i
	}
	fAll, err := hegyJointFStatFromInv(xtxInv, beta, sigma2, allIdx)
	if err != nil {
		return hegyDetailedResult{}, err
	}
	out.fJointAll = fAll
	return out, nil
}

// hegyJointFStat computes the F-statistic for the joint hypothesis that
// the coefficients at indices `idx` are zero. Uses the linear-restriction
// form: F = (Rβ)' [R (X'X)⁻¹ R']⁻¹ (Rβ) / (q · σ²).
func hegyJointFStat(X [][]float64, beta []float64, sigma2 float64, idx []int) (float64, error) {
	xtxInv, err := hegyXtxInverse(X)
	if err != nil {
		return 0, err
	}
	return hegyJointFStatFromInv(xtxInv, beta, sigma2, idx)
}

// hegyJointFStatFromInv is the inner core of hegyJointFStat for callers
// that already hold a (X'X)⁻¹ — avoids redundant inversion when several
// joint F's are needed from the same regression (per-pair stats).
func hegyJointFStatFromInv(xtxInv [][]float64, beta []float64, sigma2 float64, idx []int) (float64, error) {
	q := len(idx)
	rb := make([]float64, q)
	for i, k := range idx {
		rb[i] = beta[k]
	}
	sub := make([][]float64, q)
	for i, ki := range idx {
		row := make([]float64, q)
		for j, kj := range idx {
			row[j] = xtxInv[ki][kj]
		}
		sub[i] = row
	}
	subInv, err := hegyInvertSym(sub)
	if err != nil {
		return 0, err
	}
	v := make([]float64, q)
	for i := 0; i < q; i++ {
		s := 0.0
		for j := 0; j < q; j++ {
			s += subInv[i][j] * rb[j]
		}
		v[i] = s
	}
	dot := 0.0
	for i := range rb {
		dot += rb[i] * v[i]
	}
	return dot / (float64(q) * sigma2), nil
}

func hegyXtxInverse(X [][]float64) ([][]float64, error) {
	rows := len(X)
	cols := len(X[0])
	xtx := make([][]float64, cols)
	for i := range xtx {
		xtx[i] = make([]float64, cols)
	}
	for i := 0; i < cols; i++ {
		for j := 0; j < cols; j++ {
			s := 0.0
			for r := 0; r < rows; r++ {
				s += X[r][i] * X[r][j]
			}
			xtx[i][j] = s
		}
	}
	return hegyInvertSym(xtx)
}

func hegyInvertSym(A [][]float64) ([][]float64, error) {
	n := len(A)
	aug := make([][]float64, n)
	for i := range aug {
		aug[i] = make([]float64, 2*n)
		for j := 0; j < n; j++ {
			aug[i][j] = A[i][j]
		}
		aug[i][n+i] = 1
	}
	for i := 0; i < n; i++ {
		pivot := aug[i][i]
		if math.Abs(pivot) < 1e-14 {
			swapped := false
			for r := i + 1; r < n; r++ {
				if math.Abs(aug[r][i]) > 1e-14 {
					aug[i], aug[r] = aug[r], aug[i]
					pivot = aug[i][i]
					swapped = true
					break
				}
			}
			if !swapped {
				return nil, errors.New("HEGY: singular matrix")
			}
		}
		for j := 0; j < 2*n; j++ {
			aug[i][j] /= pivot
		}
		for k := 0; k < n; k++ {
			if k == i {
				continue
			}
			factor := aug[k][i]
			if factor == 0 {
				continue
			}
			for j := 0; j < 2*n; j++ {
				aug[k][j] -= factor * aug[i][j]
			}
		}
	}
	out := make([][]float64, n)
	for i := range out {
		out[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			out[i][j] = aug[i][n+j]
		}
	}
	return out, nil
}

// hegyRSpvalueFromTable is the table-driven core of uroot's
// hegy.rs.pvalue (uroot 2.1.3). Selects between F-test and t-test
// branches based on isFtest:
//   - F-test (CF, CFs, CFt tables): qchisq(df=2) for Y, reverses Q1,
//     pchisq lower.tail=FALSE for the final p-value.
//   - t-test (Ct1, Ct2 tables): qnorm for Y, no Q1 reversal, pnorm
//     lower.tail=TRUE for the final p-value.
//
// Returns the p-value in [0, 1]. Edge cases: F-test returns 1 if stat
// is below all quantiles (high p-value, fail to reject H0=unit roots);
// t-test returns 0 in that case (low p-value, reject H0=unit root).
//
// Algorithm (per Diaz-Emparanza & Carlomagno 2010 / uroot 2.x):
//
//   1. xeplc = [1, 1/n, 1/n², 1/n³, lag/n, ..., lag³/n³, S/n, S/n², S/n³] (16-dim).
//   2. Q1 = C1 · xeplc — predicted statistic at each of 221 quantile levels.
//      Reversed iff isFtest.
//   3. Find local 15-point window of (Q1, rq, sd) around stat.
//   4. GLS-fit a cubic in qi of qchisq(df=2)/qnorm(pri).
//   5. p-value from cubic(stat).
func hegyRSpvalueFromTable(stat float64, n, m, lag int, tableID hegyRSTableID, isFtest bool) float64 {
	const nobsreg = 15
	const featureCount = 16
	const nrq = 221

	tab := &hegyRSTables[tableID]

	// Build xeplc: [1, 1/n, 1/n², 1/n³, lag/n, lag/n², lag/n³,
	//               lag²/n, lag²/n², lag²/n³,
	//               lag³/n, lag³/n², lag³/n³,
	//               S/n, S/n², S/n³]
	nf := float64(n)
	lf := float64(lag)
	sf := float64(m)
	xeplc := [featureCount]float64{
		1,
		1 / nf, 1 / (nf * nf), 1 / (nf * nf * nf),
		lf / nf, lf / (nf * nf), lf / (nf * nf * nf),
		lf * lf / nf, lf * lf / (nf * nf), lf * lf / (nf * nf * nf),
		lf * lf * lf / nf, lf * lf * lf / (nf * nf), lf * lf * lf / (nf * nf * nf),
		sf / nf, sf / (nf * nf), sf / (nf * nf * nf),
	}

	q1 := make([]float64, nrq)
	sd := make([]float64, nrq)
	for i := 0; i < nrq; i++ {
		s := 0.0
		for j := 0; j < featureCount; j++ {
			s += tab[i][j] * xeplc[j]
		}
		q1[i] = s
		sd[i] = tab[i][16]
	}
	if isFtest {
		// uroot reverses Q1 + sdC1 for F-tests (rq remains unreversed
		// but indexed differently).
		reverseFloat64s(q1)
		reverseFloat64s(sd)
	}

	q1Sorted := append([]float64(nil), q1...)
	sortFloat64s(q1Sorted)
	if stat < q1Sorted[0] {
		// uroot: F-test → return 1; t-test → return 0.
		if isFtest {
			return 1
		}
		return 0
	}
	if stat > q1Sorted[nrq-1] {
		if isFtest {
			return 0
		}
		return 1
	}

	masque := -1
	for i := 0; i < nrq; i++ {
		if stat > q1[i] {
			masque = i
		}
	}
	if masque < 0 {
		masque = 0
	}
	mascer := masque
	if masque < nrq-1 {
		if math.Abs(stat-q1[masque]) > math.Abs(q1[masque+1]-stat) {
			mascer = masque + 1
		}
	}
	centro := nobsreg/2 + 1
	centroup := nrq - centro + 1
	if mascer < centro {
		mascer = centro
	}
	if mascer > centroup {
		mascer = centroup
	}

	qi := make([]float64, nobsreg)
	pri := make([]float64, nobsreg)
	si := make([]float64, nobsreg)
	for i := 0; i < nobsreg; i++ {
		j := mascer - centro + i
		idx := j - 1
		if idx < 0 {
			idx = 0
		}
		if idx >= nrq {
			idx = nrq - 1
		}
		qi[i] = q1[idx]
		// uroot pairs Q1[j] (whether reversed or not) with rq[j]
		// directly — see hegy.rs.pvalue source. Same indexing for
		// both F- and t-test branches.
		pri[i] = hegyRSQuantiles[idx]
		si[i] = sd[idx]
	}

	Y := make([]float64, nobsreg)
	if isFtest {
		// qchisq(p, df=2) = -2·ln(1 - p) — exact for χ² df=2.
		for i := 0; i < nobsreg; i++ {
			Y[i] = -2.0 * math.Log(1.0-pri[i])
		}
	} else {
		// qnorm — standard normal inverse CDF.
		for i := 0; i < nobsreg; i++ {
			Y[i] = qnormInvCDF(pri[i])
		}
	}

	// X = [1, qi, qi², qi³] — design matrix for cubic fit.
	X := make([][]float64, nobsreg)
	for i := 0; i < nobsreg; i++ {
		X[i] = []float64{1, qi[i], qi[i] * qi[i], qi[i] * qi[i] * qi[i]}
	}

	// Build covariance matrix Σ_ij = si[i]*si[j] *
	//   sqrt( (pri[min(i,j)] * (1 - pri[max(i,j)])) /
	//         (pri[max(i,j)] * (1 - pri[min(i,j)])) )
	sigma := make([][]float64, nobsreg)
	for i := range sigma {
		sigma[i] = make([]float64, nobsreg)
	}
	for i := 0; i < nobsreg; i++ {
		for j := 0; j < nobsreg; j++ {
			lo := i
			hi := j
			if i > j {
				lo, hi = j, i
			}
			ratio := (pri[lo] * (1 - pri[hi])) / (pri[hi] * (1 - pri[lo]))
			sigma[i][j] = si[i] * si[j] * math.Sqrt(ratio)
		}
	}

	// Cholesky decomposition: Σ = L · L'. We need P_inv = (L')^(-1) so
	// that P_inv · Σ · P_inv' = I — i.e., GLS whitening.
	L, err := hegyCholesky(sigma)
	if err != nil {
		// Cholesky failed → fall back to ordinary least squares.
		return hegyOLSCubicPvalue(qi, Y, stat, isFtest)
	}
	// Solve L · z = b for z (forward substitution), then L' · y = z
	// (back substitution). For our purposes: PY = L^(-1) · Y, PX =
	// L^(-1) · X (each column of X). Then OLS on (PX, PY).
	PY := hegyForwardSub(L, Y)
	PX := make([][]float64, nobsreg)
	for i := 0; i < nobsreg; i++ {
		PX[i] = make([]float64, 4)
	}
	for col := 0; col < 4; col++ {
		colVec := make([]float64, nobsreg)
		for i := 0; i < nobsreg; i++ {
			colVec[i] = X[i][col]
		}
		whitenedCol := hegyForwardSub(L, colVec)
		for i := 0; i < nobsreg; i++ {
			PX[i][col] = whitenedCol[i]
		}
	}
	// OLS: β = (PX' · PX)^(-1) · PX' · PY.
	co, err := hegyOLSCoefs(PX, PY)
	if err != nil {
		return hegyOLSCubicPvalue(qi, Y, stat, isFtest)
	}
	// valorcomp = β · [1, x, x², x³]
	xs := stat
	valorcomp := co[0] + co[1]*xs + co[2]*xs*xs + co[3]*xs*xs*xs
	// F-test:  pchisq(|valorcomp|, df=2, lower.tail=FALSE) = exp(-|x|/2).
	// t-test:  pnorm(valorcomp, lower.tail=TRUE).
	var p float64
	if isFtest {
		p = math.Exp(-math.Abs(valorcomp) / 2.0)
	} else {
		p = pnormCDF(valorcomp)
	}
	if p < 0 {
		p = 0
	}
	if p > 1 {
		p = 1
	}
	return p
}

// hegyOLSCubicPvalue is the fallback when Cholesky fails — fits a
// plain OLS cubic and returns the F-test χ² df=2 or t-test pnorm p-value.
func hegyOLSCubicPvalue(qi, Y []float64, stat float64, isFtest bool) float64 {
	X := make([][]float64, len(qi))
	for i := range qi {
		X[i] = []float64{1, qi[i], qi[i] * qi[i], qi[i] * qi[i] * qi[i]}
	}
	co, err := hegyOLSCoefs(X, Y)
	if err != nil {
		return 0.5
	}
	xs := stat
	valorcomp := co[0] + co[1]*xs + co[2]*xs*xs + co[3]*xs*xs*xs
	if isFtest {
		return math.Exp(-math.Abs(valorcomp) / 2.0)
	}
	return pnormCDF(valorcomp)
}

// qnormInvCDF returns the inverse standard-normal CDF (the quantile
// function). Direct port of the Wichura (1988) AS241 algorithm —
// double-precision accurate (~16 digits) over (0, 1). uroot uses
// R's qnorm() which calls qnorm_DPQ → AS241.
func qnormInvCDF(p float64) float64 {
	if p <= 0 {
		return math.Inf(-1)
	}
	if p >= 1 {
		return math.Inf(1)
	}
	q := p - 0.5
	var r, val float64
	if math.Abs(q) <= 0.425 {
		// Central region: AS241 uses r = 0.180625 - q² (NOT q²).
		r = 0.180625 - q*q
		val = q * (((((((2509.0809287301226727*r+33430.575583588128105)*r+
			67265.770927008700853)*r+45921.953931549871457)*r+
			13731.693765509461125)*r+1971.5909503065514427)*r+
			133.14166789178437745)*r + 3.387132872796366608) /
			(((((((5226.495278852854561*r+28729.085735721942674)*r+
				39307.89580009271061)*r+21213.794301586595867)*r+
				5394.1960214247511077)*r+687.1870074920579083)*r+
				42.313330701600911252)*r + 1.0)
		return val
	}
	// Tail region.
	if q < 0 {
		r = p
	} else {
		r = 1 - p
	}
	r = math.Sqrt(-math.Log(r))
	if r <= 5 {
		r -= 1.6
		val = (((((((7.74545014278341407640e-4*r+0.0227238449892691845833)*r+
			0.24178072517745061177)*r+1.27045825245236838258)*r+
			3.64784832476320460504)*r+5.7694972214606914055)*r+
			4.6303378461565452959)*r + 1.42343711074968357734) /
			(((((((1.05075007164441684324e-9*r+5.475938084995344946e-4)*r+
				0.0151986665636164571966)*r+0.14810397642748007459)*r+
				0.68976733498510000455)*r+1.6763848301838038494)*r+
				2.05319162663775882187)*r + 1.0)
	} else {
		r -= 5
		val = (((((((2.01033439929228813265e-7*r+2.71155556874348757815e-5)*r+
			0.0012426609473880784386)*r+0.026532189526576123093)*r+
			0.29656057182850489123)*r+1.7848265399172913358)*r+
			5.4637849111641143699)*r + 6.6579046435011037772) /
			(((((((2.04426310338993978564e-15*r+1.4215117583164458887e-7)*r+
				1.8463183175100546818e-5)*r+7.868691311456132591e-4)*r+
				0.0148753612908506148525)*r+0.13692988092273580531)*r+
				0.59983220655588793769)*r + 1.0)
	}
	if q < 0 {
		val = -val
	}
	return val
}

// pnormCDF returns the standard-normal CDF Φ(x) — used for the t-test
// branch's p-value (uroot: pnorm(valorcomp, lower.tail=TRUE)).
func pnormCDF(x float64) float64 {
	return 0.5 * math.Erfc(-x/math.Sqrt2)
}

// hegyCholesky returns the lower-triangular L such that L · L' = A.
func hegyCholesky(A [][]float64) ([][]float64, error) {
	n := len(A)
	L := make([][]float64, n)
	for i := range L {
		L[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			s := A[i][j]
			for k := 0; k < j; k++ {
				s -= L[i][k] * L[j][k]
			}
			if i == j {
				if s <= 0 {
					return nil, errors.New("HEGY: Cholesky failed (non-PSD covariance)")
				}
				L[i][j] = math.Sqrt(s)
			} else {
				L[i][j] = s / L[j][j]
			}
		}
	}
	return L, nil
}

// hegyForwardSub solves L · y = b for y given lower-triangular L.
func hegyForwardSub(L [][]float64, b []float64) []float64 {
	n := len(b)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		s := b[i]
		for k := 0; k < i; k++ {
			s -= L[i][k] * y[k]
		}
		y[i] = s / L[i][i]
	}
	return y
}

// hegyOLSCoefs solves min ||X β - y||² via the normal equations.
func hegyOLSCoefs(X [][]float64, y []float64) ([]float64, error) {
	rows := len(X)
	cols := len(X[0])
	xtx := make([][]float64, cols)
	for i := range xtx {
		xtx[i] = make([]float64, cols)
	}
	xty := make([]float64, cols)
	for r := 0; r < rows; r++ {
		for i := 0; i < cols; i++ {
			xty[i] += X[r][i] * y[r]
			for j := 0; j < cols; j++ {
				xtx[i][j] += X[r][i] * X[r][j]
			}
		}
	}
	inv, err := hegyInvertSym(xtx)
	if err != nil {
		return nil, err
	}
	out := make([]float64, cols)
	for i := 0; i < cols; i++ {
		s := 0.0
		for j := 0; j < cols; j++ {
			s += inv[i][j] * xty[j]
		}
		out[i] = s
	}
	return out, nil
}

func reverseFloat64s(a []float64) {
	for i, j := 0, len(a)-1; i < j; i, j = i+1, j-1 {
		a[i], a[j] = a[j], a[i]
	}
}

func sortFloat64s(a []float64) {
	// Simple insertion sort — good enough for n=221.
	for i := 1; i < len(a); i++ {
		v := a[i]
		j := i - 1
		for j >= 0 && a[j] > v {
			a[j+1] = a[j]
			j--
		}
		a[j+1] = v
	}
}

