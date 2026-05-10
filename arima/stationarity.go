package arima

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/mat"
)

// tseriesPpSum is the Newey-West-style autocovariance sum used by KPSS and PP.
//
// Mirrors C_tseries_pp_sum from pmdarima/arima/_arima.pyx.
func tseriesPpSum(u []float64, n, L int, s float64) float64 {
	tmp1 := 0.0
	for i := 1; i <= L; i++ {
		tmp2 := 0.0
		for j := i; j < n; j++ {
			tmp2 += u[j] * u[j-i]
		}
		tmp2 *= 1.0 - float64(i)/(float64(L)+1.0)
		tmp1 += tmp2
	}
	tmp1 /= float64(n)
	tmp1 *= 2.0
	return s + tmp1
}

// embed lags x into a k-by-(n-k+1) matrix where row i has x lagged by (k-1-i).
//
// Matches the rows-form returned by _BaseStationarityTest._embed.
func embed(x []float64, k int) ([][]float64, error) {
	n := len(x)
	if k > n {
		return nil, errors.New("k cannot exceed y dim")
	}
	if k < 1 {
		return nil, errors.New("k must be >= 1")
	}
	rows := make([][]float64, k)
	cols := n - k + 1
	for i := 0; i < k; i++ {
		j := k - 1 - i
		row := make([]float64, cols)
		// x[j: n-i]
		for c := 0; c < cols; c++ {
			row[c] = x[j+c]
		}
		rows[i] = row
	}
	return rows, nil
}

// embedT returns the transpose of embed: cols=k, rows=n-k+1.
func embedT(x []float64, k int) ([][]float64, error) {
	r, err := embed(x, k)
	if err != nil {
		return nil, err
	}
	cols := len(r[0])
	out := make([][]float64, cols)
	for i := 0; i < cols; i++ {
		row := make([]float64, k)
		for j := 0; j < k; j++ {
			row[j] = r[j][i]
		}
		out[i] = row
	}
	return out, nil
}

// kpssTablePLevels returns the KPSS critical values for level (constant) null.
var kpssTablePLevels = []float64{0.739, 0.574, 0.463, 0.347}

// kpssTablePTrend returns the KPSS critical values for trend null.
var kpssTablePTrend = []float64{0.216, 0.176, 0.146, 0.119}

// kpssTableP holds the corresponding p-values.
var kpssTableP = []float64{0.01, 0.025, 0.05, 0.10}

// KPSSResult is the (pvalue, shouldDiff) outcome of a KPSS test.
type KPSSResult struct {
	PValue     float64
	ShouldDiff bool
	Stat       float64
}

// KPSSTestOpts groups the KPSS test parameters.
type KPSSTestOpts struct {
	Alpha  float64 // significance level (default 0.05)
	Null   string  // "level" or "trend" (default "level")
	LShort bool    // truncation rule (default true)
	// UseLag, when > 0, sets the Newey-West lag truncation explicitly,
	// overriding LShort. This is the knob R's `urca::ur.kpss` exposes
	// (`use.lag`); `forecast::ndiffs(test="kpss")` calls into it with
	// `trunc(3 * sqrt(n) / 13)` rather than the tseries `lshort` rule.
	// Direct callers wanting tseries-style behaviour can leave UseLag=0.
	UseLag int
}

// KPSSTest implements the Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
func KPSSTest(x []float64, opts KPSSTestOpts) (KPSSResult, error) {
	if opts.Alpha == 0 {
		opts.Alpha = 0.05
	}
	if opts.Null == "" {
		opts.Null = "level"
	}
	if len(x) == 0 {
		return KPSSResult{PValue: math.NaN(), ShouldDiff: false}, nil
	}
	n := len(x)
	var t [][]float64
	var table []float64
	addIntercept := false
	switch opts.Null {
	case "level":
		// x ~ 1 (intercept-only OLS): residuals = x - mean(x).
		t = make([][]float64, n)
		for i := range t {
			t[i] = []float64{1}
		}
		table = kpssTablePLevels
	case "trend":
		// x ~ 1 + t (R's lm(x ~ t) includes intercept): residuals from a
		// linear-trend regression. pmdarima uses sklearn LinearRegression
		// which defaults to fit_intercept=True.
		t = make([][]float64, n)
		for i := range t {
			t[i] = []float64{float64(i)}
		}
		table = kpssTablePTrend
		addIntercept = true
	default:
		return KPSSResult{}, errors.New(`null must be "level" or "trend"`)
	}
	beta, err := olsFit(t, x, addIntercept)
	if err != nil {
		return KPSSResult{}, err
	}
	resid := make([]float64, n)
	for i := 0; i < n; i++ {
		pred := 0.0
		off := 0
		if addIntercept {
			pred = beta[0]
			off = 1
		}
		for j, v := range t[i] {
			pred += v * beta[j+off]
		}
		resid[i] = x[i] - pred
	}
	cumS := 0.0
	eta := 0.0
	for _, e := range resid {
		cumS += e
		eta += cumS * cumS
	}
	eta /= float64(n) * float64(n)
	s2 := 0.0
	for _, e := range resid {
		s2 += e * e
	}
	s2 /= float64(n)
	var l int
	switch {
	case opts.UseLag > 0:
		l = opts.UseLag
	case opts.LShort:
		l = int(math.Trunc(4 * math.Pow(float64(n)/100, 0.25)))
	default:
		l = int(math.Trunc(12 * math.Pow(float64(n)/100, 0.25)))
	}
	s2 = tseriesPpSum(resid, n, l, s2)
	stat := eta / s2
	out, err := Approx(table, kpssTableP, []float64{stat}, Linear, RuleClip)
	if err != nil {
		return KPSSResult{}, err
	}
	pval := out[0]
	return KPSSResult{PValue: pval, ShouldDiff: pval < opts.Alpha, Stat: stat}, nil
}

// adfTable holds the augmented Dickey-Fuller τ_τ critical-value matrix
// (intercept + linear trend regression — the "ct" / "trend" formulation,
// matching `tseries::adf.test`). Rows = sample sizes (adfTableT); cols =
// p-values (adfTableP). Source: pmdarima/arima/stationarity.py /
// Banerjee, Dolado, Galbraith, Hendry 1993 Table 4.2.
var adfTable = [][]float64{
	{-4.38, -3.95, -3.60, -3.24, -1.14, -0.80, -0.50, -0.15},
	{-4.15, -3.80, -3.50, -3.18, -1.19, -0.87, -0.58, -0.24},
	{-4.04, -3.73, -3.45, -3.15, -1.22, -0.90, -0.62, -0.28},
	{-3.99, -3.69, -3.43, -3.13, -1.23, -0.92, -0.64, -0.31},
	{-3.98, -3.68, -3.42, -3.13, -1.24, -0.93, -0.65, -0.32},
	{-3.96, -3.66, -3.41, -3.12, -1.25, -0.94, -0.66, -0.33},
}

// adfTableMu holds the τ_μ critical-value matrix for the ADF "drift"
// formulation (intercept only, no trend — matches `urca::ur.df(...,
// type="drift")` which is what R's `forecast::ndiffs(test="adf")` uses
// by default). Source: Banerjee, Dolado, Galbraith, Hendry 1993
// Table 4.1.
//
// Rows: n = 25, 50, 100, 250, 500, ∞ (same grid as adfTableT).
// Cols: p = 0.01, 0.025, 0.05, 0.10, 0.90, 0.95, 0.975, 0.99
//       (same grid as adfTableP).
var adfTableMu = [][]float64{
	{-3.75, -3.33, -3.00, -2.63, -0.37, 0.00, 0.34, 0.72},
	{-3.58, -3.22, -2.93, -2.60, -0.40, -0.03, 0.29, 0.66},
	{-3.51, -3.17, -2.89, -2.58, -0.42, -0.05, 0.26, 0.63},
	{-3.46, -3.14, -2.88, -2.57, -0.42, -0.06, 0.24, 0.62},
	{-3.44, -3.13, -2.87, -2.57, -0.43, -0.07, 0.24, 0.61},
	{-3.43, -3.12, -2.86, -2.57, -0.44, -0.07, 0.23, 0.60},
}

// adfTableT is the row labels (sample sizes).
var adfTableT = []float64{25, 50, 100, 250, 500, 100000}

// adfTableP is the column labels (p-values).
var adfTableP = []float64{0.01, 0.025, 0.05, 0.10, 0.90, 0.95, 0.975, 0.99}

// ADFResult is the (pvalue, shouldDiff) outcome of an ADF test.
type ADFResult struct {
	PValue     float64
	ShouldDiff bool
	Stat       float64
}

// ADFType selects the ADF auxiliary-regression formulation.
type ADFType int

const (
	// ADFDrift fits Δy_t = α + β·y_{t-1} + Σ φ_j·Δy_{t-j} + ε_t
	// (intercept only, no trend — the τ_μ formulation). Matches R's
	// `urca::ur.df(..., type="drift")`, which is what
	// `forecast::ndiffs(test="adf")` uses by default. Critical values
	// from `adfTableMu` (Banerjee et al. 1993 Table 4.1).
	ADFDrift ADFType = iota
	// ADFTrend fits Δy_t = α + β·y_{t-1} + γ·t + Σ φ_j·Δy_{t-j} + ε_t
	// (intercept + linear time trend — the τ_τ formulation). Matches
	// `tseries::adf.test`. Critical values from `adfTable` (Banerjee
	// et al. 1993 Table 4.2). Pre-PG-110 this was the only goarima
	// formulation.
	ADFTrend
)

// ADFTestOpts groups the Augmented Dickey-Fuller parameters.
type ADFTestOpts struct {
	Alpha float64 // significance level (default 0.05)
	K     int     // lag order; 0 → trunc((n-1)^(1/3))
	HasK  bool    // explicit K supplied
	// Type selects the auxiliary-regression formulation. **Default
	// `ADFDrift` (the zero value) for R `forecast::ndiffs(test="adf")`
	// parity** — pre-PG-110 the regression hardcoded τ_τ (drift +
	// trend) which diverged from R on trending series like
	// airpassengers. Set Type=ADFTrend to recover the pre-2026-05-07
	// behaviour or to match `tseries::adf.test`'s default.
	Type ADFType
}

// ADFTest implements the Augmented Dickey-Fuller stationarity test.
//
// Returns (p-value, shouldDiff) where shouldDiff = pvalue > alpha (matches pmdarima).
func ADFTest(x []float64, opts ADFTestOpts) (ADFResult, error) {
	if opts.Alpha == 0 {
		opts.Alpha = 0.05
	}
	if opts.HasK && opts.K < 0 {
		return ADFResult{}, errors.New("k must be a positive integer (>= 0)")
	}
	if len(x) == 0 {
		return ADFResult{PValue: math.NaN(), ShouldDiff: false}, nil
	}
	k := opts.K
	if !opts.HasK {
		k = int(math.Trunc(math.Pow(float64(len(x)-1), 1.0/3.0)))
	}
	k++
	yDiff, err := diffOnce(x)
	if err != nil {
		return ADFResult{}, err
	}
	n := len(yDiff)
	z, err := embed(yDiff, k) // z[r=0..k-1][c=0..n-k]
	if err != nil {
		return ADFResult{}, err
	}
	zT, err := embedT(yDiff, k)
	if err != nil {
		return ADFResult{}, err
	}
	yt := z[0]
	tt := make([]float64, n-k+1)
	for i := range tt {
		tt[i] = float64(k + i) // mirrors Python "tt += 1" after using as range mask
	}
	xt1 := make([]float64, len(tt))
	for i := range tt {
		xt1[i] = x[k-1+i]
	}
	rows := len(yt)
	X := make([][]float64, rows)
	// Branch on Type: τ_τ (default pre-PG-110) includes intercept +
	// y_{t-1} + time trend; τ_μ (default post-PG-110) drops the trend
	// column for intercept-only regression. The y_{t-1} coefficient
	// stays at index 1 either way so olsStdErr(X, resid, 1) below
	// targets the correct column.
	for i := 0; i < rows; i++ {
		var row []float64
		if opts.Type == ADFTrend {
			row = []float64{1, xt1[i], tt[i]}
		} else { // ADFDrift (zero value, default)
			row = []float64{1, xt1[i]}
		}
		if k > 1 {
			row = append(row, zT[i][1:k]...)
		}
		X[i] = row
	}
	beta, err := olsFit(X, yt, false)
	if err != nil {
		return ADFResult{}, err
	}
	resid := make([]float64, rows)
	for i := 0; i < rows; i++ {
		pred := 0.0
		for j, b := range beta {
			pred += X[i][j] * b
		}
		resid[i] = yt[i] - pred
	}
	// std error of beta[1]: sqrt(sigma^2 * (X^T X)^{-1}[1,1])
	stdErr, err := olsStdErr(X, resid, 1)
	if err != nil {
		return ADFResult{}, err
	}
	stat := beta[1] / stdErr

	// Pick the right critical-value table for the formulation.
	useTable := adfTableMu // τ_μ for ADFDrift
	if opts.Type == ADFTrend {
		useTable = adfTable // τ_τ for ADFTrend
	}
	// Interpolate critical table column-by-column at the sample size.
	tableiPL := make([]float64, len(adfTableP))
	for i := range adfTableP {
		col := make([]float64, len(useTable))
		for r := range useTable {
			col[r] = useTable[r][i]
		}
		out, err := Approx(adfTableT, col, []float64{float64(n)}, Linear, RuleClip)
		if err != nil {
			return ADFResult{}, err
		}
		tableiPL[i] = out[0]
	}
	out, err := Approx(tableiPL, adfTableP, []float64{stat}, Linear, RuleClip)
	if err != nil {
		return ADFResult{}, err
	}
	pval := out[0]
	return ADFResult{PValue: pval, ShouldDiff: pval > opts.Alpha, Stat: stat}, nil
}

// ppTable is the Phillips-Perron Z(alpha) critical-value matrix for the
// τ_τ formulation (intercept + linear trend), matching `tseries::pp.test`.
// Rows = sample sizes (ppTableT); cols = p-values (ppTableP).
var ppTable = [][]float64{
	{-22.5, -19.9, -17.9, -15.6, -3.66, -2.51, -1.53, -0.43},
	{-25.7, -22.4, -19.8, -16.8, -3.71, -2.60, -1.66, -0.65},
	{-27.4, -23.6, -20.7, -17.5, -3.74, -2.62, -1.73, -0.75},
	{-28.4, -24.4, -21.3, -18.0, -3.75, -2.64, -1.78, -0.82},
	{-28.9, -24.8, -21.5, -18.1, -3.76, -2.65, -1.78, -0.84},
	{-29.5, -25.1, -21.8, -18.3, -3.77, -2.66, -1.79, -0.87},
}

// PG-110b: PP Z(τ) form uses the same Dickey-Fuller asymptotic
// distribution as the ADF t-stat, so we can reuse the ADF critical-
// value tables (`adfTable` for τ_τ, `adfTableMu` for τ_μ). The Z(α)
// form has its own scale and would need ppTable / a τ_μ Z(α) table —
// goarima keeps Z(α) only as a non-default opt-in via PPZAlphaTrend.

// ppTableT is the sample-size grid.
var ppTableT = []float64{25, 50, 100, 250, 500, 100000}

// ppTableP is the p-value grid for ppTable columns.
var ppTableP = []float64{0.01, 0.025, 0.05, 0.10, 0.90, 0.95, 0.975, 0.99}

// PPResult is the Phillips-Perron test outcome.
type PPResult struct {
	PValue     float64
	ShouldDiff bool
	Stat       float64
}

// PPType selects the Phillips-Perron statistic and regression formulation.
type PPType int

const (
	// PPZTauDrift uses the Z(τ) statistic with intercept-only regression
	// (τ_μ). Matches R's `urca::ur.pp(type="Z-tau", model="constant")`,
	// which is what `forecast::ndiffs(test="pp")` uses by default. Zero
	// value, default — pre-PG-110b the only option was Z(α) with trend
	// (now PPZAlphaTrend) and verdicts diverged from R on trending series.
	PPZTauDrift PPType = iota
	// PPZTauTrend uses the Z(τ) statistic with intercept + linear trend
	// regression (τ_τ). Matches `tseries::pp.test` default.
	PPZTauTrend
	// PPZAlphaTrend uses the Z(α) statistic with intercept + linear trend
	// regression. Pre-PG-110b goarima default. Kept as opt-in for users
	// who specifically want this variant (the α form has historical
	// significance — Phillips-Perron 1988 original); the Z(τ) form is
	// what R's auto.arima path uses.
	PPZAlphaTrend
)

// PPTestOpts groups the Phillips-Perron test parameters.
type PPTestOpts struct {
	Alpha  float64
	LShort bool
	// Type selects the statistic and regression formulation. Default
	// PPZTauDrift (zero value) for R `forecast::ndiffs(test="pp")`
	// parity. Pre-PG-110b goarima hardcoded what's now PPZAlphaTrend.
	Type PPType
}

// PPTest implements the Phillips-Perron unit-root test.
//
// Default Type=PPZTauDrift uses the Z(τ) statistic with intercept-only
// regression — matches R's auto.arima path. Set Type=PPZAlphaTrend for
// the legacy Z(α) statistic with trend (pre-PG-110b goarima default,
// matches Phillips-Perron 1988 original).
func PPTest(x []float64, opts PPTestOpts) (PPResult, error) {
	if opts.Alpha == 0 {
		opts.Alpha = 0.05
	}
	if len(x) == 0 {
		return PPResult{PValue: math.NaN(), ShouldDiff: false}, nil
	}
	z, err := embed(x, 2)
	if err != nil {
		return PPResult{}, err
	}
	yt := z[0]
	yt1 := z[1]
	n := len(yt)
	tt := make([]float64, n)
	for i := range tt {
		tt[i] = float64(i+1) - float64(n)/2.0
	}
	// Branch on Type: τ_τ (PPZTauTrend, PPZAlphaTrend) includes intercept
	// + trend + y_{t-1}; τ_μ (PPZTauDrift) drops the trend column. The
	// y_{t-1} coefficient stays at the last position so the index used by
	// olsStdErr below is always (len(X[0]) - 1).
	X := make([][]float64, n)
	for i := 0; i < n; i++ {
		if opts.Type == PPZTauDrift {
			X[i] = []float64{1, yt1[i]}
		} else {
			X[i] = []float64{1, tt[i], yt1[i]}
		}
	}
	beta, err := olsFit(X, yt, false)
	if err != nil {
		return PPResult{}, err
	}
	resid := make([]float64, n)
	for i := 0; i < n; i++ {
		pred := 0.0
		for j, b := range beta {
			pred += X[i][j] * b
		}
		resid[i] = yt[i] - pred
	}
	ssqru := 0.0
	for _, u := range resid {
		ssqru += u * u
	}
	ssqru /= float64(n)
	scalar := 4.0
	if !opts.LShort {
		scalar = 12.0
	}
	l := int(math.Trunc(scalar * math.Pow(float64(n)/100.0, 0.25)))
	ssqrtl := tseriesPpSum(resid, n, l, ssqru)

	n2 := float64(n) * float64(n)
	syt11n := 0.0
	sumYt1Sq := 0.0
	sumYt1 := 0.0
	for i := 0; i < n; i++ {
		syt11n += yt1[i] * float64(i+1)
		sumYt1Sq += yt1[i] * yt1[i]
		sumYt1 += yt1[i]
	}
	trm1 := n2 * (n2 - 1) * sumYt1Sq / 12.0
	trm2 := float64(n) * (syt11n * syt11n)
	trm3 := float64(n) * float64(n+1) * syt11n * sumYt1
	trm4 := (float64(n) * float64(n+1) * float64(2*n+1) * sumYt1 * sumYt1) / 6.0
	dx := trm1 - trm2 + trm3 - trm4

	// alpha = OLS coefficient on y_{t-1} (last column of X).
	alphaIdx := len(X[0]) - 1
	alpha := beta[alphaIdx]

	var stat float64
	var critTable [][]float64
	var critP []float64
	var critT []float64

	switch opts.Type {
	case PPZTauDrift, PPZTauTrend:
		// PG-110b: Z(τ) statistic — t-stat form, matches R's
		// `urca::ur.pp(type="Z-tau")`. Reuses ADF τ_μ / τ_τ critical-
		// value tables since the Z(τ) asymptotic distribution is the
		// same Dickey-Fuller distribution as ADF's t-stat.
		//
		// Z_τ = sqrt(σ²_u / σ²_l) * t̂_α - 0.5 * (σ²_l - σ²_u) *
		//        sqrt(n) * SE(α̂) / (sqrt(σ²_l) * sqrt(σ²_u))
		seAlpha, seErr := olsStdErr(X, resid, alphaIdx)
		if seErr != nil {
			return PPResult{}, seErr
		}
		tAlpha := (alpha - 1) / seAlpha
		factor := math.Sqrt(ssqru / ssqrtl)
		correction := 0.5 * (ssqrtl - ssqru) * math.Sqrt(float64(n)) * seAlpha /
			(math.Sqrt(ssqrtl) * math.Sqrt(ssqru))
		stat = factor*tAlpha - correction
		if opts.Type == PPZTauDrift {
			critTable = adfTableMu
		} else {
			critTable = adfTable
		}
		critP = adfTableP
		critT = adfTableT
	default: // PPZAlphaTrend — legacy Z(α) form with trend
		stat = float64(n)*(alpha-1) - math.Pow(float64(n), 6)/(24.0*dx)*(ssqrtl-ssqru)
		critTable = ppTable
		critP = ppTableP
		critT = ppTableT
	}

	tableiPL := make([]float64, len(critP))
	for i := range critP {
		col := make([]float64, len(critTable))
		for r := range critTable {
			col[r] = critTable[r][i]
		}
		out, err := Approx(critT, col, []float64{float64(n)}, Linear, RuleClip)
		if err != nil {
			return PPResult{}, err
		}
		tableiPL[i] = out[0]
	}
	out, err := Approx(tableiPL, critP, []float64{stat}, Linear, RuleClip)
	if err != nil {
		return PPResult{}, err
	}
	pval := out[0]
	return PPResult{PValue: pval, ShouldDiff: pval > opts.Alpha, Stat: stat}, nil
}

// diffOnce returns the lag-1 first difference.
func diffOnce(x []float64) ([]float64, error) {
	if len(x) < 2 {
		return nil, errors.New("series too short to diff")
	}
	out := make([]float64, len(x)-1)
	for i := range out {
		out[i] = x[i+1] - x[i]
	}
	return out, nil
}

// olsFit solves min ||X b - y||^2 via QR.
// If addIntercept is true, prepends a constant column to X.
func olsFit(X [][]float64, y []float64, addIntercept bool) ([]float64, error) {
	n := len(X)
	if n == 0 || n != len(y) {
		return nil, errors.New("x/y dimension mismatch")
	}
	cols := len(X[0])
	totalCols := cols
	if addIntercept {
		totalCols++
	}
	flat := make([]float64, n*totalCols)
	for i, row := range X {
		off := i * totalCols
		if addIntercept {
			flat[off] = 1
			off++
		}
		for j, v := range row {
			flat[i*totalCols+j+boolToInt(addIntercept)] = v
		}
	}
	return olsFitDense(flat, n, totalCols, y)
}

// olsFitDense solves min ||X b - y||^2 via QR on a flat row-major design
// matrix (rows × cols). Caller is responsible for prepending the
// intercept column if needed. CDX-5: avoids the per-row alloc churn
// of building a [][]float64 only to flatten it inside olsFit.
func olsFitDense(flat []float64, rows, cols int, y []float64) ([]float64, error) {
	if rows == 0 || rows != len(y) {
		return nil, errors.New("x/y dimension mismatch")
	}
	A := mat.NewDense(rows, cols, flat)
	b := mat.NewVecDense(rows, append([]float64{}, y...))
	var qr mat.QR
	qr.Factorize(A)
	var beta mat.VecDense
	if err := qr.SolveVecTo(&beta, false, b); err != nil {
		return nil, err
	}
	out := make([]float64, cols)
	for i := 0; i < cols; i++ {
		out[i] = beta.AtVec(i)
	}
	return out, nil
}

// olsStdErr computes the standard error of beta[idx] from residuals & X.
func olsStdErr(X [][]float64, resid []float64, idx int) (float64, error) {
	n := len(X)
	cols := len(X[0])
	if n <= cols {
		return 0, errors.New("not enough degrees of freedom")
	}
	// sigma^2 = sum(resid^2) / (n - cols)
	sse := 0.0
	for _, r := range resid {
		sse += r * r
	}
	sigma2 := sse / float64(n-cols)

	// (X^T X)^{-1}
	flat := make([]float64, n*cols)
	for i, row := range X {
		for j, v := range row {
			flat[i*cols+j] = v
		}
	}
	A := mat.NewDense(n, cols, flat)
	var XtX mat.Dense
	XtX.Mul(A.T(), A)
	var inv mat.Dense
	if err := inv.Inverse(&XtX); err != nil {
		return 0, err
	}
	return math.Sqrt(sigma2 * inv.At(idx, idx)), nil
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
