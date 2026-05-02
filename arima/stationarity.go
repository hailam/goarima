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
}

// KPSSTestOpts groups the KPSS test parameters.
type KPSSTestOpts struct {
	Alpha  float64 // significance level (default 0.05)
	Null   string  // "level" or "trend" (default "level")
	LShort bool    // truncation rule (default true)
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
	switch opts.Null {
	case "level":
		t = make([][]float64, n)
		for i := range t {
			t[i] = []float64{1}
		}
		table = kpssTablePLevels
	case "trend":
		t = make([][]float64, n)
		for i := range t {
			t[i] = []float64{float64(i)}
		}
		table = kpssTablePTrend
	default:
		return KPSSResult{}, errors.New(`null must be "level" or "trend"`)
	}
	// fit OLS without intercept: x ~ t
	beta, err := olsFit(t, x, false)
	if err != nil {
		return KPSSResult{}, err
	}
	resid := make([]float64, n)
	for i := 0; i < n; i++ {
		pred := 0.0
		for j, b := range beta {
			pred += t[i][j] * b
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
	if opts.LShort {
		l = int(math.Trunc(4 * math.Pow(float64(n)/100, 0.25)))
	} else {
		l = int(math.Trunc(12 * math.Pow(float64(n)/100, 0.25)))
	}
	s2 = tseriesPpSum(resid, n, l, s2)
	stat := eta / s2
	out, err := Approx(table, kpssTableP, []float64{stat}, Linear, RuleClip)
	if err != nil {
		return KPSSResult{}, err
	}
	pval := out[0]
	return KPSSResult{PValue: pval, ShouldDiff: pval < opts.Alpha}, nil
}

// adfTable holds the augmented Dickey-Fuller critical-value matrix
// (rows = sample sizes; cols = significance levels), per pmdarima/arima/stationarity.py.
var adfTable = [][]float64{
	{-4.38, -3.95, -3.60, -3.24, -1.14, -0.80, -0.50, -0.15},
	{-4.15, -3.80, -3.50, -3.18, -1.19, -0.87, -0.58, -0.24},
	{-4.04, -3.73, -3.45, -3.15, -1.22, -0.90, -0.62, -0.28},
	{-3.99, -3.69, -3.43, -3.13, -1.23, -0.92, -0.64, -0.31},
	{-3.98, -3.68, -3.42, -3.13, -1.24, -0.93, -0.65, -0.32},
	{-3.96, -3.66, -3.41, -3.12, -1.25, -0.94, -0.66, -0.33},
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

// ADFTestOpts groups the Augmented Dickey-Fuller parameters.
type ADFTestOpts struct {
	Alpha float64 // significance level (default 0.05)
	K     int     // lag order; 0 → trunc((n-1)^(1/3))
	HasK  bool    // explicit K supplied
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
	for i := 0; i < rows; i++ {
		row := []float64{1, xt1[i], tt[i]}
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

	// Interpolate critical table column-by-column at the sample size.
	tableiPL := make([]float64, len(adfTableP))
	for i := range adfTableP {
		col := make([]float64, len(adfTable))
		for r := range adfTable {
			col[r] = adfTable[r][i]
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

// ppTable is the Phillips-Perron Z(alpha) critical-value matrix.
var ppTable = [][]float64{
	{-22.5, -19.9, -17.9, -15.6, -3.66, -2.51, -1.53, -0.43},
	{-25.7, -22.4, -19.8, -16.8, -3.71, -2.60, -1.66, -0.65},
	{-27.4, -23.6, -20.7, -17.5, -3.74, -2.62, -1.73, -0.75},
	{-28.4, -24.4, -21.3, -18.0, -3.75, -2.64, -1.78, -0.82},
	{-28.9, -24.8, -21.5, -18.1, -3.76, -2.65, -1.78, -0.84},
	{-29.5, -25.1, -21.8, -18.3, -3.77, -2.66, -1.79, -0.87},
}

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

// PPTestOpts groups the Phillips-Perron test parameters.
type PPTestOpts struct {
	Alpha  float64
	LShort bool
}

// PPTest implements the Phillips-Perron unit-root test (Z-alpha variant).
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
	X := make([][]float64, n)
	for i := 0; i < n; i++ {
		X[i] = []float64{1, tt[i], yt1[i]}
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

	alpha := beta[2]
	stat := float64(n)*(alpha-1) - math.Pow(float64(n), 6)/(24.0*dx)*(ssqrtl-ssqru)

	tableiPL := make([]float64, len(ppTableP))
	for i := range ppTableP {
		col := make([]float64, len(ppTable))
		for r := range ppTable {
			col[r] = ppTable[r][i]
		}
		out, err := Approx(ppTableT, col, []float64{float64(n)}, Linear, RuleClip)
		if err != nil {
			return PPResult{}, err
		}
		tableiPL[i] = out[0]
	}
	out, err := Approx(tableiPL, ppTableP, []float64{stat}, Linear, RuleClip)
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
	if addIntercept {
		cols++
	}
	flat := make([]float64, n*cols)
	for i, row := range X {
		off := i * cols
		if addIntercept {
			flat[off] = 1
			off++
		}
		for j, v := range row {
			flat[i*cols+j+boolToInt(addIntercept)] = v
		}
	}
	A := mat.NewDense(n, cols, flat)
	b := mat.NewVecDense(n, append([]float64{}, y...))
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
