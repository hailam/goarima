package arima

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/mat"
)

// chCritVals is the table of CH (Canova-Hansen) critical values for m=2..12.
var chCritVals = []float64{
	0.4617146, 0.7479655, 1.0007818, 1.2375350, 1.4625240,
	1.6920200, 1.9043096, 2.1169602, 2.3268562, 2.5406922,
	2.7391007,
}

// CalcCHCritVal returns the CH critical value for seasonal period m.
//
// Mirrors CHTest._calc_ch_crit_val.
func CalcCHCritVal(m int) float64 {
	if m <= 12 && m >= 2 {
		return chCritVals[m-2]
	}
	switch m {
	case 24:
		return 5.098624
	case 52:
		return 10.341416
	case 365:
		return 65.44445
	}
	return 0.269 * math.Pow(float64(m), 0.928)
}

// chSeasDummy returns the n-by-(m-1) matrix of Fourier seasonal dummies
// used by the Canova-Hansen test.
//
// Column layout matches pmdarima:
//
//	col 2*(i-1)   = cos(2*pi*i*tt/m)
//	col (2*i)-1   = sin(2*pi*i*tt/m)
//
// for i in 1..m, then truncated to first m-1 columns.
func chSeasDummy(n, m int) [][]float64 {
	full := make([][]float64, n)
	cols := 2 * m
	for j := 0; j < n; j++ {
		full[j] = make([]float64, cols)
	}
	for i := 1; i <= m; i++ {
		for tt := 1; tt <= n; tt++ {
			arg := 2 * math.Pi * float64(i) * float64(tt) / float64(m)
			full[tt-1][2*i-1] = math.Sin(arg)
			full[tt-1][2*(i-1)] = math.Cos(arg)
		}
	}
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = append([]float64{}, full[i][:m-1]...)
	}
	return out
}

// CHTestStat computes the Canova-Hansen seasonal stability statistic.
//
// Mirrors CHTest._sd_test.
func CHTestStat(wts []float64, s int) (float64, error) {
	n := len(wts)
	if n <= s {
		return 0, errors.New("series too short for CH test")
	}
	frecLen := (s + 1) / 2 // length of seasonal frequencies vector (all ones)
	ltrunc := int(math.Round(float64(s) * math.Pow(float64(n)/100.0, 0.25)))
	R1 := chSeasDummy(n, s)
	cols := s - 1
	// Fit OLS with intercept: wts = b0 + R1 * b. Pmdarima's
	// `make_pipeline(StandardScaler(with_mean=False), LinearRegression())`
	// uses fit_intercept=True (the default). Standardizing R1 doesn't
	// change residuals, so we skip the scaler and just include a constant.
	beta, err := olsFit(R1, wts, true)
	if err != nil {
		return 0, err
	}
	resid := make([]float64, n)
	for i := 0; i < n; i++ {
		pred := beta[0] // intercept
		for c := 0; c < cols; c++ {
			pred += R1[i][c] * beta[c+1]
		}
		resid[i] = wts[i] - pred
	}
	// Fhataux = R1 * resid (column-wise)
	Fhataux := make([][]float64, n)
	for i := 0; i < n; i++ {
		row := make([]float64, cols)
		for c := 0; c < cols; c++ {
			row[c] = R1[i][c] * resid[i]
		}
		Fhataux[i] = row
	}

	// Build A from the frecob mask (mirrors C_canova_hansen_sd_test).
	A := buildCHA(s, frecLen)

	// Compute Omfhat = Bartlett kernel-smoothed long-run covariance of Fhataux.
	Omfhat := bartlettCov(Fhataux, ltrunc)

	// AtOmfhatA = A^T * Omfhat * A
	var Aden mat.Dense
	Aden.CloneFrom(matFrom2D(A))
	var OmDense mat.Dense
	OmDense.CloneFrom(matFrom2D(Omfhat))
	var AtOm mat.Dense
	AtOm.Mul(Aden.T(), &OmDense)
	var AtOmA mat.Dense
	AtOmA.Mul(&AtOm, &Aden)

	// SVD eigenvalues
	var svd mat.SVD
	if !svd.Factorize(&AtOmA, mat.SVDNone) {
		return 0, errors.New("CH SVD failed")
	}
	sv := svd.Values(nil)
	smin := math.Inf(1)
	for _, v := range sv {
		if v < smin {
			smin = v
		}
	}
	const eps = 2.220446049250313e-16
	if smin < eps {
		return 0, nil
	}

	// solved = inverse(AtOmfhatA)
	var solved mat.Dense
	if err := solved.Inverse(&AtOmA); err != nil {
		return 0, err
	}
	// Fhat = cumsum of Fhataux along axis 0
	Fhat := make([][]float64, n)
	cum := make([]float64, cols)
	for i := 0; i < n; i++ {
		row := make([]float64, cols)
		for c := 0; c < cols; c++ {
			cum[c] += Fhataux[i][c]
			row[c] = cum[c]
		}
		Fhat[i] = row
	}
	// (1/n^2) * trace( solved * A^T * Fhat^T * Fhat * A )
	Fmat := matFrom2D(Fhat)
	var FtF mat.Dense
	FtF.Mul(Fmat.T(), Fmat)
	var AtFtF mat.Dense
	AtFtF.Mul(Aden.T(), &FtF)
	var AtFtFA mat.Dense
	AtFtFA.Mul(&AtFtF, &Aden)
	var prod mat.Dense
	prod.Mul(&solved, &AtFtFA)
	tr := 0.0
	r, _ := prod.Dims()
	for i := 0; i < r; i++ {
		tr += prod.At(i, i)
	}
	return tr / (float64(n) * float64(n)), nil
}

// CHTest evaluates the CH seasonal differencing decision (returns 0 or 1).
func CHTest(x []float64, m int) (int, error) {
	if len(x) == 0 {
		return 0, nil
	}
	if m < 2 {
		return 0, errors.New("m must be > 1")
	}
	n := len(x)
	if n < 2*m+5 {
		return 0, nil
	}
	stat, err := CHTestStat(x, m)
	if err != nil {
		return 0, err
	}
	crit := CalcCHCritVal(m)
	if stat > crit {
		return 1, nil
	}
	return 0, nil
}

// buildCHA returns the (s-1) x a selection matrix used by the CH test.
//
// Mirrors the frecob/A construction inside C_canova_hansen_sd_test, when the
// frequency vector frec is all ones (length nFrec = (s+1)/2).
func buildCHA(s, nFrec int) [][]float64 {
	halfS := s/2 - 1
	frecob := make([]int, s-1)
	sq := make([]int, nFrec)
	for i := range sq {
		sq[i] = 2 * i
	}
	for i := 0; i < nFrec; i++ {
		// frec[i] == 1 by assumption.
		if i == halfS {
			if sq[i] >= 0 && sq[i] < len(frecob) {
				frecob[sq[i]] = 1
			}
		}
		if i < halfS {
			frecob[sq[i]] = 1
			frecob[sq[i]+1] = 1
		}
	}
	a := 0
	for _, v := range frecob {
		if v == 1 {
			a++
		}
	}
	A := make([][]float64, s-1)
	for i := range A {
		A[i] = make([]float64, a)
	}
	j := 0
	for i, v := range frecob {
		if v == 1 {
			A[i][j] = 1
			j++
		}
	}
	return A
}

// matFrom2D converts a row-major slice-of-slices to a *mat.Dense.
func matFrom2D(x [][]float64) *mat.Dense {
	n := len(x)
	if n == 0 {
		return mat.NewDense(0, 0, nil)
	}
	c := len(x[0])
	flat := make([]float64, n*c)
	for i, row := range x {
		copy(flat[i*c:(i+1)*c], row)
	}
	return mat.NewDense(n, c, flat)
}

// bartlettCov returns the Bartlett-kernel long-run covariance of the rows of
// F. Mirrors pmdarima's C_canova_hansen_sd_test:
//
//	Omfhat = (F^T F + Omnw + Omnw^T) / Ne
//
// where Omnw = sum_{k=1..L} w_k * Gamma_k and w_k = 1 - k/(L+1),
// Gamma_k[a,b] = sum_{t=k}^{Ne-1} F_t[a] F_{t-k}[b].
func bartlettCov(F [][]float64, L int) [][]float64 {
	N := len(F)
	if N == 0 {
		return nil
	}
	c := len(F[0])
	omf := make([][]float64, c)
	for i := range omf {
		omf[i] = make([]float64, c)
	}
	for i := 0; i < N; i++ {
		for a := 0; a < c; a++ {
			for b := 0; b < c; b++ {
				omf[a][b] += F[i][a] * F[i][b]
			}
		}
	}
	for k := 1; k <= L; k++ {
		w := 1.0 - float64(k)/(float64(L)+1.0)
		for i := k; i < N; i++ {
			for a := 0; a < c; a++ {
				for b := 0; b < c; b++ {
					v := F[i][a] * F[i-k][b]
					omf[a][b] += w * v
					omf[b][a] += w * v
				}
			}
		}
	}
	for a := 0; a < c; a++ {
		for b := 0; b < c; b++ {
			omf[a][b] /= float64(N)
		}
	}
	return omf
}

// OCSBLagMethod selects the lag selection method.
type OCSBLagMethod int

const (
	// OCSBFixed uses max_lag directly.
	OCSBFixed OCSBLagMethod = iota
	// OCSBAIC selects via AIC.
	OCSBAIC
	// OCSBBIC selects via BIC.
	OCSBBIC
	// OCSBAICc selects via corrected AIC.
	OCSBAICc
)

// CalcOCSBCritVal returns the OCSB critical value for seasonal period m.
//
// Mirrors OCSBTest._calc_ocsb_crit_val.
func CalcOCSBCritVal(m int) float64 {
	logM := math.Log(float64(m))
	return -0.2937411*math.Exp(-0.2850853*(logM-0.7656451)+(-0.05983644)*math.Pow(logM-0.7656451, 2)) - 1.652202
}

// OCSBTest evaluates the seasonal differencing decision via the OCSB test.
//
// Returns 0 or 1 (per pmdarima convention). lagMethod and maxLag follow
// pmdarima.OCSBTest semantics.
func OCSBTest(x []float64, m int, lagMethod OCSBLagMethod, maxLag int) (int, error) {
	if len(x) == 0 {
		return 0, nil
	}
	if m < 2 {
		return 0, errors.New("m must be > 1")
	}
	stat, err := ocsbStat(x, m, lagMethod, maxLag)
	if err != nil {
		return 0, err
	}
	crit := CalcOCSBCritVal(m)
	if stat > crit {
		return 1, nil
	}
	return 0, nil
}

// ocsbStat computes the OCSB Z5 t-statistic with the chosen lag method.
func ocsbStat(x []float64, m int, lagMethod OCSBLagMethod, maxLag int) (float64, error) {
	if maxLag <= 0 {
		return 0, errors.New("maxLag must be positive")
	}
	bestLag := maxLag
	if lagMethod != OCSBFixed {
		bestIC := math.Inf(1)
		bestIdx := -1
		for lag := 1; lag <= maxLag; lag++ {
			ic, err := ocsbFitIC(x, m, lag, maxLag, lagMethod)
			if err != nil {
				continue
			}
			if !math.IsNaN(ic) && ic < bestIC {
				bestIC = ic
				bestIdx = lag
			}
		}
		if bestIdx == -1 {
			return 0, errors.New("all lag values produced singular matrices")
		}
		bestLag = bestIdx
	}
	stat, _, err := ocsbFit(x, m, bestLag, maxLag)
	return stat, err
}

// ocsbFitIC fits OCSB at a particular lag and returns the chosen IC.
func ocsbFitIC(x []float64, m, lag, maxLag int, method OCSBLagMethod) (float64, error) {
	_, ic, err := ocsbFitFull(x, m, lag, maxLag, method)
	return ic, err
}

// ocsbFit returns the Z5 t-statistic and the residual variance.
func ocsbFit(x []float64, m, lag, maxLag int) (float64, float64, error) {
	stat, _, err := ocsbFitFull(x, m, lag, maxLag, OCSBFixed)
	return stat, 0, err
}


// ocsbFitFull mirrors OCSBTest._fit_ocsb and returns Z5's t-statistic plus the
// requested information criterion.
func ocsbFitFull(x []float64, m, lag, maxLag int, method OCSBLagMethod) (float64, float64, error) {
	yfod := applyDiff(x, m, 1)
	if len(yfod) == 0 {
		return 0, math.NaN(), errors.New("not enough samples after seasonal differencing")
	}
	y := applyDiff(yfod, 1, 1)
	ylag := genLags(y, lag)
	if maxLag > -1 {
		// y = y[max_lag:]
		if maxLag >= len(y) {
			return 0, math.NaN(), errors.New("max_lag exceeds y length")
		}
		y = y[maxLag:]
	}
	mf := ylag
	if len(mf) > len(y) {
		mf = mf[:len(y)]
	}
	// AR fit: y ~ const + mf
	X := addConstantCol(mf)
	beta, err := olsFit(X, y, false)
	if err != nil {
		return 0, math.NaN(), err
	}

	// Z4
	z4y := yfod[lag:]
	z4lag := genLags(yfod, lag)
	if len(z4lag) > len(z4y) {
		z4lag = z4lag[:len(z4y)]
	}
	z4preds := predictLinear(addConstantCol(z4lag), beta)
	z4 := make([]float64, len(z4y))
	for i := range z4 {
		z4[i] = z4y[i] - z4preds[i]
	}

	// Z5
	z5y := applyDiff(x, 1, 1)
	z5lag := genLags(z5y, lag)
	z5y = z5y[lag:]
	if len(z5lag) > len(z5y) {
		z5lag = z5lag[:len(z5y)]
	}
	z5preds := predictLinear(addConstantCol(z5lag), beta)
	z5 := make([]float64, len(z5y))
	for i := range z5 {
		z5[i] = z5y[i] - z5preds[i]
	}

	// Final regression: y ~ mf + z4 + z5
	mfRows := len(mf)
	if len(z4) < mfRows {
		mfRows = len(z4)
	}
	if len(z5) < mfRows {
		mfRows = len(z5)
	}
	if mfRows < 3 {
		return 0, math.NaN(), errors.New("too few rows for OCSB final regression")
	}
	Xf := make([][]float64, mfRows)
	for i := 0; i < mfRows; i++ {
		row := make([]float64, len(mf[i])+2)
		copy(row, mf[i])
		row[len(mf[i])] = z4[i]
		row[len(mf[i])+1] = z5[i]
		Xf[i] = row
	}
	yf := y[:mfRows]
	betaF, err := olsFit(Xf, yf, false)
	if err != nil {
		return 0, math.NaN(), err
	}
	residF := make([]float64, mfRows)
	predF := predictLinear(Xf, betaF)
	for i := range residF {
		residF[i] = yf[i] - predF[i]
	}
	cols := len(Xf[0])
	stdErr, err := olsStdErr(Xf, residF, cols-1)
	if err != nil {
		return 0, math.NaN(), err
	}
	tZ5 := betaF[cols-1] / stdErr

	// Compute IC for the AR fit (intercept + mf).
	residAr := make([]float64, len(y))
	predAr := predictLinear(X, beta)
	for i := range residAr {
		residAr[i] = y[i] - predAr[i]
	}
	sse := 0.0
	for _, r := range residAr {
		sse += r * r
	}
	n := float64(len(y))
	if n == 0 {
		return tZ5, math.NaN(), nil
	}
	k := float64(len(beta))
	sigma2 := sse / n
	if sigma2 <= 0 {
		return tZ5, math.NaN(), nil
	}
	logL := -0.5 * n * (math.Log(2*math.Pi*sigma2) + 1)
	var ic float64
	switch method {
	case OCSBAIC:
		ic = 2*k - 2*logL
	case OCSBBIC:
		ic = math.Log(n)*k - 2*logL
	case OCSBAICc:
		if n-k-1 <= 0 {
			ic = math.Inf(1)
		} else {
			ic = 2*k - 2*logL + 2*k*(k+1)/(n-k-1)
		}
	default:
		ic = 0
	}
	return tZ5, ic, nil
}

// genLags returns the omit_na=true variant from OCSBTest._gen_lags / _do_lag.
// For lag=1 returns x as a single column; for lag>1 returns the dropped-NA
// matrix of overlapping windows.
func genLags(y []float64, lag int) [][]float64 {
	if lag <= 0 {
		out := make([][]float64, len(y))
		for i := range out {
			out[i] = []float64{0}
		}
		return out
	}
	if lag == 1 {
		out := make([][]float64, len(y))
		for i, v := range y {
			out[i] = []float64{v}
		}
		return out
	}
	n := len(y)
	rows := n - lag + 1
	if rows <= 0 {
		return [][]float64{}
	}
	out := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		row := make([]float64, lag)
		for j := 0; j < lag; j++ {
			row[j] = y[i+lag-1-j]
		}
		out[i] = row
	}
	return out
}

func addConstantCol(X [][]float64) [][]float64 {
	out := make([][]float64, len(X))
	for i, row := range X {
		nr := make([]float64, len(row)+1)
		nr[0] = 1
		copy(nr[1:], row)
		out[i] = nr
	}
	return out
}

func predictLinear(X [][]float64, beta []float64) []float64 {
	out := make([]float64, len(X))
	for i, row := range X {
		s := 0.0
		for j, b := range beta {
			if j < len(row) {
				s += row[j] * b
			}
		}
		out[i] = s
	}
	return out
}

// NDiffsTest selects the unit-root test for ndiffs.
type NDiffsTest int

const (
	// NDiffsKPSS uses the KPSS test (default).
	NDiffsKPSS NDiffsTest = iota
	// NDiffsADF uses the Augmented Dickey-Fuller test.
	NDiffsADF
	// NDiffsPP uses the Phillips-Perron test.
	NDiffsPP
)

// NDiffsOpts groups the configuration for NDiffs.
type NDiffsOpts struct {
	Alpha  float64
	Test   NDiffsTest
	MaxD   int
	Null   string // KPSS only: "level" or "trend"
	LShort bool   // KPSS / PP
}

// NDiffs estimates the non-seasonal differencing term d.
//
// Mirrors pmdarima.arima.utils.ndiffs.
func NDiffs(x []float64, opts NDiffsOpts) (int, error) {
	if opts.MaxD <= 0 {
		return 0, errors.New("max_d must be a positive integer")
	}
	if opts.Alpha == 0 {
		opts.Alpha = 0.05
	}
	xs := append([]float64{}, x...)
	if IsConstantSlice(xs) {
		return 0, nil
	}
	pval, doDiff, err := runDiffTest(xs, opts)
	if err != nil {
		return 0, err
	}
	if math.IsNaN(pval) {
		return 0, nil
	}
	d := 0
	for doDiff && d < opts.MaxD {
		d++
		nxt, derr := diffOnce(xs)
		if derr != nil {
			return d - 1, nil
		}
		xs = nxt
		if IsConstantSlice(xs) {
			return d, nil
		}
		pval, doDiff, err = runDiffTest(xs, opts)
		if err != nil {
			return d, err
		}
		if math.IsNaN(pval) {
			return d - 1, nil
		}
	}
	return d, nil
}

func runDiffTest(x []float64, opts NDiffsOpts) (pval float64, doDiff bool, err error) {
	switch opts.Test {
	case NDiffsKPSS:
		null := opts.Null
		if null == "" {
			null = "level"
		}
		res, e := KPSSTest(x, KPSSTestOpts{Alpha: opts.Alpha, Null: null, LShort: opts.LShort || true})
		return res.PValue, res.ShouldDiff, e
	case NDiffsADF:
		res, e := ADFTest(x, ADFTestOpts{Alpha: opts.Alpha})
		return res.PValue, res.ShouldDiff, e
	case NDiffsPP:
		res, e := PPTest(x, PPTestOpts{Alpha: opts.Alpha, LShort: opts.LShort || true})
		return res.PValue, res.ShouldDiff, e
	}
	return math.NaN(), false, errors.New("unknown test")
}

// IsConstantSlice returns true if all elements of x are equal.
func IsConstantSlice(x []float64) bool {
	if len(x) <= 1 {
		return true
	}
	for _, v := range x[1:] {
		if v != x[0] {
			return false
		}
	}
	return true
}

// NSDiffsTest selects the seasonal differencing test for NSDiffs.
type NSDiffsTest int

const (
	// NSDiffsOCSB uses the OCSB test (default).
	NSDiffsOCSB NSDiffsTest = iota
	// NSDiffsCH uses the Canova-Hansen test.
	NSDiffsCH
)

// NSDiffsOpts groups the configuration for NSDiffs.
type NSDiffsOpts struct {
	M         int
	MaxD      int
	Test      NSDiffsTest
	LagMethod OCSBLagMethod
	MaxLag    int
}

// NSDiffs estimates the seasonal differencing term D.
//
// Mirrors pmdarima.arima.utils.nsdiffs.
func NSDiffs(x []float64, opts NSDiffsOpts) (int, error) {
	if opts.MaxD <= 0 {
		return 0, errors.New("max_D must be a positive integer")
	}
	if opts.M <= 1 {
		return 0, errors.New("m must be > 1")
	}
	if opts.MaxLag <= 0 {
		opts.MaxLag = 3
	}
	xs := append([]float64{}, x...)
	if IsConstantSlice(xs) {
		return 0, nil
	}
	stepFn := func(s []float64) (int, error) {
		switch opts.Test {
		case NSDiffsCH:
			return CHTest(s, opts.M)
		default:
			return OCSBTest(s, opts.M, opts.LagMethod, opts.MaxLag)
		}
	}
	D := 0
	doDiff, err := stepFn(xs)
	if err != nil {
		return 0, err
	}
	for doDiff == 1 && D < opts.MaxD {
		D++
		nxt := applyDiff(xs, opts.M, 1)
		if len(nxt) == 0 {
			return D, nil
		}
		xs = nxt
		if IsConstantSlice(xs) {
			return D, nil
		}
		if len(xs) < opts.M {
			return D, nil
		}
		doDiff, err = stepFn(xs)
		if err != nil {
			return D, nil
		}
	}
	return D, nil
}
