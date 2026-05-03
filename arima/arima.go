package arima

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"

	"gonum.org/v1/gonum/optimize"
)

// InfoCriterion enumerates supported information criteria.
type InfoCriterion int

const (
	// AIC = 2k - 2log L
	AIC InfoCriterion = iota
	// AICc = AIC + 2k(k+1)/(n-k-1)
	AICc
	// BIC = log(n) k - 2 log L
	BIC
	// HQIC = 2 log(log n) k - 2 log L
	HQIC
)

// DiffuseConv tags which reference implementation the user wants to match.
// Today both modes use the same likelihood form — `n_effective = n_std`,
// observations during the diffuse phase are excluded — because both R's
// stats::arima and statsmodels SARIMAX agree on that. The flag is kept
// for forward-compat; differences between R and statsmodels for AR+seasonal-
// differencing models stem from internal state-space implementation details
// that do not change the likelihood formula.
type DiffuseConv int

const (
	// DiffuseR matches R's stats::arima.
	DiffuseR DiffuseConv = iota
	// DiffuseStatsmodels matches statsmodels SARIMAX(simple_differencing=False).
	DiffuseStatsmodels
)

// Method selects the fitting estimator.
type Method int

const (
	// MethodCSS uses Conditional Sum of Squares (fast).
	MethodCSS Method = iota
	// MethodML uses exact Gaussian likelihood via Kalman filter (state-space).
	MethodML
	// MethodCSSML starts with CSS and refines with ML.
	MethodCSSML
)

// ARIMA fits an ARIMA(p,d,q)(P,D,Q,m) model with optional exogenous regressors.
//
// API mirrors pmdarima.arima.ARIMA. Exogenous regressors X enter the model as
// linear predictors: y_t = X_t @ beta + u_t, where u_t follows the ARIMA
// process. Beta is estimated jointly with the ARMA parameters by maximising
// the Gaussian log-likelihood.
type ARIMA struct {
	// Configuration
	Order         Order
	Seasonal      SeasonalOrder
	WithIntercept bool   // include constant term in differenced series
	Method        Method // fitting method
	MaxIter       int    // optimizer max iterations (default 100)

	// Lambda enables a Box-Cox transform of y before fitting and inverts on
	// Predict. nil = no transform; *Lambda = power (use 0 for log). Useful
	// for stabilising variance on positive series. Mirrors pmdarima's
	// `lambda` argument on ARIMA / auto_arima.
	Lambda  *float64
	Lambda2 float64 // additive shift before transform; ignored if Lambda is nil

	// NonSimpleDifferencing fits via the integrated state-space form (R's
	// `stats::arima` and statsmodels SARIMAX). Default false → simple
	// differencing (pre-difference y, fit ARMA), which matches statsmodels
	// SARIMAX(simple_differencing=True) exactly.
	//
	// When true, we use the exact diffuse Kalman filter of Durbin-Koopman
	// (2003) with R's Gardner-Harvey-Phillips stationary covariance (getQ0).
	// The likelihood convention is selectable via DiffuseConvention.
	NonSimpleDifferencing bool

	// DiffuseConvention selects the likelihood treatment of the diffuse
	// (initial integrated-state) phase. Only matters when NonSimpleDifferencing
	// is true.
	DiffuseConvention DiffuseConv

	// DriftIncluded marks the model as having a leading exog column
	// representing a linear time index (drift). Set automatically by RArima
	// when IncludeDrift=true. When set, Predict/PredictBoot/Simulate
	// transparently prepend the drift column to user-supplied futureExog so
	// callers don't have to reconstruct the [n+1, n+2, …] sequence manually.
	// Mirrors R's forecast::forecast(model, h) behavior.
	DriftIncluded bool

	// GradientWorkers caps the goroutine count used by the BFGS numerical
	// gradient inside Fit. 0 → GOMAXPROCS (default). AutoArima sets this to
	// `max(1, GOMAXPROCS/n_parallel_fits)` when dispatching parallel
	// candidate fits, so the nested parallelism (outer fits × inner gradient
	// workers) doesn't exceed available cores.
	GradientWorkers int

	// Fitted state
	phi   []float64 // non-seasonal AR
	theta []float64 // non-seasonal MA
	Phi   []float64 // seasonal AR
	Theta []float64 // seasonal MA
	c     float64   // intercept (on differenced series)
	mean  float64   // mean of differenced series (when no intercept)
	beta  []float64 // exog coefficients (length = #exog cols)

	sigma2  float64
	logL    float64
	nobs    int
	resids  []float64
	yTrain  []float64   // original y, for forecasts
	xTrain  [][]float64 // original X, for forecasts (nil if not used)
	nExog   int
	fitted  bool
	psiInf  []float64 // truncated MA(infinity) for forecast variance
	psiInfN int

	// Predict caches (set at end of Fit, reused on every Predict call).
	yMSCache       []float64 // yTrain on the model scale (Box-Cox-applied if used)
	wsCenteredCache []float64 // differenced + centered training series (= residOf(best))
}

// NewARIMA constructs an ARIMA with default configuration.
func NewARIMA(order Order) *ARIMA {
	return &ARIMA{
		Order:    order,
		Seasonal: SeasonalOrder{},
		Method:   MethodCSSML,
		MaxIter:  100,
	}
}

// Fit estimates the parameters from training data y. exog is optional: pass
// nil for no exogenous regressors, or a [n_obs][k] matrix for k regressors.
func (m *ARIMA) Fit(y []float64, exog [][]float64) error {
	if len(y) == 0 {
		return errors.New("y must be non-empty")
	}
	// Reject NaN/Inf in y. The optimizer would silently produce nonsense
	// otherwise (objective evaluations would NaN-poison the BFGS state).
	for i, v := range y {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("y[%d] = %v (NaN/Inf not supported)", i, v)
		}
	}
	// Reject negative orders.
	if m.Order.P < 0 || m.Order.D < 0 || m.Order.Q < 0 {
		return fmt.Errorf("order has negative field: %+v", m.Order)
	}
	if m.Seasonal.P < 0 || m.Seasonal.D < 0 || m.Seasonal.Q < 0 || m.Seasonal.M < 0 {
		return fmt.Errorf("seasonal has negative field: %+v", m.Seasonal)
	}
	if exog != nil {
		if len(exog) != len(y) {
			return fmt.Errorf("exog rows (%d) must match len(y) (%d)", len(exog), len(y))
		}
		// Validate each row: same width as the first row, no NaN/Inf.
		// Pre-fix, ragged exog or NaN-laden columns silently produced bad
		// fits instead of clear errors.
		k0 := len(exog[0])
		if k0 == 0 {
			return errors.New("exog has zero columns; pass nil for no regressors")
		}
		for i, row := range exog {
			if len(row) != k0 {
				return fmt.Errorf("exog row %d has %d cols, want %d (ragged exog)", i, len(row), k0)
			}
			for j, v := range row {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					return fmt.Errorf("exog[%d][%d] = %v (NaN/Inf not supported)", i, j, v)
				}
			}
		}
	}
	if m.MaxIter == 0 {
		m.MaxIter = 100
	}
	m.yTrain = append([]float64{}, y...)
	// Apply Box-Cox transform up front. yTrain stores the *original* y so
	// Predict can inverse-transform back to the original scale.
	yEff := y
	if m.Lambda != nil {
		t, err := boxCoxApply(y, *m.Lambda, m.Lambda2)
		if err != nil {
			return fmt.Errorf("Box-Cox: %w", err)
		}
		yEff = t
	}
	y = yEff
	m.yMSCache = append([]float64(nil), y...)
	if exog != nil {
		m.xTrain = cloneMat(exog)
		m.nExog = len(exog[0])
	} else {
		m.xTrain = nil
		m.nExog = 0
	}

	// Apply differencing to y (and X column-wise).
	ws := append([]float64{}, y...)
	var wX [][]float64
	if exog != nil {
		wX = cloneMat(exog)
	}
	if m.Order.D > 0 {
		ws = applyDiff(ws, 1, m.Order.D)
		if wX != nil {
			wX = applyMatDiff(wX, 1, m.Order.D)
		}
	}
	if m.Seasonal.Active() && m.Seasonal.D > 0 {
		hd := m.Seasonal.D * m.Seasonal.M
		if hd > len(ws)+m.Order.D {
			return fmt.Errorf("seasonal differencing %d exceeds data length", hd)
		}
		ws = applyDiff(ws, m.Seasonal.M, m.Seasonal.D)
		if wX != nil {
			wX = applyMatDiff(wX, m.Seasonal.M, m.Seasonal.D)
		}
	}
	if len(ws) < 2 {
		return errors.New("differenced series too short")
	}

	// Statsmodels behavior: no constant by default when d>0 or D>0.
	// Only center on the mean when the series is fully un-differenced and no
	// intercept term is requested. (When exog is supplied, the regression
	// soaks up the level; we still skip the explicit centering.)
	mean := 0.0
	totalDiff := m.Order.D
	if m.Seasonal.Active() {
		totalDiff += m.Seasonal.D
	}
	if !m.WithIntercept && totalDiff == 0 && exog == nil {
		for _, v := range ws {
			mean += v
		}
		mean /= float64(len(ws))
		for i := range ws {
			ws[i] -= mean
		}
	}
	m.mean = mean
	m.nobs = len(ws)

	p, q := m.Order.P, m.Order.Q
	P, Q := 0, 0
	if m.Seasonal.Active() {
		P, Q = m.Seasonal.P, m.Seasonal.Q
	}
	k := m.nExog
	nFree := p + q + P + Q
	if m.WithIntercept {
		nFree++
	}
	nFree += k
	// DiffuseStatsmodels: scale data so σ²≈1, then unit-σ² Kalman with
	// kappa=1e6 matches statsmodels' kappa=1e6 absolute. We do this once
	// up-front; the optimizer doesn't see σ² as a free parameter.
	useStatsmodels := m.NonSimpleDifferencing && m.DiffuseConvention == DiffuseStatsmodels
	dataScale := 1.0
	if useStatsmodels {
		// Estimate σ² from the differenced series.
		mean0 := 0.0
		for _, v := range ws {
			mean0 += v
		}
		mean0 /= float64(len(ws))
		ss := 0.0
		for _, v := range ws {
			d := v - mean0
			ss += d * d
		}
		s2est := ss / float64(len(ws))
		if s2est > 0 {
			dataScale = math.Sqrt(s2est)
		}
	}

	if nFree == 0 {
		// Pure white-noise / random walk after differencing
		sse := 0.0
		for _, v := range ws {
			sse += v * v
		}
		m.sigma2 = sse / float64(len(ws))
		m.resids = append([]float64{}, ws...)
		m.wsCenteredCache = append([]float64(nil), ws...)
		m.logL = -0.5 * float64(len(ws)) * (math.Log(2*math.Pi*m.sigma2) + 1)
		m.fitted = true
		m.computePsi()
		return nil
	}

	// Initial guess: Hannan-Rissanen for AR/MA, OLS for beta, zeros for the
	// rest. HR yields ARMA params close to the MLE so the optimizer is much
	// more likely to land at the same local maximum that R's `stats::arima`
	// finds (which itself uses HR-style initial values via the CSS warm-up).
	x0 := make([]float64, nFree)
	{
		// Build a "clean" residual series for HR: subtract intercept (zero
		// initial guess) and OLS-projected exog before running HR on the
		// differenced data.
		resid := make([]float64, len(ws))
		copy(resid, ws)
		var betaInit []float64
		if k > 0 {
			if b, err := olsFit(wX, ws, false); err == nil {
				betaInit = b
				for i := range resid {
					for j := 0; j < k; j++ {
						resid[i] -= betaInit[j] * wX[i][j]
					}
				}
			}
		}
		// Hannan-Rissanen on the non-seasonal AR/MA structure.
		phiHR, thetaHR := hannanRissanen(resid, p, q)
		// Convert to the unconstrained (transform-input) parameter space.
		off := 0
		if p > 0 && phiHR != nil {
			raw := invertARTransform(phiHR)
			copy(x0[off:off+p], raw)
		}
		off += p
		if q > 0 && thetaHR != nil {
			raw := invertMATransform(thetaHR)
			copy(x0[off:off+q], raw)
		}
		off += q
		// Seasonal AR/MA stay zero — pure-seasonal HR is rarely a big win and
		// adds complexity. The CSS warm-up step further refines them.
		off += P + Q
		if m.WithIntercept {
			off++
		}
		if k > 0 && betaInit != nil {
			copy(x0[off:off+k], betaInit)
		}
	}

	// Compute residual series given a parameter vector — applied identically
	// inside CSS and Kalman objectives.
	residOf := func(params []float64) []float64 {
		_, _, _, _, c, beta := unpackParamsX(params, p, q, P, Q, m.WithIntercept, k)
		out := make([]float64, len(ws))
		for i, v := range ws {
			r := v - c
			if k > 0 {
				for j := 0; j < k; j++ {
					r -= beta[j] * wX[i][j]
				}
			}
			out[i] = r
		}
		return out
	}

	// For non-simple differencing the objective uses the full integrated
	// state-space form on the un-differenced y (minus intercept and exog).
	// We need the un-differenced data inside this closure; build it now.
	yUndiff := append([]float64{}, m.yTrain...)
	if m.Lambda != nil {
		// Apply the same Box-Cox transform that yEff applied above.
		t, err2 := boxCoxApply(yUndiff, *m.Lambda, m.Lambda2)
		if err2 == nil {
			yUndiff = t
		}
	}
	xUndiff := m.xTrain

	residOfFull := func(params []float64) []float64 {
		_, _, _, _, c, beta := unpackParamsX(params, p, q, P, Q, m.WithIntercept, k)
		out := make([]float64, len(yUndiff))
		for i, v := range yUndiff {
			rr := v - c
			if k > 0 {
				for j := 0; j < k; j++ {
					rr -= beta[j] * xUndiff[i][j]
				}
			}
			out[i] = rr
		}
		return out
	}

	// Scaled y (for DiffuseStatsmodels mode): y_scaled = y / dataScale.
	yUndiffScaled := yUndiff
	if useStatsmodels && dataScale != 1 {
		yUndiffScaled = make([]float64, len(yUndiff))
		for i, v := range yUndiff {
			yUndiffScaled[i] = v / dataScale
		}
	}
	xUndiffScaled := xUndiff
	if useStatsmodels && dataScale != 1 && xUndiff != nil {
		xUndiffScaled = make([][]float64, len(xUndiff))
		for i, row := range xUndiff {
			scaled := make([]float64, len(row))
			for j, v := range row {
				scaled[j] = v / dataScale
			}
			xUndiffScaled[i] = scaled
		}
	}

	objective := func(params []float64) float64 {
		phi, theta, sPhi, sTheta, _, _ := unpackParamsX(params, p, q, P, Q, m.WithIntercept, k)
		if useStatsmodels {
			Dord := 0
			mPer := 0
			if m.Seasonal.Active() {
				Dord = m.Seasonal.D
				mPer = m.Seasonal.M
			}
			yEff := yUndiffScaled
			if m.WithIntercept || k > 0 {
				_, _, _, _, c, beta := unpackParamsX(params, p, q, P, Q, m.WithIntercept, k)
				yAdj := make([]float64, len(yUndiffScaled))
				for i, v := range yUndiffScaled {
					adj := v - c
					if k > 0 && xUndiffScaled != nil {
						for j := 0; j < k; j++ {
							adj -= beta[j] * xUndiffScaled[i][j]
						}
					}
					yAdj[i] = adj
				}
				yEff = yAdj
			}
			// Run unit-σ² Kalman on scaled data; kappa_scaled = 1e6 / σ²_est
			// ensures statsmodels' kappa=1e6 absolute is preserved.
			kappaUnit := 1e6 / (dataScale * dataScale)
			ll, _, _ := kalmanSARIMAX(yEff, m.Order.D, mPer, Dord,
				phi, theta, sPhi, sTheta, kappaUnit)
			if math.IsNaN(ll) || math.IsInf(ll, 0) {
				return math.Inf(1)
			}
			return ll
		}
		if m.NonSimpleDifferencing {
			Dord := 0
			mPer := 0
			if m.Seasonal.Active() {
				Dord = m.Seasonal.D
				mPer = m.Seasonal.M
			}
			ll, _, _ := kalmanARIMAFullConv(residOfFull(params), m.Order.D, mPer, Dord,
				phi, theta, sPhi, sTheta, 1e6, m.DiffuseConvention)
			if math.IsNaN(ll) || math.IsInf(ll, 0) {
				return math.Inf(1)
			}
			return ll
		}
		fullPhi := expandSARMA(phi, sPhi, m.Seasonal.M)
		fullTheta := expandSMA(theta, sTheta, m.Seasonal.M)
		r := residOf(params)
		switch m.Method {
		case MethodCSS:
			ll, _, _ := armaCSS(r, fullPhi, fullTheta)
			return ll
		default:
			ll, _, _ := kalmanARMALikelihood(r, fullPhi, fullTheta)
			if math.IsNaN(ll) || math.IsInf(ll, 0) {
				return math.Inf(1)
			}
			return ll
		}
	}

	if m.Method == MethodCSSML {
		cssObj := func(params []float64) float64 {
			phi, theta, sPhi, sTheta, _, _ := unpackParamsX(params, p, q, P, Q, m.WithIntercept, k)
			fullPhi := expandSARMA(phi, sPhi, m.Seasonal.M)
			fullTheta := expandSMA(theta, sTheta, m.Seasonal.M)
			ll, _, _ := armaCSS(residOf(params), fullPhi, fullTheta)
			return ll
		}
		x0 = minimize(cssObj, x0, m.MaxIter, m.gradientWorkers())
	}

	best := minimize(objective, x0, m.MaxIter, m.gradientWorkers())
	phi, theta, sPhi, sTheta, c, beta := unpackParamsX(best, p, q, P, Q, m.WithIntercept, k)
	m.phi = phi
	m.theta = theta
	m.Phi = sPhi
	m.Theta = sTheta
	m.c = c
	m.beta = beta

	fullPhi := expandSARMA(phi, sPhi, m.Seasonal.M)
	fullTheta := expandSMA(theta, sTheta, m.Seasonal.M)
	r := residOf(best)
	var negLL, sigma2 float64
	switch {
	case useStatsmodels:
		Dord := 0
		mPer := 0
		if m.Seasonal.Active() {
			Dord = m.Seasonal.D
			mPer = m.Seasonal.M
		}
		yEff := yUndiffScaled
		if m.WithIntercept || k > 0 {
			yAdj := make([]float64, len(yUndiffScaled))
			for i, v := range yUndiffScaled {
				adj := v - c
				if k > 0 && xUndiffScaled != nil {
					for j := 0; j < k; j++ {
						adj -= beta[j] * xUndiffScaled[i][j]
					}
				}
				yAdj[i] = adj
			}
			yEff = yAdj
		}
		// Concentrated unit-σ² Kalman on scaled data with statsmodels-equivalent kappa.
		kappaUnit := 1e6 / (dataScale * dataScale)
		ll, s2Scaled, _ := kalmanSARIMAX(yEff, m.Order.D, mPer, Dord,
			phi, theta, sPhi, sTheta, kappaUnit)
		// Un-scale: σ²_orig = σ²_scaled * dataScale².
		// logL_orig = logL_scaled - n_eff * log(dataScale).
		// negLL_orig = negLL_scaled + n_eff * log(dataScale).
		nEff := len(yEff)
		// Drop the burn observations from the n_eff count.
		// kStatesDiff isn't directly available here, recompute.
		burn := m.Order.D
		if m.Seasonal.Active() {
			burn += m.Seasonal.D * m.Seasonal.M
		}
		nEff -= burn
		if nEff < 1 {
			nEff = 1
		}
		negLL = ll + float64(nEff)*math.Log(dataScale)
		sigma2 = s2Scaled * dataScale * dataScale
		// Restore intercept on the original scale so c parameter is meaningful.
		// (y_scaled = c_scaled + … on scaled data; multiply by dataScale to
		// recover original-scale intercept.)
		m.c = c * dataScale
		// β does NOT need a rescale here: the optimizer saw both y and x
		// scaled by 1/dataScale, so the regression coefficients it found
		// already apply 1:1 on the original scale (y/s = β · x/s ⇒ y = β · x).
		// Pre-fix this multiplied β by dataScale, overstating coefficients.
		if k > 0 {
			for i := range m.beta {
				m.beta[i] = beta[i]
			}
		}
	case m.NonSimpleDifferencing:
		Dord := 0
		mPer := 0
		if m.Seasonal.Active() {
			Dord = m.Seasonal.D
			mPer = m.Seasonal.M
		}
		ll, s2, _ := kalmanARIMAFullConv(residOfFull(best), m.Order.D, mPer, Dord,
			phi, theta, sPhi, sTheta, 1e6, m.DiffuseConvention)
		negLL, sigma2 = ll, s2
	case m.Method == MethodCSS:
		// MethodCSS optimizes the conditional sum-of-squares profile
		// likelihood; report the same family of stats so logL/IC align with
		// the estimator that produced the parameters. Mirrors R's
		// stats::arima(method="CSS") convention. Note AIC under CSS is not
		// directly comparable to AIC under ML across estimators.
		ll, s2, _ := armaCSS(r, fullPhi, fullTheta)
		negLL, sigma2 = ll, s2
	default:
		ll, s2, _ := kalmanARMALikelihood(r, fullPhi, fullTheta)
		negLL, sigma2 = ll, s2
	}
	if math.IsNaN(negLL) || math.IsInf(negLL, 0) {
		_, sigma2, _ = armaCSS(r, fullPhi, fullTheta)
		negLL = float64(len(r)) / 2 * math.Log(sigma2)
	}
	m.sigma2 = sigma2
	m.logL = -negLL

	// Recompute residuals + cache using the FINAL (possibly post-rescale)
	// m.c and m.beta. In DiffuseStatsmodels mode the optimizer runs on
	// scaled data and m.c/m.beta are rescaled back at the end of the switch
	// above; the optimizer's `r` used the un-rescaled coefficients. Predict
	// expects coefficients consistent with wsCenteredCache, so we rebuild
	// here from m.c/m.beta directly. For all other modes the rescale is a
	// no-op and rFinal == r.
	//
	// Note: `ws` is already mean-subtracted earlier in Fit (line ~197) when
	// the centering branch fires, so we MUST NOT subtract m.mean again here
	// — that would double-center. residOf has the same shape: `ws - c - β·x`.
	rFinal := make([]float64, len(ws))
	for i, v := range ws {
		rr := v - m.c
		if m.nExog > 0 {
			for j, b := range m.beta {
				rr -= b * wX[i][j]
			}
		}
		rFinal[i] = rr
	}
	_, _, res := armaCSS(rFinal, fullPhi, fullTheta)
	m.resids = res
	m.wsCenteredCache = rFinal

	m.fitted = true
	m.computePsi()
	return nil
}

// parallelGradient returns a central-difference gradient function that
// evaluates the 2*n perturbed objective calls concurrently across goroutines.
//
// `f` must be safe for concurrent invocation with distinct argument vectors —
// our objectives allocate fresh state per call and do not mutate captured
// slices, which satisfies this. For tiny problems (<4 params) the goroutine
// overhead can dominate, so we fall back to a sequential loop in that case.
func parallelGradient(f func([]float64) float64, nWorkers int) func(grad, x []float64) {
	const eps = 1e-7
	if nWorkers < 1 {
		nWorkers = 1
	}
	return func(grad, x []float64) {
		n := len(x)
		if n < 4 {
			// sequential — overhead exceeds gain at small n
			for i := 0; i < n; i++ {
				save := x[i]
				x[i] = save + eps
				fp := f(x)
				x[i] = save - eps
				fm := f(x)
				x[i] = save
				grad[i] = (fp - fm) / (2 * eps)
			}
			return
		}
		// Cap workers at the number of jobs — extra workers would spin on an
		// empty channel and consume scheduler cycles for no benefit.
		nw := nWorkers
		if nw > n {
			nw = n
		}
		jobs := make(chan int, n)
		var wg sync.WaitGroup
		for w := 0; w < nw; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				xLocal := make([]float64, n)
				for i := range jobs {
					copy(xLocal, x)
					xLocal[i] += eps
					fp := f(xLocal)
					xLocal[i] -= 2 * eps
					fm := f(xLocal)
					xLocal[i] += eps // restore for next iter (cheap)
					grad[i] = (fp - fm) / (2 * eps)
				}
			}()
		}
		for i := 0; i < n; i++ {
			jobs <- i
		}
		close(jobs)
		wg.Wait()
	}
}

// minimize runs BFGS with a numerical gradient. Falls back to Nelder-Mead
// if BFGS fails or stalls. Returns the best parameter vector found.
//
// The numerical gradient is computed in parallel — each component requires
// two independent f(x) calls, dispatched to goroutines (a Go-specific win
// since each f() allocates fresh state and does not mutate shared memory).
// gradientWorkers returns the effective worker count for parallelGradient,
// respecting the optional GradientWorkers cap. AutoArima uses this to avoid
// oversubscription when dispatching parallel candidate fits.
func (m *ARIMA) gradientWorkers() int {
	if m.GradientWorkers > 0 {
		return m.GradientWorkers
	}
	return runtime.GOMAXPROCS(0)
}

func minimize(f func([]float64) float64, x0 []float64, maxIter, nWorkers int) []float64 {
	if nWorkers < 1 {
		nWorkers = runtime.GOMAXPROCS(0)
	}
	gradFn := parallelGradient(f, nWorkers)
	prob := optimize.Problem{Func: f, Grad: gradFn}
	settings := &optimize.Settings{
		MajorIterations:   maxIter,
		FuncEvaluations:   maxIter * 50,
		GradientThreshold: 1e-6,
	}
	bestX := make([]float64, len(x0))
	copy(bestX, x0)
	bestF := f(x0)

	// Nelder-Mead serves two purposes: (1) safety net when BFGS errors or
	// stalls, (2) global-optimum search — for SARIMA in particular, BFGS
	// often converges to a local minimum and NM finds a substantially
	// better point (TestRParityStatsmodelsWineind would lose ~17 logL units
	// without NM). So we cannot skip NM outright. But when BFGS converged
	// cleanly we can run NM with a reduced budget — most local-min escapes
	// happen in the first ~maxIter major iters anyway.
	bfgsConverged := false
	if res, err := optimize.Minimize(prob, x0, settings, &optimize.BFGS{}); err == nil && res != nil {
		if res.F < bestF {
			bestF = res.F
			copy(bestX, res.X)
		}
		switch res.Status {
		case optimize.GradientThreshold,
			optimize.FunctionConvergence,
			optimize.StepConvergence,
			optimize.FunctionThreshold:
			bfgsConverged = true
		}
	}

	nmIters := maxIter * 4
	nmFuncEvals := maxIter * 200
	if bfgsConverged {
		// Cheap polish — still escapes most local minima but at ~1/4 cost.
		nmIters = maxIter
		nmFuncEvals = maxIter * 50
	}
	probNM := optimize.Problem{Func: f}
	settingsNM := &optimize.Settings{
		MajorIterations: nmIters,
		FuncEvaluations: nmFuncEvals,
	}
	if res, err := optimize.Minimize(probNM, bestX, settingsNM, &optimize.NelderMead{}); err == nil && res != nil {
		if res.F < bestF {
			bestX = make([]float64, len(res.X))
			copy(bestX, res.X)
		}
	}
	return bestX
}

// unpackParamsX splits the flat parameter vector into AR/MA/seasonal/intercept/beta.
// Applies stationarity/invertibility transforms.
func unpackParamsX(params []float64, p, q, P, Q int, withIntercept bool, k int) (phi, theta, sPhi, sTheta []float64, c float64, beta []float64) {
	idx := 0
	if p > 0 {
		phi = arTransparams(params[idx : idx+p])
		idx += p
	}
	if q > 0 {
		theta = maTransparams(params[idx : idx+q])
		idx += q
	}
	if P > 0 {
		sPhi = arTransparams(params[idx : idx+P])
		idx += P
	}
	if Q > 0 {
		sTheta = maTransparams(params[idx : idx+Q])
		idx += Q
	}
	if withIntercept {
		c = params[idx]
		idx++
	}
	if k > 0 {
		beta = append([]float64{}, params[idx:idx+k]...)
	}
	return
}

// IC returns the chosen information criterion of the fitted model.
func (m *ARIMA) IC(ic InfoCriterion) float64 {
	if !m.fitted {
		return math.Inf(1)
	}
	k := float64(len(m.phi) + len(m.theta) + len(m.Phi) + len(m.Theta) + len(m.beta))
	if m.WithIntercept {
		k++
	}
	k++ // sigma^2
	n := float64(m.nobs)
	switch ic {
	case AIC:
		return 2*k - 2*m.logL
	case AICc:
		if n-k-1 <= 0 {
			return math.Inf(1)
		}
		return 2*k - 2*m.logL + 2*k*(k+1)/(n-k-1)
	case BIC:
		return math.Log(n)*k - 2*m.logL
	case HQIC:
		if n <= 1 {
			return math.Inf(1)
		}
		return 2*math.Log(math.Log(n))*k - 2*m.logL
	}
	return math.Inf(1)
}

// AIC returns AIC of the fit.
func (m *ARIMA) AIC() float64 { return m.IC(AIC) }

// BIC returns BIC of the fit.
func (m *ARIMA) BIC() float64 { return m.IC(BIC) }

// AICc returns AICc of the fit.
func (m *ARIMA) AICc() float64 { return m.IC(AICc) }

// LogLikelihood returns the maximised log-likelihood.
func (m *ARIMA) LogLikelihood() float64 { return m.logL }

// Sigma2 returns the residual variance estimate.
func (m *ARIMA) Sigma2() float64 { return m.sigma2 }

// Resid returns the in-sample residuals (length = differenced obs count).
func (m *ARIMA) Resid() []float64 {
	out := make([]float64, len(m.resids))
	copy(out, m.resids)
	return out
}

// Beta returns the exogenous-regressor coefficients (empty if no exog).
func (m *ARIMA) Beta() []float64 {
	out := make([]float64, len(m.beta))
	copy(out, m.beta)
	return out
}

// Params returns the fitted parameter vector in the order
// [phi..., theta..., Phi..., Theta..., intercept (if any), beta...].
func (m *ARIMA) Params() []float64 {
	out := make([]float64, 0, len(m.phi)+len(m.theta)+len(m.Phi)+len(m.Theta)+1+len(m.beta))
	out = append(out, m.phi...)
	out = append(out, m.theta...)
	out = append(out, m.Phi...)
	out = append(out, m.Theta...)
	if m.WithIntercept {
		out = append(out, m.c)
	}
	out = append(out, m.beta...)
	return out
}

// FittedValues returns in-sample one-step-ahead predictions y_hat[t].
//
// Computed as y_hat[t] = y[t] - residual[t] on the model scale, then
// inverse-Box-Cox (if Lambda was set) so the output matches the user's
// original units. Aligned to the original time index; length =
// len(yTrain) - (d + D*m). The first (d + D*m) values are undefined and
// not returned.
//
// Pre-fix, the subtraction used m.yTrain (original units) directly with
// m.resids (model-scale, post-Box-Cox), which produced meaningless values
// when Lambda was set. Now both operands are on the model scale.
func (m *ARIMA) FittedValues() []float64 {
	if !m.fitted {
		return nil
	}
	dHead := m.Order.D
	if m.Seasonal.Active() {
		dHead += m.Seasonal.D * m.Seasonal.M
	}
	n := len(m.yTrain) - dHead
	if n <= 0 || len(m.resids) == 0 {
		return nil
	}
	residTail := m.resids
	if len(residTail) > n {
		residTail = residTail[len(residTail)-n:]
	} else if len(residTail) < n {
		// pad front with zeros
		pad := make([]float64, n-len(residTail))
		residTail = append(pad, residTail...)
	}
	// Use the model-scale yTrain (Box-Cox-applied if Lambda is set) so the
	// subtraction is unit-consistent with the residuals.
	yMS := m.yMSCache
	if yMS == nil {
		yMS = m.yTrain // older snapshots without cache; Lambda must be nil
	}
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = yMS[dHead+i] - residTail[i]
	}
	if m.Lambda != nil {
		out = boxCoxInvert(out, *m.Lambda, m.Lambda2)
	}
	return out
}

// PredictInSample returns the in-sample one-step-ahead predictions, alias of
// FittedValues. Mirrors pmdarima.ARIMA.predict_in_sample.
func (m *ARIMA) PredictInSample() []float64 { return m.FittedValues() }

// Update appends new observations to the training data and re-fits the model
// (with the same orders/configuration). Mirrors pmdarima.ARIMA.update.
//
// newY are the new observations; newX (optional) the matching exogenous rows.
func (m *ARIMA) Update(newY []float64, newX [][]float64) error {
	if !m.fitted {
		return errors.New("model not fitted")
	}
	if newX != nil && len(newX) != len(newY) {
		return fmt.Errorf("newX rows (%d) != len(newY) (%d)", len(newX), len(newY))
	}
	if (m.nExog > 0) != (newX != nil) {
		return errors.New("exog provided/missing inconsistent with original fit")
	}
	combinedY := append([]float64{}, m.yTrain...)
	combinedY = append(combinedY, newY...)
	var combinedX [][]float64
	if newX != nil {
		combinedX = append(combinedX, m.xTrain...)
		combinedX = append(combinedX, cloneMat(newX)...)
	}
	return m.Fit(combinedY, combinedX)
}

// extendDriftIfNeeded prepends the drift column [n+1, n+2, …, n+nPeriods]
// to user-supplied futureExog when m.DriftIncluded is true. The user passes
// only the OTHER exog columns (or nil when drift is the only one) — drift
// is reconstructed automatically because it's a deterministic time index.
//
// Returns the (possibly augmented) futureExog. No-op when DriftIncluded
// is false.
func (m *ARIMA) extendDriftIfNeeded(futureExog [][]float64, nPeriods int) [][]float64 {
	if !m.DriftIncluded || nPeriods <= 0 {
		return futureExog
	}
	n0 := len(m.yTrain)
	if futureExog == nil {
		// Drift is the only exog column. Build it directly.
		out := make([][]float64, nPeriods)
		for i := 0; i < nPeriods; i++ {
			out[i] = []float64{float64(n0 + i + 1)}
		}
		return out
	}
	// User supplied the OTHER exog columns; prepend drift.
	out := make([][]float64, nPeriods)
	for i := 0; i < nPeriods; i++ {
		row := make([]float64, 1+len(futureExog[i]))
		row[0] = float64(n0 + i + 1)
		copy(row[1:], futureExog[i])
		out[i] = row
	}
	return out
}

// Predict produces nPeriods forward forecasts. If alpha > 0, lower/upper
// confidence intervals are returned alongside; otherwise lower/upper are nil.
//
// futureExog is required if the model was fitted with exogenous regressors;
// it must have nPeriods rows.
func (m *ARIMA) Predict(nPeriods int, alpha float64, futureExog [][]float64) (forecast, lower, upper []float64, err error) {
	if !m.fitted {
		return nil, nil, nil, errors.New("model not fitted")
	}
	if nPeriods <= 0 {
		return []float64{}, nil, nil, nil
	}
	// If the model includes a drift column, transparently prepend it to
	// futureExog. Done before validation so the user-facing futureExog
	// shape is "the OTHER exog cols only" (or nil when drift is the only one).
	futureExog = m.extendDriftIfNeeded(futureExog, nPeriods)
	if m.nExog > 0 {
		if futureExog == nil {
			return nil, nil, nil, errors.New("future exog required for forecasting")
		}
		if len(futureExog) != nPeriods {
			return nil, nil, nil, fmt.Errorf("future exog rows (%d) must match nPeriods (%d)", len(futureExog), nPeriods)
		}
		// Validate every row width — checking only [0] would let a ragged
		// matrix through and panic later during differencing.
		for i, row := range futureExog {
			if len(row) != m.nExog {
				return nil, nil, nil, fmt.Errorf("future exog row %d cols (%d) must match training (%d)",
					i, len(row), m.nExog)
			}
		}
	} else if futureExog != nil {
		return nil, nil, nil, errors.New("model was fitted without exog; do not pass futureExog")
	}

	fullPhi := expandSARMA(m.phi, m.Phi, m.Seasonal.M)
	fullTheta := expandSMA(m.theta, m.Theta, m.Seasonal.M)

	// Cached at end of Fit: yMS is yTrain on the model scale (Box-Cox-applied
	// if Lambda was set), wsCentered is the differenced+centered training
	// series, and m.resids holds the in-sample residuals.
	yMS := m.yMSCache
	wsCentered := m.wsCenteredCache
	res := m.resids

	// Difference the future exog. We only need the last `dHead` historical
	// rows of m.xTrain as context for the differencing operator (not the
	// full training matrix). For long-history exog this trims a per-call
	// O(len(xTrain) × nExog) matrix copy down to O(dHead × nExog).
	var futureWX [][]float64
	if futureExog != nil {
		dHead := m.Order.D
		if m.Seasonal.Active() {
			dHead += m.Seasonal.D * m.Seasonal.M
		}
		if dHead == 0 {
			// No differencing — futureExog rows pass through unchanged.
			futureWX = futureExog
		} else {
			trimStart := len(m.xTrain) - dHead
			if trimStart < 0 {
				trimStart = 0
			}
			combined := make([][]float64, 0, dHead+nPeriods)
			combined = append(combined, m.xTrain[trimStart:]...)
			combined = append(combined, futureExog...)
			diffed := combined
			if m.Order.D > 0 {
				diffed = applyMatDiff(diffed, 1, m.Order.D)
			}
			if m.Seasonal.Active() && m.Seasonal.D > 0 {
				diffed = applyMatDiff(diffed, m.Seasonal.M, m.Seasonal.D)
			}
			// After both diff steps `diffed` has exactly nPeriods rows
			// (combined was sized so the diff head consumes everything before
			// futureExog) — but slice from the tail to be defensive against
			// any boundary off-by-ones.
			futureWX = diffed[len(diffed)-nPeriods:]
		}
	}

	// Forecast residuals on differenced/centered scale.
	forecastDiffed := make([]float64, nPeriods)
	hist := append([]float64{}, wsCentered...)
	residHist := append([]float64{}, res...)
	for h := 0; h < nPeriods; h++ {
		yh := 0.0
		for i, ph := range fullPhi {
			idx := len(hist) - 1 - i
			if idx >= 0 {
				yh += ph * hist[idx]
			}
		}
		for j, th := range fullTheta {
			idx := len(residHist) - 1 - j
			if idx >= 0 && idx < len(res) {
				yh += th * residHist[idx]
			}
		}
		forecastDiffed[h] = yh
		hist = append(hist, yh)
		residHist = append(residHist, 0)
	}
	// Re-add intercept, mean, and exog contribution on the differenced scale.
	for i := range forecastDiffed {
		forecastDiffed[i] += m.mean + m.c
		if futureWX != nil {
			for j, b := range m.beta {
				forecastDiffed[i] += b * futureWX[i][j]
			}
		}
	}

	// Integrate seasonal differencing back. Use model-scale y (Box-Cox-
	// transformed if applicable) since forecasts are in model units.
	out := forecastDiffed
	if m.Seasonal.Active() && m.Seasonal.D > 0 {
		head := lastN(diffStream(yMS, 1, m.Order.D), m.Seasonal.D*m.Seasonal.M)
		full := integrateBack(out, head, m.Seasonal.M, m.Seasonal.D)
		out = full[len(head):]
	}
	if m.Order.D > 0 {
		head := lastN(yMS, m.Order.D)
		full := integrateBack(out, head, 1, m.Order.D)
		out = full[len(head):]
	}

	// Confidence intervals using truncated MA(infinity) coefficients.
	if alpha > 0 && alpha < 1 && len(m.psiInf) > 0 {
		z := normPPF(1 - alpha/2)
		lower = make([]float64, nPeriods)
		upper = make([]float64, nPeriods)
		var2 := 0.0
		for h := 0; h < nPeriods; h++ {
			if h < len(m.psiInf) {
				var2 += m.psiInf[h] * m.psiInf[h]
			}
			se := math.Sqrt(m.sigma2 * var2)
			lower[h] = out[h] - z*se
			upper[h] = out[h] + z*se
		}
	}

	// Invert Box-Cox if applied during Fit.
	if m.Lambda != nil {
		out = boxCoxInvert(out, *m.Lambda, m.Lambda2)
		if lower != nil {
			lower = boxCoxInvert(lower, *m.Lambda, m.Lambda2)
			upper = boxCoxInvert(upper, *m.Lambda, m.Lambda2)
		}
	}
	return out, lower, upper, nil
}

// FitPredict combines Fit and Predict in one step (for convenience).
func (m *ARIMA) FitPredict(y []float64, exog [][]float64, nPeriods int, futureExog [][]float64) ([]float64, error) {
	if err := m.Fit(y, exog); err != nil {
		return nil, err
	}
	fc, _, _, err := m.Predict(nPeriods, 0, futureExog)
	return fc, err
}

// diffStream applies non-seasonal differencing only.
func diffStream(y []float64, lag, times int) []float64 {
	if times == 0 {
		out := make([]float64, len(y))
		copy(out, y)
		return out
	}
	return applyDiff(y, lag, times)
}

func lastN(x []float64, n int) []float64 {
	if n >= len(x) {
		out := make([]float64, len(x))
		copy(out, x)
		return out
	}
	out := make([]float64, n)
	copy(out, x[len(x)-n:])
	return out
}

// applyMatDiff differences each column of x by lag, repeated times.
func applyMatDiff(x [][]float64, lag, times int) [][]float64 {
	if times == 0 || len(x) == 0 {
		return cloneMat(x)
	}
	rows := len(x)
	cols := len(x[0])
	out := cloneMat(x)
	for t := 0; t < times; t++ {
		if rows <= lag {
			return [][]float64{}
		}
		next := make([][]float64, rows-lag)
		for i := 0; i < rows-lag; i++ {
			row := make([]float64, cols)
			for j := 0; j < cols; j++ {
				row[j] = out[i+lag][j] - out[i][j]
			}
			next[i] = row
		}
		out = next
		rows = len(out)
	}
	return out
}

// cloneMat returns a deep copy of a row-major matrix.
func cloneMat(x [][]float64) [][]float64 {
	if x == nil {
		return nil
	}
	out := make([][]float64, len(x))
	for i, row := range x {
		r := make([]float64, len(row))
		copy(r, row)
		out[i] = r
	}
	return out
}

// computePsi precomputes truncated MA(infinity) coefficients for forecast variance.
//
// For ARIMA(p,d,q)(P,D,Q,m) the integrated process is
//
//	y_t = [theta(B) Theta(B^m)] / [phi(B) Phi(B^m) (1-B)^d (1-B^m)^D] · e_t
//
// We first compute psi for the ARMA part via the standard recursion, then
// integrate to account for differencing:
//
//   - 1/(1-B)   = 1 + B + B² + …  ⇒ apply cumulative sum, once per d.
//   - 1/(1-B^m) = 1 + B^m + B^{2m} + … ⇒ apply stride-m cumulative sum, once per D.
//
// Without this integration step, forecast variance is constant in h for any
// purely-integrated model — e.g. ARIMA(0,1,0) gets flat alpha-CIs instead of
// the SD growing like √h. Matches R's `forecast::forecast` and
// statsmodels' `SARIMAXResults.get_forecast` variance growth.
func (m *ARIMA) computePsi() {
	const maxLag = 100
	fullPhi := expandSARMA(m.phi, m.Phi, m.Seasonal.M)
	fullTheta := expandSMA(m.theta, m.Theta, m.Seasonal.M)
	psi := make([]float64, maxLag)
	psi[0] = 1.0
	for h := 1; h < maxLag; h++ {
		s := 0.0
		for i, ph := range fullPhi {
			if h-1-i >= 0 {
				s += ph * psi[h-1-i]
			}
		}
		if h-1 < len(fullTheta) {
			s += fullTheta[h-1]
		}
		psi[h] = s
	}
	// Integrate non-seasonal differencing (apply 1/(1-B), d times).
	for k := 0; k < m.Order.D; k++ {
		for h := 1; h < maxLag; h++ {
			psi[h] += psi[h-1]
		}
	}
	// Integrate seasonal differencing (apply 1/(1-B^m), D times).
	if m.Seasonal.Active() && m.Seasonal.D > 0 && m.Seasonal.M > 0 {
		for k := 0; k < m.Seasonal.D; k++ {
			for h := m.Seasonal.M; h < maxLag; h++ {
				psi[h] += psi[h-m.Seasonal.M]
			}
		}
	}
	m.psiInf = psi
	m.psiInfN = maxLag
}

// normPPF returns the inverse CDF of the standard normal.
func normPPF(p float64) float64 {
	if p <= 0 || p >= 1 {
		return math.NaN()
	}
	a := []float64{
		-3.969683028665376e+01, 2.209460984245205e+02,
		-2.759285104469687e+02, 1.383577518672690e+02,
		-3.066479806614716e+01, 2.506628277459239e+00,
	}
	b := []float64{
		-5.447609879822406e+01, 1.615858368580409e+02,
		-1.556989798598866e+02, 6.680131188771972e+01,
		-1.328068155288572e+01,
	}
	c := []float64{
		-7.784894002430293e-03, -3.223964580411365e-01,
		-2.400758277161838e+00, -2.549732539343734e+00,
		4.374664141464968e+00, 2.938163982698783e+00,
	}
	d := []float64{
		7.784695709041462e-03, 3.224671290700398e-01,
		2.445134137142996e+00, 3.754408661907416e+00,
	}
	const pLow = 0.02425
	const pHigh = 1 - pLow
	var x float64
	switch {
	case p < pLow:
		q := math.Sqrt(-2 * math.Log(p))
		x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q + c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
	case p <= pHigh:
		q := p - 0.5
		r := q * q
		x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r + a[5]) * q /
			(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r + 1)
	default:
		q := math.Sqrt(-2 * math.Log(1-p))
		x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q + c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
	}
	return x
}
