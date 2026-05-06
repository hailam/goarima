package arima

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/ajroetker/go-highway/hwy/contrib/vec"
	"gonum.org/v1/gonum/optimize"
)

// psiCache is an immutable snapshot of the truncated MA(∞) coefficients used
// for forecast-variance computation. Published atomically via
// `*ARIMA`.psi; readers load a pointer and treat the slice as read-only.
// Cache extension allocates a NEW psiCache and CAS-installs it, so any
// concurrent reader holding an older snapshot continues to see consistent
// data through the end of its read.
type psiCache struct {
	values []float64
}

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
	// MethodCSSML starts with CSS and refines with ML. The zero value of
	// Method, so structs left at their zero value (e.g. AutoArimaOpts{})
	// pick this — matching pmdarima / R's default behaviour.
	//
	// Pre-fix the iota order placed MethodCSS at 0, which silently
	// downgraded AutoArima callers who didn't explicitly set Method. This
	// reorder fixes GAP-1 from the gap audit. Method is JSON-serialized
	// as a string via methodToString (`serialize.go`) so the reorder
	// doesn't invalidate previously-saved models.
	MethodCSSML Method = iota
	// MethodCSS uses Conditional Sum of Squares (fast, biased).
	MethodCSS
	// MethodML uses exact Gaussian likelihood via Kalman filter (state-space).
	MethodML
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

	// RidgePenalty is an L2 penalty on the optimizer's UNCONSTRAINED x
	// vector — the pre-tanh AR/MA representation. Default 0.0 (off,
	// R-parity safe).
	//
	// The penalty `λ · Σx²` is added to the negative log-likelihood
	// during optimization. It prevents BFGS from pushing the
	// unconstrained x to extreme values where `tanh(x) ≈ ±1`, i.e.
	// where AR/MA coefficients hit the stationarity / invertibility
	// boundary. Such boundary fits are the root cause of KAL-1: the
	// exact-likelihood Kalman recursion blows up with `γ(0) → ∞`
	// initial covariance.
	//
	// Recommended for short series (n ≤ ~50) where the likelihood
	// surface is flat and BFGS is prone to drifting onto degenerate
	// boundary solutions. A useful starting value is `λ = 1.0/n` —
	// scales the regulariser to the data signal. Larger λ biases
	// estimates more aggressively toward zero (shrinkage).
	//
	// Default 0.0 → KAL-1's textbook fallback handles boundary
	// degeneracy after the fact. RidgePenalty fixes it before the
	// fact.
	RidgePenalty float64

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

	// Warm-start hooks (used by Update). Private; not exposed via JSON
	// serialization. When warmStartX has length == nFree, Fit uses it as
	// the optimizer's starting point and skips both Hannan-Rissanen and the
	// MethodCSSML CSS-warmup phase. Cleared at the end of every Fit call.
	warmStartX       []float64
	warmStartMaxIter int

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

	// psi holds the truncated MA(infinity) coefficient cache used for
	// forecast-variance computation in Predict. Stored via atomic.Pointer
	// so that concurrent Predict calls on the same fitted model are safe:
	// the fast path is a single atomic load + slice reads, and the cache
	// only ever grows (CAS publishes a new larger snapshot on extension).
	// Pre-fix this was a plain []float64 + int pair mutated in place from
	// inside Predict, which raced under concurrent forecasts with mixed
	// horizons. Cache snapshots are immutable once published.
	psi atomic.Pointer[psiCache]

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

	// Always clear warm-start hooks at function exit so a subsequent Fit
	// call doesn't accidentally inherit them.
	defer func() {
		m.warmStartX = nil
		m.warmStartMaxIter = 0
	}()

	// Initial guess: Hannan-Rissanen for AR/MA, OLS for beta, zeros for the
	// rest — UNLESS Update has supplied warm-start parameters from the
	// previous fit. Warm-start skips HR entirely (the existing fit's params
	// are already a near-optimal starting point on the new combined data).
	x0 := make([]float64, nFree)
	useWarmStart := len(m.warmStartX) == nFree
	if useWarmStart {
		copy(x0, m.warmStartX)
	} else {
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

	// Pre-transpose wX to column-major once (it's invariant across the
	// optimizer's residOf calls). This lets the closure dispatch a SIMD
	// AXPY (MulConstAddTo) per regressor column instead of an O(k)
	// inner loop per row — 2.5–2.9× faster on n≥144 with k≥2 exog cols
	// (Apple-silicon NEON / AVX2). On non-SIMD targets go-highway falls
	// back to scalar Go transparently. See docs/decisions/0002-simd-go-highway.md.
	var wXT [][]float64
	if k > 0 {
		wXT = make([][]float64, k)
		nWS := len(ws)
		for j := 0; j < k; j++ {
			col := make([]float64, nWS)
			for i := 0; i < nWS; i++ {
				col[i] = wX[i][j]
			}
			wXT[j] = col
		}
	}

	// Compute residual series given a parameter vector — applied identically
	// inside CSS and Kalman objectives.
	//
	// Hot-path optimization: residOf only needs `c` (intercept) and
	// `beta` (exog coefs) from the unpacked parameters. The
	// AR/MA/seasonal-AR/seasonal-MA tans-formed coefficients are NOT
	// used here. Skipping the full unpackParamsX call eliminates
	// ~25% of per-Fit allocations (arTransparams + maTransparams
	// were the #2 and #4 alloc sources in profiling). We index `c`
	// and `beta` directly from the params layout:
	// [phi(p), theta(q), Phi(P), Theta(Q), c (if intercept), beta(k)].
	cIdx := p + q + P + Q
	residOf := func(params []float64) []float64 {
		c := 0.0
		offset := cIdx
		if m.WithIntercept {
			c = params[offset]
			offset++
		}
		var beta []float64
		if k > 0 {
			beta = params[offset : offset+k]
		}
		out := make([]float64, len(ws))
		for i, v := range ws {
			out[i] = v - c
		}
		for j := 0; j < k; j++ {
			vec.MulConstAddTo(out, -beta[j], wXT[j])
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
	// Same column-transpose trick as residOf, against the un-differenced
	// exog matrix used by the integrated state-space objective.
	var xUndiffT [][]float64
	if k > 0 && xUndiff != nil {
		xUndiffT = make([][]float64, k)
		nU := len(yUndiff)
		for j := 0; j < k; j++ {
			col := make([]float64, nU)
			for i := 0; i < nU; i++ {
				col[i] = xUndiff[i][j]
			}
			xUndiffT[j] = col
		}
	}

	residOfFull := func(params []float64) []float64 {
		c := 0.0
		offset := cIdx
		if m.WithIntercept {
			c = params[offset]
			offset++
		}
		var beta []float64
		if k > 0 {
			beta = params[offset : offset+k]
		}
		out := make([]float64, len(yUndiff))
		for i, v := range yUndiff {
			out[i] = v - c
		}
		for j := 0; j < k; j++ {
			vec.MulConstAddTo(out, -beta[j], xUndiffT[j])
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

	// LSC-1: single-slot exact-match memoization for the ML objective.
	// gonum/optimize sometimes calls f(x) and grad(x) at the same x during
	// BFGS line-search bookkeeping, and BFGS revisits old points after NM
	// polish. Empirically the hit rate is 3% on simple no-exog fits but
	// rises to ~65% on fits with k=5 exog (higher-dim line searches do
	// more bookkeeping). At ~6 µs per Kalman call, even the lower-end hit
	// rates save measurable time.
	//
	// Mutex-protected because parallelGradient calls the objective
	// concurrently from multiple workers. Each worker evaluates at a
	// DIFFERENT x (its own perturbation index) so they all miss the
	// cache; contention is trivial. The cache only delivers hits in the
	// SERIAL phases (BFGS line search bookkeeping, NM polish).
	var lscMu sync.Mutex
	var lscLastX []float64
	var lscLastF float64
	compute := func(params []float64) float64 {
		s := acquireParamScratch()
		defer releaseParamScratch(s)
		phi, theta, sPhi, sTheta, _, _ := unpackParamsXInto(s, params, p, q, P, Q, m.WithIntercept, k)
		// Small-n ridge penalty: applied to the AR/MA portion of the
		// unconstrained x vector (indices 0..p+q+P+Q). Adding `λ·Σx²`
		// to the negLL prevents BFGS from saturating tanh and pushing
		// AR/MA coefs to the unit circle. Indices beyond the AR/MA
		// region (intercept, exog β) are untouched — those don't have
		// boundary issues. Default λ=0 makes this a no-op, preserving
		// R-parity for existing tests.
		var ridgePenalty float64
		if m.RidgePenalty > 0 {
			armaEnd := p + q + P + Q
			if armaEnd > len(params) {
				armaEnd = len(params)
			}
			for i := 0; i < armaEnd; i++ {
				ridgePenalty += params[i] * params[i]
			}
			ridgePenalty *= m.RidgePenalty
		}
		if useStatsmodels {
			Dord := 0
			mPer := 0
			if m.Seasonal.Active() {
				Dord = m.Seasonal.D
				mPer = m.Seasonal.M
			}
			yEff := yUndiffScaled
			if m.WithIntercept || k > 0 {
				_, _, _, _, c, beta := unpackParamsXInto(s, params, p, q, P, Q, m.WithIntercept, k)
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
			return ll + ridgePenalty
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
			return ll + ridgePenalty
		}
		fullPhi := expandSARMAInto(s, phi, sPhi, m.Seasonal.M)
		fullTheta := expandSMAInto(s, theta, sTheta, m.Seasonal.M)
		// Note: residOf and armaCSS still allocate per call. Pooling
		// these (DEEP-AUDIT followup attempt) was empirically a net
		// loss — Go's mcache amortises small-slice allocations more
		// efficiently than ensureLenZ's explicit O(n) zeroing.
		// Documented in PERF_TODO.
		r := residOf(params)
		switch m.Method {
		case MethodCSS:
			ll, _, _ := armaCSS(r, fullPhi, fullTheta)
			return ll + ridgePenalty
		default:
			// KAL-WORKSPACE: pooled buffers via paramScratch.kalman.
			ll, _ := kalmanARMALikelihoodInto(r, fullPhi, fullTheta, &s.kalman)
			if math.IsNaN(ll) || math.IsInf(ll, 0) {
				return math.Inf(1)
			}
			return ll + ridgePenalty
		}
	}
	objective := func(params []float64) float64 {
		lscMu.Lock()
		if lscLastX != nil && len(lscLastX) == len(params) {
			match := true
			for i := range params {
				if params[i] != lscLastX[i] {
					match = false
					break
				}
			}
			if match {
				cached := lscLastF
				lscMu.Unlock()
				return cached
			}
		}
		lscMu.Unlock()
		ll := compute(params)
		lscMu.Lock()
		if cap(lscLastX) < len(params) {
			lscLastX = make([]float64, len(params))
		} else {
			lscLastX = lscLastX[:len(params)]
		}
		copy(lscLastX, params)
		lscLastF = ll
		lscMu.Unlock()
		return ll
	}

	// Skip the CSS warmup phase when warm-starting: the existing fit's
	// optimizer-space x is already a refined point, and the whole purpose
	// of warm-start is to avoid the redundant work CSS would do here.
	cssWarmedX := false
	if m.Method == MethodCSSML && !useWarmStart {
		cssObj := func(params []float64) float64 {
			s := acquireParamScratch()
			defer releaseParamScratch(s)
			phi, theta, sPhi, sTheta, _, _ := unpackParamsXInto(s, params, p, q, P, Q, m.WithIntercept, k)
			fullPhi := expandSARMAInto(s, phi, sPhi, m.Seasonal.M)
			fullTheta := expandSMAInto(s, theta, sTheta, m.Seasonal.M)
			ll, _, _ := armaCSS(residOf(params), fullPhi, fullTheta)
			// Ridge penalty applies in CSS warmup too — otherwise the
			// warmup could land on a boundary point, propagating bad
			// x0 into ML refinement.
			if m.RidgePenalty > 0 {
				armaEnd := p + q + P + Q
				if armaEnd > len(params) {
					armaEnd = len(params)
				}
				rp := 0.0
				for i := 0; i < armaEnd; i++ {
					rp += params[i] * params[i]
				}
				ll += m.RidgePenalty * rp
			}
			return ll
		}
		x0 = minimize(cssObj, x0, m.MaxIter, m.gradientWorkers(), len(ws))
		cssWarmedX = true
	}

	mi := m.MaxIter
	if m.warmStartMaxIter > 0 {
		mi = m.warmStartMaxIter
	}
	// Warm-started ML phase (after CSS or after a previous fit) can skip
	// the Nelder-Mead polish when BFGS converges cleanly — CSS already did
	// the global search, so NM here is mostly redundant.
	mlWarmStarted := cssWarmedX || useWarmStart
	best := minimizeNM(objective, x0, mi, m.gradientWorkers(), mlWarmStarted, len(ws))
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
		// KAL-1 sanity check. The exact-Kalman likelihood includes a
		// log-determinant transient bounded by O(log γ(0)/σ²) — typically
		// a few units, at most ~3·r for well-conditioned stationary fits.
		// When BFGS pushes φ/Φ near the unit circle, stationaryCovGardner
		// returns huge initial P (1e10+) and the resulting log F sum
		// dominates negLL with values that disagree wildly with the
		// concentrated-Gaussian textbook form. R's stats::arima falls
		// back to the conditional likelihood in such cases; we mirror
		// that here. The parameters themselves are unchanged — only the
		// reported logL/AIC/AICc are sanitized so model selection isn't
		// driven by this numerical artefact.
		if !math.IsNaN(negLL) && !math.IsInf(negLL, 0) && sigma2 > 0 {
			textbookNegLL := 0.5 * float64(len(r)) * (math.Log(2*math.Pi*sigma2) + 1)
			rState := len(fullPhi)
			if len(fullTheta)+1 > rState {
				rState = len(fullTheta) + 1
			}
			tol := 3.0 * float64(rState)
			if tol < 20 {
				tol = 20
			}
			if math.Abs(negLL-textbookNegLL) > tol {
				negLL = textbookNegLL
			}
		}
	}
	if math.IsNaN(negLL) || math.IsInf(negLL, 0) {
		// KAL-1: when the Kalman early-aborts (e.g., F<=0 from
		// non-PSD P at boundary parameters), recover σ² via CSS and
		// report the textbook concentrated-Gaussian negLL with full
		// constants. Pre-fix this used (n/2)·log(σ²) — missing the
		// (n/2)·(log(2π)+1) offset, so reported AIC/AICc were ~n·1.42
		// units lower than R/statsmodels for the same fit.
		_, sigma2, _ = armaCSS(r, fullPhi, fullTheta)
		negLL = 0.5 * float64(len(r)) * (math.Log(2*math.Pi*sigma2) + 1)
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
//
// To force sequential gradient regardless of n, set the caller's
// nWorkers=1. AutoArima exposes this via `AutoArimaOpts.GradientWorkers=1`
// — useful in environments where pthread_cond_signal scheduling cost
// exceeds the parallel arithmetic speedup (e.g. some Linux container
// configurations with limited cores or NUMA effects).
//
// `dataLen` is reserved for future work-product-based gating heuristics;
// currently only the n<4 floor applies.
func parallelGradient(f func([]float64) float64, nWorkers, dataLen int) func(grad, x []float64) {
	_ = dataLen // reserved for future work-product threshold
	const eps = 1e-7
	if nWorkers < 1 {
		nWorkers = 1
	}
	// serialBuf is reused across gradient calls in the serial path; sized
	// to len(x) on first call, grown if needed.
	var serialBuf []float64
	return func(grad, x []float64) {
		n := len(x)
		// Sequential when nWorkers=1 (caller-forced) or n < 4
		// (per-call goroutine overhead floor). Use a local copy of x
		// so the perturbation pattern mirrors the parallel path exactly
		// (fresh `copy(xLocal, x); xLocal[i]+=eps; …`). The earlier
		// in-place version mutated x[] across f-calls, which interacted
		// non-deterministically with optimizer-side x reuse on certain
		// problems (m5_with_exog, sunspot fast) — same orderKey, two
		// different local minima depending on which worker count was
		// used. The local copy fully serializes f-input identity so
		// serial-vs-parallel produce bit-equivalent gradients.
		if nWorkers == 1 || n < 4 {
			// Bit-equivalent to the parallel-branch perturbation
			// (`+= eps; -= 2*eps`). The earlier in-place serial used
			// `save+eps` / `save-eps` directly, which differs from
			// the parallel pattern by 1 ulp at the perturbation
			// arithmetic. On most problems that ulp drift is harmless
			// noise, but on ill-conditioned objectives (m5_with_exog
			// intermittent demand has many close-by local minima)
			// even ulp-level gradient drift can push BFGS into a
			// different basin, surfacing as same-orderKey/different-
			// AICc when the search visited more candidates first.
			// Aligning serial bit-for-bit with parallel makes the
			// fit deterministic regardless of nWorkers.
			if cap(serialBuf) < n {
				serialBuf = make([]float64, n)
			}
			xLocal := serialBuf[:n]
			for i := 0; i < n; i++ {
				copy(xLocal, x)
				xLocal[i] += eps
				fp := f(xLocal)
				xLocal[i] -= 2 * eps
				fm := f(xLocal)
				grad[i] = (fp - fm) / (2 * eps)
			}
			return
		}
		nw := nWorkers
		if nw > n {
			nw = n
		}
		var wg sync.WaitGroup
		buf := make([]float64, nw*n)
		for w := 0; w < nw; w++ {
			w := w
			wg.Add(1)
			go func() {
				defer wg.Done()
				xLocal := buf[w*n : (w+1)*n]
				for i := w; i < n; i += nw {
					copy(xLocal, x)
					xLocal[i] += eps
					fp := f(xLocal)
					xLocal[i] -= 2 * eps
					fm := f(xLocal)
					xLocal[i] += eps
					grad[i] = (fp - fm) / (2 * eps)
				}
			}()
		}
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

func minimize(f func([]float64) float64, x0 []float64, maxIter, nWorkers, dataLen int) []float64 {
	return minimizeNM(f, x0, maxIter, nWorkers, false, dataLen)
}

// minimizeNM is the BFGS+Nelder-Mead optimizer. When `warmStarted` is true,
// the caller is signalling that x0 is already close to the optimum (e.g.
// after a CSS warmup phase that fed the ML refinement). In that case, NM
// is skipped when BFGS converges cleanly — saving ~30-50% of optimizer
// time. NM is still run when BFGS fails, when warmStarted is false, or
// when BFGS reports a non-convergence status, since SARIMA likelihoods
// have many local minima and NM is the only reliable global-search step.
//
// Empirically: skipping NM after a converged warm-started BFGS keeps
// likelihood within ~1e-6 units (well under the parity test tolerances)
// and saves the cost of a full second optimizer run. See PERF_TODO
// CSS-1 for the bench numbers.
func minimizeNM(f func([]float64) float64, x0 []float64, maxIter, nWorkers int, warmStarted bool, dataLen int) []float64 {
	if nWorkers < 1 {
		nWorkers = runtime.GOMAXPROCS(0)
	}
	gradFn := parallelGradient(f, nWorkers, dataLen)
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
	// without NM). So we cannot skip NM outright on cold starts. When
	// warmStarted=true and BFGS converges cleanly, we DO skip NM — the
	// CSS phase has already done the global search.
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

	if warmStarted && bfgsConverged {
		return bestX
	}

	nmIters := maxIter * 4
	nmFuncEvals := maxIter * 200
	if bfgsConverged || warmStarted {
		// Cheap polish — still escapes most local minima but at ~1/4 cost.
		// We use the reduced budget when EITHER (a) BFGS converged cleanly,
		// or (b) we were warm-started: in both cases the global search is
		// already done and NM only needs to refine.
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
//
// Convention: AIC / AICc / BIC / HQIC are computed as
//
//	AIC  = 2k − 2·logL
//	AICc = AIC + 2k(k+1)/(n−k−1)
//	BIC  = log(n)·k − 2·logL
//	HQIC = 2·log(log(n))·k − 2·logL
//
// where k counts σ² (always +1) plus the AR/MA/seasonal-AR/seasonal-MA
// counts, plus 1 if WithIntercept, plus the exog regressor count.
//
// `logL` is the maximised log-likelihood as stored in `m.logL`. Whether
// it includes the Gaussian normalising constants `(n/2)·log(2π) + n/2`
// depends on the fit Method:
//
//   - **MethodCSSML / MethodML** (Kalman path) — INCLUDES the constants.
//     Reported logL = −0.5·(n·(log(2π·σ²)+1) + Σ log F_t). Same scale as
//     R's `stats::arima` and Python's `statsmodels.SARIMAX` — directly
//     comparable.
//   - **MethodCSS** (CSS path) — EXCLUDES the constants. Reported
//     logL = −(n/2)·log(σ²). To get an MLE-comparable value add
//     `n·(log(2π) + 1) / 2`. R's `stats::arima(method="CSS")` follows
//     the same drop-the-constants convention.
//
// Cross-method AIC comparisons therefore offset by `n·(log(2π) + 1)`;
// within-method comparisons (the typical use case) are unaffected.
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
//
// **Includes Gaussian normalising constants under MethodCSSML / MethodML
// (the Kalman path) — directly comparable to R's `logLik(stats::arima(...))`
// and statsmodels SARIMAX results. EXCLUDES the constants under
// MethodCSS, matching R's `stats::arima(method="CSS")` convention; add
// `n·(log(2π) + 1) / 2` to convert.** See `IC()` doc for details.
func (m *ARIMA) LogLikelihood() float64 { return m.logL }

// Sigma2 returns the residual variance estimate.
func (m *ARIMA) Sigma2() float64 { return m.sigma2 }

// Resid returns the in-sample residuals aligned to the original time index.
// Output length equals len(yTrain). The first `d + D*m` entries are NaN —
// the differencing-warmup region where one-step-ahead residuals aren't
// defined. Matches both pmdarima.ARIMA.arima_res_.resid (which returns
// length-len(y) but fills warmup with innovations) and R's
// residuals.Arima (which returns a ts of length n with NA in warmup).
//
// Older versions returned a shorter slice without the warmup region —
// callers that pass the result straight to a length-agnostic test (Ljung-
// Box, ACF) keep working; callers that align by index now use the value
// directly (matches yTrain[i]) and skip NaN entries.
func (m *ARIMA) Resid() []float64 {
	if !m.fitted {
		return nil
	}
	dHead := m.Order.D
	if m.Seasonal.Active() {
		dHead += m.Seasonal.D * m.Seasonal.M
	}
	fullLen := len(m.yTrain)
	if fullLen == 0 {
		return nil
	}
	out := make([]float64, fullLen)
	for i := 0; i < dHead; i++ {
		out[i] = math.NaN()
	}
	residCount := fullLen - dHead
	if residCount > 0 && len(m.resids) > 0 {
		// Take the trailing residCount entries of m.resids (front-pad with
		// zeros if shorter, mirroring FittedValues' resid alignment).
		residTail := m.resids
		if len(residTail) > residCount {
			residTail = residTail[len(residTail)-residCount:]
		} else if len(residTail) < residCount {
			pad := make([]float64, residCount-len(residTail))
			residTail = append(pad, residTail...)
		}
		copy(out[dHead:], residTail)
	}
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

// FittedValues returns in-sample one-step-ahead predictions y_hat[t],
// aligned to the original time index. Output length equals len(yTrain).
//
// The first `d + D*m` entries are NaN — they're the differencing-warmup
// region where one-step-ahead predictions aren't defined. This matches
// pmdarima.ARIMA.predict_in_sample (returns NaN-padded array) and R's
// fitted.Arima (returns ts with NA in the warmup).
//
// Computed as y_hat[t] = y[t] - residual[t] on the model scale, then
// inverse-Box-Cox (if Lambda was set) so the output is in the user's
// original units. Both operands are on the model scale before the inverse,
// so the arithmetic is unit-consistent under Box-Cox.
//
// Older versions returned a shorter slice without the warmup region —
// callers that index by `i` should now use the value directly (it aligns
// with yTrain[i]) and skip NaN entries.
func (m *ARIMA) FittedValues() []float64 {
	if !m.fitted {
		return nil
	}
	dHead := m.Order.D
	if m.Seasonal.Active() {
		dHead += m.Seasonal.D * m.Seasonal.M
	}
	fullLen := len(m.yTrain)
	residCount := fullLen - dHead
	if residCount <= 0 || len(m.resids) == 0 {
		return nil
	}
	residTail := m.resids
	if len(residTail) > residCount {
		residTail = residTail[len(residTail)-residCount:]
	} else if len(residTail) < residCount {
		// pad front with zeros
		pad := make([]float64, residCount-len(residTail))
		residTail = append(pad, residTail...)
	}
	// Use the model-scale yTrain (Box-Cox-applied if Lambda is set) so the
	// subtraction is unit-consistent with the residuals.
	yMS := m.yMSCache
	if yMS == nil {
		yMS = m.yTrain // older snapshots without cache; Lambda must be nil
	}
	out := make([]float64, fullLen)
	// Warmup region: NaN, matching pmdarima/R behavior.
	for i := 0; i < dHead; i++ {
		out[i] = math.NaN()
	}
	for i := 0; i < residCount; i++ {
		out[dHead+i] = yMS[dHead+i] - residTail[i]
	}
	if m.Lambda != nil {
		// boxCoxInvert preserves NaN for the warmup positions.
		out = boxCoxInvert(out, *m.Lambda, m.Lambda2)
	}
	return out
}

// PredictInSample returns the in-sample one-step-ahead predictions, alias of
// FittedValues. Mirrors pmdarima.ARIMA.predict_in_sample.
func (m *ARIMA) PredictInSample() []float64 { return m.FittedValues() }

// Update appends new observations to the training data and runs a quick
// MLE refresh on the existing parameters. Mirrors pmdarima.ARIMA.update
// and R's `Arima(model = existing, x = new_y)` — same orders and intercept
// choice are kept; only the parameter values move slightly to accommodate
// the new data.
//
// Use Update when:
//   - You have new observations and want to refresh the fit quickly.
//   - You want to preserve the AutoArima-selected orders.
//
// Use Refit when:
//   - You want a full cold-start fit on the combined data, including
//     Hannan-Rissanen warmup and Nelder-Mead polish. Slower; matches
//     calling Fit on the combined series from scratch.
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

	// Pack the existing fit's parameters back into the optimizer's
	// transformed parameter space. arTransparams / maTransparams are
	// non-linear maps from R^p → stationarity-bounded space; their
	// inverses (invertARTransform / invertMATransform) recover the
	// optimizer x from m.phi / m.theta.
	p := m.Order.P
	q := m.Order.Q
	P := 0
	Q := 0
	if m.Seasonal.Active() {
		P = m.Seasonal.P
		Q = m.Seasonal.Q
	}
	k := m.nExog
	nFree := p + q + P + Q + k
	if m.WithIntercept {
		nFree++
	}
	if nFree > 0 {
		warmX := make([]float64, nFree)
		off := 0
		if p > 0 {
			copy(warmX[off:off+p], invertARTransform(m.phi))
			off += p
		}
		if q > 0 {
			copy(warmX[off:off+q], invertMATransform(m.theta))
			off += q
		}
		if P > 0 {
			copy(warmX[off:off+P], invertARTransform(m.Phi))
			off += P
		}
		if Q > 0 {
			copy(warmX[off:off+Q], invertMATransform(m.Theta))
			off += Q
		}
		if m.WithIntercept {
			warmX[off] = m.c
			off++
		}
		if k > 0 {
			copy(warmX[off:off+k], m.beta)
		}
		m.warmStartX = warmX
		// Tight iteration budget — we're already near optimum.
		m.warmStartMaxIter = 25
	}

	combinedY := append([]float64{}, m.yTrain...)
	combinedY = append(combinedY, newY...)
	var combinedX [][]float64
	if newX != nil {
		combinedX = append(combinedX, m.xTrain...)
		combinedX = append(combinedX, cloneMat(newX)...)
	}
	// Fit clears m.warmStartX / m.warmStartMaxIter via deferred cleanup
	// regardless of success/failure.
	return m.Fit(combinedY, combinedX)
}

// Refit appends new observations and runs a full cold-start fit on the
// combined series — Hannan-Rissanen warmup, full BFGS, and Nelder-Mead
// polish. Equivalent to calling Fit on `[m.yTrain..newY]`.
//
// Slower than Update but more thorough: reaches a different local
// optimum if the new data shifts the likelihood landscape enough that
// the existing parameter neighborhood isn't optimal anymore.
//
// Note: Refit does NOT re-search ARIMA orders — `m.Order` and
// `m.Seasonal` are preserved from the existing model. To re-search
// orders, run AutoArima fresh on the combined series.
func (m *ARIMA) Refit(newY []float64, newX [][]float64) error {
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
// is false. Returns a clean error when the user's futureExog row count
// doesn't match nPeriods, instead of letting the caller-side check fire
// later — pre-fix the loop below indexed `futureExog[i]` for i < nPeriods
// without a length guard, producing an opaque slice-out-of-range panic.
func (m *ARIMA) extendDriftIfNeeded(futureExog [][]float64, nPeriods int) ([][]float64, error) {
	if !m.DriftIncluded || nPeriods <= 0 {
		return futureExog, nil
	}
	if futureExog != nil && len(futureExog) != nPeriods {
		return nil, fmt.Errorf("future exog rows (%d) must match nPeriods (%d)",
			len(futureExog), nPeriods)
	}
	n0 := len(m.yTrain)
	if futureExog == nil {
		// Drift is the only exog column. Build it directly.
		out := make([][]float64, nPeriods)
		for i := 0; i < nPeriods; i++ {
			out[i] = []float64{float64(n0 + i + 1)}
		}
		return out, nil
	}
	// User supplied the OTHER exog columns; prepend drift.
	out := make([][]float64, nPeriods)
	for i := 0; i < nPeriods; i++ {
		row := make([]float64, 1+len(futureExog[i]))
		row[0] = float64(n0 + i + 1)
		copy(row[1:], futureExog[i])
		out[i] = row
	}
	return out, nil
}

// PredictVar returns the per-step forecast variance for nPeriods ahead.
// Variance at horizon h is `σ² · Σ_{i=0..h} psi[i]²` where `psi` are
// the MA(∞) coefficients of the fitted ARIMA — same computation that
// drives Predict's CI bands but exposed directly so callers can build
// custom intervals (e.g. asymmetric or non-Gaussian) without
// reverse-engineering the bands.
//
// When the model was fitted with Box-Cox (`m.Lambda != nil`), the
// returned variance is on the **model (post-transform) scale**. There's
// no closed-form variance on the original scale because the Box-Cox
// inverse is non-linear. For original-scale uncertainty use Predict's
// bands (which inverse-transform endpoints) or PredictBoot (empirical
// quantiles on inverted paths).
//
// futureExog is required when the model was fitted with exog (validated
// for shape) but does NOT affect the variance — variance is determined
// by AR/MA structure and σ², not by the exog values. Argument kept for
// API symmetry with Predict.
//
// Closes PRED-VAR.
func (m *ARIMA) PredictVar(nPeriods int, futureExog [][]float64) ([]float64, error) {
	if !m.fitted {
		return nil, errors.New("model not fitted")
	}
	if nPeriods <= 0 {
		return []float64{}, nil
	}
	var driftErr error
	futureExog, driftErr = m.extendDriftIfNeeded(futureExog, nPeriods)
	if driftErr != nil {
		return nil, driftErr
	}
	if m.nExog > 0 {
		if futureExog == nil || len(futureExog) != nPeriods {
			return nil, fmt.Errorf("future exog rows (%d) must match nPeriods (%d)",
				len(futureExog), nPeriods)
		}
		for i, row := range futureExog {
			if len(row) != m.nExog {
				return nil, fmt.Errorf("future exog row %d cols (%d) must match training (%d)",
					i, len(row), m.nExog)
			}
		}
	} else if futureExog != nil {
		return nil, errors.New("model was fitted without exog; do not pass futureExog")
	}
	psi := m.ensurePsiAtLeast(nPeriods)
	out := make([]float64, nPeriods)
	cum := 0.0
	for h := 0; h < nPeriods; h++ {
		if h < len(psi) {
			cum += psi[h] * psi[h]
		}
		out[h] = m.sigma2 * cum
	}
	return out, nil
}

// PredictSE returns per-step forecast standard error: sqrt(PredictVar).
// See PredictVar for caveats (model-scale under Box-Cox; futureExog
// kept for symmetry but doesn't affect output).
func (m *ARIMA) PredictSE(nPeriods int, futureExog [][]float64) ([]float64, error) {
	v, err := m.PredictVar(nPeriods, futureExog)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(v))
	for i, vi := range v {
		out[i] = math.Sqrt(vi)
	}
	return out, nil
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
	var driftErr error
	futureExog, driftErr = m.extendDriftIfNeeded(futureExog, nPeriods)
	if driftErr != nil {
		return nil, nil, nil, driftErr
	}
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

	// Forecast residuals on differenced/centered scale. Pre-fix this copied
	// the entire training history (`append([]float64{}, wsCentered...)`)
	// even though the AR/MA recursion only reads the last len(fullPhi) /
	// len(fullTheta) elements. Same fix as #C18 for PredictBoot: extract the
	// AR/MA lag windows once and reuse small buffers sized to lag + horizon.
	pLag := len(fullPhi)
	qLag := len(fullTheta)
	phiWin := lastN(wsCentered, pLag)
	thetaWin := lastN(res, qLag)
	thetaOriginalLen := len(thetaWin)
	forecastDiffed := make([]float64, nPeriods)
	hist := make([]float64, 0, pLag+nPeriods)
	hist = append(hist, phiWin...)
	residHist := make([]float64, 0, qLag+nPeriods)
	residHist = append(residHist, thetaWin...)
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
			// Only read the ORIGINAL training residuals (the windowed prefix).
			// Future innovations are zero in expectation — leaving them as the
			// zero pads we append below. This matches the pre-fix behaviour
			// where `idx < len(res)` excluded the just-appended zeros from
			// contributing to the MA(future) part redundantly.
			if idx >= 0 && idx < thetaOriginalLen {
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

	// Confidence intervals using truncated MA(infinity) coefficients. The
	// psi cache is published via atomic.Pointer so concurrent Predict
	// calls on the same fitted model are safe — every reader gets a
	// stable immutable snapshot, and cache extension (when nPeriods
	// exceeds the cached length) installs a new larger snapshot via CAS
	// without invalidating any concurrent reader's view.
	if alpha > 0 && alpha < 1 {
		psi := m.ensurePsiAtLeast(nPeriods)
		if len(psi) > 0 {
			z := normPPF(1 - alpha/2)
			lower = make([]float64, nPeriods)
			upper = make([]float64, nPeriods)
			var2 := 0.0
			for h := 0; h < nPeriods; h++ {
				if h < len(psi) {
					var2 += psi[h] * psi[h]
				}
				se := math.Sqrt(m.sigma2 * var2)
				lower[h] = out[h] - z*se
				upper[h] = out[h] + z*se
			}
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

// computePsi precomputes truncated MA(infinity) coefficients for forecast
// variance and publishes them via the atomic psi cache. Called once from
// Fit with a 100-entry default; Predict extends the cache if a longer
// horizon is requested (see ensurePsiAtLeast).
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
	psi := m.buildPsi(100)
	m.psi.Store(&psiCache{values: psi})
}

// buildPsi constructs an immutable MA(∞) coefficient slice of length maxLag.
// Pure builder — no side effects on m. Callers either Store() the result
// directly (Fit's initial publish) or CompareAndSwap a wrapping psiCache
// to monotonically grow the published cache (Predict's lazy extension).
func (m *ARIMA) buildPsi(maxLag int) []float64 {
	if maxLag < 1 {
		maxLag = 1
	}
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
	return psi
}

// ensurePsiAtLeast returns a psi slice of length ≥ minLen, atomically
// extending the cached snapshot if needed. Concurrent-safe: every caller
// gets a coherent slice (either the cached one or a freshly built one);
// the cache only ever grows monotonically via CAS.
//
// Returns (nil) when nFree==0 (white-noise model — no psi to compute).
func (m *ARIMA) ensurePsiAtLeast(minLen int) []float64 {
	for {
		cur := m.psi.Load()
		if cur != nil && len(cur.values) >= minLen {
			return cur.values
		}
		// Build a longer snapshot and CAS-publish. If another goroutine
		// raced in with its own (possibly larger) snapshot, the CAS fails
		// and we re-load — discarding our build but inheriting whatever
		// won. Worst-case waste under contention is one redundant build,
		// which is fine.
		next := &psiCache{values: m.buildPsi(minLen)}
		if m.psi.CompareAndSwap(cur, next) {
			return next.values
		}
	}
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
