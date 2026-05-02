package arima

import (
	"errors"
	"fmt"
	"math"

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

// ARIMA fits an ARIMA(p,d,q)(P,D,Q,m) time-series model.
//
// API mirrors pmdarima.arima.ARIMA.
type ARIMA struct {
	// Configuration
	Order         Order
	Seasonal      SeasonalOrder
	WithIntercept bool   // include constant term in differenced series
	Method        Method // fitting method
	MaxIter       int    // optimizer max iterations (default 50)

	// Fitted state
	phi   []float64 // non-seasonal AR
	theta []float64 // non-seasonal MA
	Phi   []float64 // seasonal AR
	Theta []float64 // seasonal MA
	c     float64   // intercept (on differenced series)
	mean  float64   // mean of differenced series (when no intercept)

	sigma2  float64
	logL    float64
	nobs    int
	resids  []float64
	yTrain  []float64 // original y, for forecasts
	dHead   []float64 // values consumed by non-seasonal differencing
	sdHead  []float64 // values consumed by seasonal differencing
	fitted  bool
	psiInf  []float64 // truncated MA(infinity) for forecast variance
	psiInfN int
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

// Fit estimates the parameters from training data y.
func (m *ARIMA) Fit(y []float64) error {
	if len(y) == 0 {
		return errors.New("y must be non-empty")
	}
	if m.MaxIter == 0 {
		m.MaxIter = 100
	}
	m.yTrain = append([]float64{}, y...)

	// Apply differencing.
	ws := y
	if m.Order.D > 0 {
		m.dHead = append([]float64{}, ws[:m.Order.D]...)
		ws = applyDiff(ws, 1, m.Order.D)
	}
	if m.Seasonal.Active() && m.Seasonal.D > 0 {
		hd := m.Seasonal.D * m.Seasonal.M
		if hd > len(ws) {
			return fmt.Errorf("seasonal differencing %d exceeds data length %d", hd, len(ws))
		}
		m.sdHead = append([]float64{}, ws[:hd]...)
		ws = applyDiff(ws, m.Seasonal.M, m.Seasonal.D)
	}
	if len(ws) < 2 {
		return errors.New("differenced series too short")
	}

	// Statsmodels behavior: no constant by default when d>0 or D>0.
	// Only center on the mean when the series is fully un-differenced.
	mean := 0.0
	totalDiff := m.Order.D
	if m.Seasonal.Active() {
		totalDiff += m.Seasonal.D
	}
	if !m.WithIntercept && totalDiff == 0 {
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

	// Define parameter layout: [phi*p, theta*q, Phi*P, Theta*Q, (intercept), log_sigma]
	p, q := m.Order.P, m.Order.Q
	P, Q := 0, 0
	if m.Seasonal.Active() {
		P, Q = m.Seasonal.P, m.Seasonal.Q
	}
	nFree := p + q + P + Q
	if m.WithIntercept {
		nFree++
	}
	if nFree == 0 {
		// Pure white-noise / random walk after differencing
		sse := 0.0
		for _, v := range ws {
			sse += v * v
		}
		m.sigma2 = sse / float64(len(ws))
		m.resids = append([]float64{}, ws...)
		m.logL = -0.5 * float64(len(ws)) * (math.Log(2*math.Pi*m.sigma2) + 1)
		m.fitted = true
		m.computePsi()
		return nil
	}

	// Initial guess: zeros (transformed → near zero coefficients).
	x0 := make([]float64, nFree)

	objective := func(params []float64) float64 {
		phi, theta, sPhi, sTheta, c := unpackParams(params, p, q, P, Q, m.WithIntercept)
		// Combine seasonal × non-seasonal AR/MA polynomials.
		fullPhi := expandSARMA(phi, sPhi, m.Seasonal.M)
		fullTheta := expandSMA(theta, sTheta, m.Seasonal.M)
		// Subtract intercept (acts on the differenced series after centering)
		if m.WithIntercept {
			for i := range ws {
				ws[i] -= c
			}
			defer func() {
				for i := range ws {
					ws[i] += c
				}
			}()
		}
		switch m.Method {
		case MethodCSS:
			ll, _, _ := armaCSS(ws, fullPhi, fullTheta)
			return ll
		default: // ML or CSSML — use exact Kalman likelihood
			ll, _, _ := kalmanARMALikelihood(ws, fullPhi, fullTheta)
			if math.IsNaN(ll) || math.IsInf(ll, 0) {
				return math.Inf(1)
			}
			return ll
		}
	}

	// CSS-ML: warm-start with CSS optimum
	if m.Method == MethodCSSML {
		cssObj := func(params []float64) float64 {
			phi, theta, sPhi, sTheta, c := unpackParams(params, p, q, P, Q, m.WithIntercept)
			fullPhi := expandSARMA(phi, sPhi, m.Seasonal.M)
			fullTheta := expandSMA(theta, sTheta, m.Seasonal.M)
			if m.WithIntercept {
				adj := make([]float64, len(ws))
				for i, v := range ws {
					adj[i] = v - c
				}
				ll, _, _ := armaCSS(adj, fullPhi, fullTheta)
				return ll
			}
			ll, _, _ := armaCSS(ws, fullPhi, fullTheta)
			return ll
		}
		x0 = minimize(cssObj, x0, m.MaxIter)
	}

	best := minimize(objective, x0, m.MaxIter)
	phi, theta, sPhi, sTheta, c := unpackParams(best, p, q, P, Q, m.WithIntercept)
	m.phi = phi
	m.theta = theta
	m.Phi = sPhi
	m.Theta = sTheta
	m.c = c

	fullPhi := expandSARMA(phi, sPhi, m.Seasonal.M)
	fullTheta := expandSMA(theta, sTheta, m.Seasonal.M)
	wsAdj := ws
	if m.WithIntercept {
		wsAdj = make([]float64, len(ws))
		for i, v := range ws {
			wsAdj[i] = v - c
		}
	}
	negLL, sigma2, _ := kalmanARMALikelihood(wsAdj, fullPhi, fullTheta)
	if math.IsNaN(negLL) || math.IsInf(negLL, 0) {
		// fallback to CSS variance
		_, sigma2, _ = armaCSS(wsAdj, fullPhi, fullTheta)
		negLL = float64(len(wsAdj)) / 2 * math.Log(sigma2)
	}
	m.sigma2 = sigma2
	m.logL = -negLL

	// recompute residuals via CSS for storage
	_, _, res := armaCSS(wsAdj, fullPhi, fullTheta)
	m.resids = res

	m.fitted = true
	m.computePsi()
	return nil
}

// minimize runs BFGS with a numerical gradient. Falls back to Nelder-Mead
// if BFGS fails or stalls. Returns the best parameter vector found.
func minimize(f func([]float64) float64, x0 []float64, maxIter int) []float64 {
	gradFn := func(grad, x []float64) {
		const eps = 1e-7
		base := f(x)
		_ = base
		for i := range x {
			save := x[i]
			x[i] = save + eps
			fp := f(x)
			x[i] = save - eps
			fm := f(x)
			x[i] = save
			grad[i] = (fp - fm) / (2 * eps)
		}
	}
	prob := optimize.Problem{Func: f, Grad: gradFn}
	settings := &optimize.Settings{
		MajorIterations:   maxIter,
		FuncEvaluations:   maxIter * 50,
		GradientThreshold: 1e-6,
	}
	bestX := make([]float64, len(x0))
	copy(bestX, x0)
	bestF := f(x0)

	if res, err := optimize.Minimize(prob, x0, settings, &optimize.BFGS{}); err == nil && res != nil {
		if res.F < bestF {
			bestF = res.F
			copy(bestX, res.X)
		}
	}

	// Refine with Nelder-Mead.
	probNM := optimize.Problem{Func: f}
	settingsNM := &optimize.Settings{
		MajorIterations: maxIter * 4,
		FuncEvaluations: maxIter * 200,
	}
	if res, err := optimize.Minimize(probNM, bestX, settingsNM, &optimize.NelderMead{}); err == nil && res != nil {
		if res.F < bestF {
			bestX = make([]float64, len(res.X))
			copy(bestX, res.X)
		}
	}
	return bestX
}

// unpackParams splits the flat parameter vector into AR/MA/seasonal/intercept.
// Applies stationarity/invertibility transforms.
func unpackParams(params []float64, p, q, P, Q int, withIntercept bool) (phi, theta, sPhi, sTheta []float64, c float64) {
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
	}
	return
}

// IC returns the chosen information criterion of the fitted model.
func (m *ARIMA) IC(ic InfoCriterion) float64 {
	if !m.fitted {
		return math.Inf(1)
	}
	k := float64(len(m.phi) + len(m.theta) + len(m.Phi) + len(m.Theta))
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

// Params returns the fitted parameter vector in the order
// [phi..., theta..., Phi..., Theta..., intercept (if any)].
func (m *ARIMA) Params() []float64 {
	out := make([]float64, 0, len(m.phi)+len(m.theta)+len(m.Phi)+len(m.Theta)+1)
	out = append(out, m.phi...)
	out = append(out, m.theta...)
	out = append(out, m.Phi...)
	out = append(out, m.Theta...)
	if m.WithIntercept {
		out = append(out, m.c)
	}
	return out
}

// Predict produces nPeriods forward forecasts. If alpha > 0, lower/upper
// confidence intervals are returned alongside; otherwise lower/upper are nil.
func (m *ARIMA) Predict(nPeriods int, alpha float64) (forecast, lower, upper []float64, err error) {
	if !m.fitted {
		return nil, nil, nil, errors.New("model not fitted")
	}
	if nPeriods <= 0 {
		return []float64{}, nil, nil, nil
	}
	fullPhi := expandSARMA(m.phi, m.Phi, m.Seasonal.M)
	fullTheta := expandSMA(m.theta, m.Theta, m.Seasonal.M)

	// Reconstruct differenced (and centered/intercept-adjusted) training series.
	ws := append([]float64{}, m.yTrain...)
	if m.Order.D > 0 {
		ws = applyDiff(ws, 1, m.Order.D)
	}
	if m.Seasonal.Active() && m.Seasonal.D > 0 {
		ws = applyDiff(ws, m.Seasonal.M, m.Seasonal.D)
	}
	wsCentered := make([]float64, len(ws))
	for i, v := range ws {
		wsCentered[i] = v - m.mean - m.c
	}
	// Recompute CSS residuals for forecasting.
	_, _, res := armaCSS(wsCentered, fullPhi, fullTheta)

	// Forecast on differenced series (centered).
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
	// Re-add intercept and mean
	for i := range forecastDiffed {
		forecastDiffed[i] += m.mean + m.c
	}

	// Integrate seasonal differencing back
	out := forecastDiffed
	if m.Seasonal.Active() && m.Seasonal.D > 0 {
		head := lastN(diffStream(m.yTrain, 1, m.Order.D), m.Seasonal.D*m.Seasonal.M)
		full := integrateBack(out, head, m.Seasonal.M, m.Seasonal.D)
		out = full[len(head):]
	}
	if m.Order.D > 0 {
		head := lastN(m.yTrain, m.Order.D)
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
	return out, lower, upper, nil
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

// computePsi precomputes truncated MA(infinity) coefficients for forecast variance.
func (m *ARIMA) computePsi() {
	maxLag := 100
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
	m.psiInf = psi
	m.psiInfN = maxLag
}

// normPPF returns the inverse CDF of the standard normal.
func normPPF(p float64) float64 {
	// Beasley–Springer–Moro algorithm.
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
