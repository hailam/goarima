package arima

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// CoefSummary describes one fitted parameter.
type CoefSummary struct {
	Name   string
	Value  float64
	StdErr float64
	Z      float64
	PValue float64
}

// FitSummary aggregates coefficients and goodness-of-fit numbers for printing.
type FitSummary struct {
	Order    Order
	Seasonal SeasonalOrder
	NObs     int
	LogLik   float64
	AIC      float64
	BIC      float64
	AICc     float64
	HQIC     float64
	Sigma2   float64
	Coefs    []CoefSummary
}

// Summary computes coefficient standard errors via numerical Hessian of the
// negative log-likelihood at the fitted parameter vector.
//
// Mirrors statsmodels' summary table content (without the formatting
// chrome). Returns nil and an error if the model is not yet fitted.
func (m *ARIMA) Summary() (*FitSummary, error) {
	if !m.fitted {
		return nil, errors.New("model not fitted")
	}
	p, q := m.Order.P, m.Order.Q
	P, Q := 0, 0
	if m.Seasonal.Active() {
		P, Q = m.Seasonal.P, m.Seasonal.Q
	}
	k := m.nExog

	// Reconstruct the differenced ws and wX (same logic as Fit).
	ws := append([]float64{}, m.yTrain...)
	if m.Order.D > 0 {
		ws = applyDiff(ws, 1, m.Order.D)
	}
	if m.Seasonal.Active() && m.Seasonal.D > 0 {
		ws = applyDiff(ws, m.Seasonal.M, m.Seasonal.D)
	}
	var wX [][]float64
	if m.xTrain != nil {
		wX = cloneMat(m.xTrain)
		if m.Order.D > 0 {
			wX = applyMatDiff(wX, 1, m.Order.D)
		}
		if m.Seasonal.Active() && m.Seasonal.D > 0 {
			wX = applyMatDiff(wX, m.Seasonal.M, m.Seasonal.D)
		}
	}
	for i := range ws {
		ws[i] -= m.mean
	}

	// Re-derive the *untransformed* parameter vector that produces the fitted
	// AR/MA coefficients. We invert tanh on the partial-autocorrelations to
	// get raw parameters back; this is approximate (the transform isn't
	// invertible 1:1 for the full polynomial), so for the Hessian we instead
	// work in the transformed space and treat phi/theta directly as parameters.
	//
	// Strategy: recompute negLL as a function of the *displayed* parameter
	// vector (phi, theta, Phi, Theta, intercept, beta), without transforms.
	// Numerical Hessian → covariance.

	flatParams, names := m.flatNamedParams()
	nParam := len(flatParams)
	if nParam == 0 {
		return &FitSummary{
			Order: m.Order, Seasonal: m.Seasonal, NObs: m.nobs,
			LogLik: m.logL, AIC: m.AIC(), BIC: m.BIC(), AICc: m.AICc(),
			HQIC: m.IC(HQIC), Sigma2: m.sigma2,
		}, nil
	}

	// Pre-compute un-differenced y on the model scale (Box-Cox-applied if
	// applicable). Used by the NonSimpleDifferencing branch of negLL since
	// kalmanARIMAFullConv handles its own differencing internally.
	yUndiff := m.yMSCache
	if yUndiff == nil {
		yUndiff = m.yTrain
	}

	// Direct negLL of (phi, theta, Phi, Theta, c, beta). Dispatches to the
	// SAME likelihood family that Fit used for this model — pre-fix this
	// always called kalmanARMALikelihood, which produced wrong standard
	// errors for NonSimpleDifferencing and MethodCSS fits (the Hessian
	// reflected a different objective than the one that produced the params).
	mPer := 0
	Dord := 0
	if m.Seasonal.Active() {
		mPer = m.Seasonal.M
		Dord = m.Seasonal.D
	}
	negLL := func(theta []float64) float64 {
		idx := 0
		var phi []float64
		if p > 0 {
			phi = append([]float64{}, theta[idx:idx+p]...)
			idx += p
		}
		var maC []float64
		if q > 0 {
			maC = append([]float64{}, theta[idx:idx+q]...)
			idx += q
		}
		var sPhi []float64
		if P > 0 {
			sPhi = append([]float64{}, theta[idx:idx+P]...)
			idx += P
		}
		var sTheta []float64
		if Q > 0 {
			sTheta = append([]float64{}, theta[idx:idx+Q]...)
			idx += Q
		}
		c := 0.0
		if m.WithIntercept {
			c = theta[idx]
			idx++
		}
		var beta []float64
		if k > 0 {
			beta = append([]float64{}, theta[idx:idx+k]...)
		}
		fullPhi := expandSARMA(phi, sPhi, m.Seasonal.M)
		fullTheta := expandSMA(maC, sTheta, m.Seasonal.M)

		var ll float64
		switch {
		case m.NonSimpleDifferencing:
			// Use the integrated state-space form with the model's
			// configured DiffuseConvention. Residual is the un-differenced y
			// minus the regression part (the integrated Kalman handles the
			// differencing internally).
			rUndiff := make([]float64, len(yUndiff))
			for i, v := range yUndiff {
				rr := v - c
				if k > 0 {
					for j := 0; j < k; j++ {
						rr -= beta[j] * m.xTrain[i][j]
					}
				}
				rUndiff[i] = rr
			}
			ll, _, _ = kalmanARIMAFullConv(rUndiff, m.Order.D, mPer, Dord,
				phi, maC, sPhi, sTheta, 1e6, m.DiffuseConvention)
		case m.Method == MethodCSS:
			// CSS profile likelihood on the differenced+centered series.
			r := make([]float64, len(ws))
			for i, v := range ws {
				rr := v - c
				if k > 0 {
					for j := 0; j < k; j++ {
						rr -= beta[j] * wX[i][j]
					}
				}
				r[i] = rr
			}
			ll, _, _ = armaCSS(r, fullPhi, fullTheta)
		default:
			// Simple-differencing Kalman on the differenced+centered series.
			r := make([]float64, len(ws))
			for i, v := range ws {
				rr := v - c
				if k > 0 {
					for j := 0; j < k; j++ {
						rr -= beta[j] * wX[i][j]
					}
				}
				r[i] = rr
			}
			ll, _, _ = kalmanARMALikelihood(r, fullPhi, fullTheta)
		}
		if math.IsNaN(ll) || math.IsInf(ll, 0) {
			return 1e15
		}
		return ll
	}

	hess := numericalHessian(negLL, flatParams, 1e-4)
	cov, err := invertSPD(hess)
	if err != nil {
		// fall back: stderr = NaN
		cov = mat.NewDense(nParam, nParam, nil)
		for i := 0; i < nParam; i++ {
			cov.Set(i, i, math.NaN())
		}
	}
	coefs := make([]CoefSummary, nParam)
	for i := 0; i < nParam; i++ {
		v := cov.At(i, i)
		se := math.NaN()
		if v >= 0 {
			se = math.Sqrt(v)
		}
		z := flatParams[i] / se
		pval := 2 * (1 - normCDF(math.Abs(z)))
		coefs[i] = CoefSummary{
			Name:   names[i],
			Value:  flatParams[i],
			StdErr: se,
			Z:      z,
			PValue: pval,
		}
	}
	return &FitSummary{
		Order: m.Order, Seasonal: m.Seasonal, NObs: m.nobs,
		LogLik: m.logL, AIC: m.AIC(), BIC: m.BIC(), AICc: m.AICc(),
		HQIC: m.IC(HQIC), Sigma2: m.sigma2,
		Coefs: coefs,
	}, nil
}

// flatNamedParams returns the displayed parameter vector and matching names.
func (m *ARIMA) flatNamedParams() ([]float64, []string) {
	var vals []float64
	var names []string
	for i, v := range m.phi {
		vals = append(vals, v)
		names = append(names, fmt.Sprintf("ar.L%d", i+1))
	}
	for i, v := range m.theta {
		vals = append(vals, v)
		names = append(names, fmt.Sprintf("ma.L%d", i+1))
	}
	for i, v := range m.Phi {
		vals = append(vals, v)
		names = append(names, fmt.Sprintf("ar.S.L%d", (i+1)*m.Seasonal.M))
	}
	for i, v := range m.Theta {
		vals = append(vals, v)
		names = append(names, fmt.Sprintf("ma.S.L%d", (i+1)*m.Seasonal.M))
	}
	if m.WithIntercept {
		vals = append(vals, m.c)
		names = append(names, "intercept")
	}
	for i, v := range m.beta {
		vals = append(vals, v)
		names = append(names, fmt.Sprintf("x%d", i+1))
	}
	return vals, names
}

// String renders a multi-line text representation similar to statsmodels.
func (s *FitSummary) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "ARIMA%s", s.Order)
	if s.Seasonal.Active() {
		fmt.Fprintf(&b, "x%s", s.Seasonal)
	}
	fmt.Fprintf(&b, "  n=%d  logL=%.4f  sigma2=%.4f\n", s.NObs, s.LogLik, s.Sigma2)
	fmt.Fprintf(&b, "AIC=%.3f  BIC=%.3f  AICc=%.3f  HQIC=%.3f\n",
		s.AIC, s.BIC, s.AICc, s.HQIC)
	if len(s.Coefs) == 0 {
		return b.String()
	}
	fmt.Fprintf(&b, "%-12s %12s %12s %10s %10s\n",
		"coef", "value", "std err", "z", "P>|z|")
	for _, c := range s.Coefs {
		fmt.Fprintf(&b, "%-12s %12.4f %12.4f %10.3f %10.3f\n",
			c.Name, c.Value, c.StdErr, c.Z, c.PValue)
	}
	return b.String()
}

// numericalHessian computes a central-difference Hessian of f at x.
func numericalHessian(f func([]float64) float64, x []float64, eps float64) *mat.Dense {
	n := len(x)
	h := mat.NewDense(n, n, nil)
	x0 := append([]float64{}, x...)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			xpp := append([]float64{}, x0...)
			xpm := append([]float64{}, x0...)
			xmp := append([]float64{}, x0...)
			xmm := append([]float64{}, x0...)
			xpp[i] += eps
			xpp[j] += eps
			xpm[i] += eps
			xpm[j] -= eps
			xmp[i] -= eps
			xmp[j] += eps
			xmm[i] -= eps
			xmm[j] -= eps
			val := (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps * eps)
			h.Set(i, j, val)
			h.Set(j, i, val)
		}
	}
	return h
}

// invertSPD inverts a symmetric matrix; returns error if singular.
func invertSPD(h *mat.Dense) (*mat.Dense, error) {
	r, _ := h.Dims()
	out := mat.NewDense(r, r, nil)
	if err := out.Inverse(h); err != nil {
		return nil, err
	}
	return out, nil
}

// normCDF returns the standard-normal CDF.
func normCDF(z float64) float64 {
	return 0.5 * (1 + math.Erf(z/math.Sqrt2))
}
