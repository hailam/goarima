package arima

import "math"

// arTransparams maps unconstrained reals to AR coefficients enforcing
// stationarity, via Jones (1980) partial-autocorrelation parameterization.
//
// This mirrors statsmodels.tsa.arima_model._ar_transparams.
func arTransparams(params []float64) []float64 {
	n := len(params)
	if n == 0 {
		return []float64{}
	}
	newp := make([]float64, n)
	tmp := make([]float64, n)
	for i, p := range params {
		newp[i] = math.Tanh(p)
		tmp[i] = newp[i]
	}
	for j := 1; j < n; j++ {
		a := newp[j]
		for k := 0; k < j; k++ {
			tmp[k] = newp[k] - a*newp[j-k-1]
		}
		for k := 0; k < j; k++ {
			newp[k] = tmp[k]
		}
	}
	return newp
}

// maTransparams is identical structure for MA invertibility (sign symmetric).
func maTransparams(params []float64) []float64 {
	n := len(params)
	if n == 0 {
		return []float64{}
	}
	newp := make([]float64, n)
	tmp := make([]float64, n)
	for i, p := range params {
		newp[i] = math.Tanh(p)
		tmp[i] = newp[i]
	}
	for j := 1; j < n; j++ {
		a := newp[j]
		for k := 0; k < j; k++ {
			tmp[k] = newp[k] + a*newp[j-k-1]
		}
		for k := 0; k < j; k++ {
			newp[k] = tmp[k]
		}
	}
	return newp
}
