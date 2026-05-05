package arima

import (
	"errors"
	"fmt"
	"math"
)

// OutlierType enumerates the supported outlier intervention types.
// goarima implements the four types from R's tsoutliers vocabulary:
//
//   - AO (Additive Outlier): one-off shock at a single timestep.
//   - LS (Level Shift): permanent step from a timestep onward.
//   - TC (Temporary Change): exponentially-decaying impulse with rate
//     δ ∈ (0, 1). Models a shock that fades over time (post-news effect,
//     post-policy decay).
//   - IO (Innovational Outlier): perturbation on the innovation rather
//     than the observation; propagates through the ARMA filter as a
//     ψ-weighted decay. Useful for cascading shocks.
type OutlierType int

const (
	OutlierAO OutlierType = iota // Additive Outlier (impulse at one timestep)
	OutlierLS                    // Level Shift  (step from a timestep onward)
	OutlierTC                    // Temporary Change (geometrically-decaying impulse)
	OutlierIO                    // Innovational Outlier (perturbation on the innovation)
)

// String returns "AO", "LS", "TC", or "IO".
func (t OutlierType) String() string {
	switch t {
	case OutlierAO:
		return "AO"
	case OutlierLS:
		return "LS"
	case OutlierTC:
		return "TC"
	case OutlierIO:
		return "IO"
	}
	return fmt.Sprintf("OutlierType(%d)", int(t))
}

// Outlier describes a single detected outlier.
type Outlier struct {
	Index int         // 0-based time index in the original y
	Type  OutlierType // AO or LS
	Coef  float64     // estimated outlier magnitude (the ω)
	Tau   float64     // |t-stat| at the round it was detected
}

// DetectOutliersOpts configures DetectOutliers.
type DetectOutliersOpts struct {
	// Order / Seasonal are the ARIMA(p,d,q)(P,D,Q,M) shape used to model
	// the clean-series dynamics. Default: ARIMA(0,1,1) — random walk + MA1
	// — the standard "approximation" model used by R's tsoutliers when
	// the true ARIMA shape is unknown.
	Order    Order
	Seasonal SeasonalOrder

	// Types lists which outlier types to test. Default: {AO, LS}.
	Types []OutlierType

	// CritVal is the |t-stat| threshold for declaring an outlier.
	// Default 3.5 for n<200, 4.0 otherwise (matches tsoutliers's defaults
	// for moderate vs. long series).
	CritVal float64

	// MaxIter caps the number of detect-then-refit rounds (each round
	// detects at most one new outlier). Default 5.
	MaxIter int

	// MaxFit caps the optimizer iterations per ARIMA refit. Default 50.
	MaxFit int

	// Method, Lambda, Lambda2 forward to the underlying ARIMA fits.
	Method  Method
	Lambda  *float64
	Lambda2 float64

	// TCDecayRate is the geometric decay rate δ ∈ (0, 1) used when
	// scoring/building TC (Temporary Change) candidates. TC's
	// regressor at time τ is δ^(t-τ) for t ≥ τ. Lower δ → faster
	// decay; higher δ → slower fade. Default 0.7 (matches R's
	// tsoutliers default).
	//
	// Only used when OutlierTC appears in Types.
	TCDecayRate float64
}

// DetectOutliers identifies AO, LS, TC, and IO outliers using a
// simplified Chen-Liu (1993) iterative search. The Types option
// selects which subset to scan — default is {AO, LS} for backward
// compatibility; opt in to {AO, LS, TC, IO} for full coverage.
//
//  1. Fit a base ARIMA on y.
//  2. Compute π-weights of the inverse ARMA filter (including the
//     differencing operator). These convert a hypothetical original-scale
//     outlier into its expected residual signature.
//  3. For each candidate (timestep, type), score the t-statistic of the
//     residual projection onto that signature.
//  4. If the largest |t-stat| exceeds CritVal, treat it as the next
//     outlier, build the corresponding regressor (impulse for AO,
//     step for LS), append it to the exog matrix, and refit.
//  5. Repeat until no candidate clears CritVal or MaxIter is reached.
//
// Returns the cumulative outlier list plus the final fitted model
// (whose Beta() contains the outlier coefficients aligned with the
// returned slice; entries appear in detection order).
//
// Useful when y has obvious one-off shocks (a holiday closure, a strike,
// a COVID drop) or regime changes, where forecasting the clean dynamics
// requires masking out the contamination first. R analogue:
// tsoutliers::tso(y).
func DetectOutliers(y []float64, opts DetectOutliersOpts) ([]Outlier, *ARIMA, error) {
	n := len(y)
	if n < 10 {
		return nil, nil, errors.New("series too short for outlier detection")
	}
	if opts.MaxIter == 0 {
		opts.MaxIter = 5
	}
	if opts.MaxFit == 0 {
		opts.MaxFit = 50
	}
	if opts.CritVal == 0 {
		if n < 200 {
			opts.CritVal = 3.5
		} else {
			opts.CritVal = 4.0
		}
	}
	if len(opts.Types) == 0 {
		// Default kept at {AO, LS} for backward compatibility with
		// pre-OUT-TC-IO callers. TC/IO must be explicitly opted in.
		opts.Types = []OutlierType{OutlierAO, OutlierLS}
	}
	if opts.TCDecayRate <= 0 || opts.TCDecayRate >= 1 {
		opts.TCDecayRate = 0.7
	}
	if opts.Order == (Order{}) && opts.Seasonal == (SeasonalOrder{}) {
		opts.Order = Order{D: 1, Q: 1}
	}

	fit := func(reg [][]float64) (*ARIMA, error) {
		m := NewARIMA(opts.Order)
		m.Seasonal = opts.Seasonal
		m.MaxIter = opts.MaxFit
		m.Method = opts.Method
		if opts.Lambda != nil {
			lam := *opts.Lambda
			m.Lambda = &lam
			m.Lambda2 = opts.Lambda2
		}
		if err := m.Fit(y, reg); err != nil {
			return nil, err
		}
		return m, nil
	}

	cur, err := fit(nil)
	if err != nil {
		return nil, nil, fmt.Errorf("base fit failed: %w", err)
	}

	dTotal := opts.Order.D + opts.Seasonal.D*opts.Seasonal.M

	var outliers []Outlier
	var reg [][]float64

	for iter := 0; iter < opts.MaxIter; iter++ {
		pi := computePiWeights(cur, n)
		sigma := math.Sqrt(cur.Sigma2())
		if sigma <= 0 || math.IsNaN(sigma) || math.IsInf(sigma, 0) {
			break
		}

		eps := cur.resids
		if len(eps) == 0 {
			break
		}
		// Map original-time t → differenced-residual index t - dTotal.
		// Residuals only exist for t >= dTotal.

		seen := map[[2]int]bool{}
		for _, o := range outliers {
			seen[[2]int{o.Index, int(o.Type)}] = true
		}

		bestTau := opts.CritVal
		bestT := -1
		var bestType OutlierType
		var bestCoef float64

		// Candidate window: leave a small buffer at the right edge for LS
		// estimability. AO can be tested anywhere from dTotal to n-1.
		minTau := dTotal
		maxTauAO := n
		maxTauLS := n - 2
		if maxTauLS < minTau+1 {
			maxTauLS = minTau + 1
		}

		for _, typ := range opts.Types {
			maxTau := maxTauAO
			if typ == OutlierLS {
				maxTau = maxTauLS
			}
			for tau := minTau; tau < maxTau; tau++ {
				if seen[[2]int{tau, int(typ)}] {
					continue
				}
				// Build signature on the differenced timeline and project.
				// AO:  sig(t,τ) = π_{t-τ}                         for t ≥ τ
				// LS:  sig(t,τ) = Σ_{k=0..t-τ} π_k                 (cumulative)
				// TC:  sig(t,τ) = Σ_{k=0..t-τ} π_k · δ^(t-τ-k)     (geometric ⊗ π)
				// IO:  sig(t,τ) = 1 if t == τ, else 0              (residual impulse)
				num := 0.0
				den := 0.0
				lsCum := 0.0
				tcCum := 0.0
				for t := tau; t < n; t++ {
					k := t - tau
					var sig float64
					switch typ {
					case OutlierAO:
						if k < len(pi) {
							sig = pi[k]
						}
					case OutlierLS:
						if k < len(pi) {
							lsCum += pi[k]
						}
						sig = lsCum
					case OutlierTC:
						pk := 0.0
						if k < len(pi) {
							pk = pi[k]
						}
						tcCum = opts.TCDecayRate*tcCum + pk
						sig = tcCum
					case OutlierIO:
						if k == 0 {
							sig = 1
						} else {
							sig = 0
						}
					}
					if t < dTotal {
						continue
					}
					residIdx := t - dTotal
					if residIdx >= len(eps) {
						break
					}
					num += sig * eps[residIdx]
					den += sig * sig
				}
				if den <= 0 {
					continue
				}
				coef := num / den
				tStat := coef * math.Sqrt(den) / sigma
				if math.Abs(tStat) > bestTau {
					bestTau = math.Abs(tStat)
					bestT = tau
					bestType = typ
					bestCoef = coef
				}
			}
		}

		if bestT < 0 {
			break
		}

		// IO regressor uses the CURRENT model's MA(∞) ψ-weights (the
		// shape of the innovation impulse response). Use cur.psi at
		// the moment of detection — subsequent refits may shift psi
		// slightly but the regressor is fixed once added (matches
		// tsoutliers' fast-mode behaviour). For non-IO types ψ is unused.
		var psiSnapshot []float64
		if bestType == OutlierIO {
			if pc := cur.psi.Load(); pc != nil {
				psiSnapshot = pc.values
			}
		}
		col := buildOutlierRegressor(n, bestT, bestType, opts.TCDecayRate, psiSnapshot)
		reg = appendOutlierCol(reg, col)
		next, ferr := fit(reg)
		if ferr != nil {
			// Refit failed; drop this candidate and stop. The local `reg`
			// is then garbage-collected with the function frame.
			break
		}
		cur = next
		outliers = append(outliers, Outlier{
			Index: bestT,
			Type:  bestType,
			Coef:  bestCoef,
			Tau:   bestTau,
		})
	}

	// Refresh outlier coefs from the final model's β (more reliable than the
	// projection coef, which used pre-refit residuals).
	if betas := cur.Beta(); len(betas) >= len(outliers) {
		offset := len(betas) - len(outliers)
		for i := range outliers {
			outliers[i].Coef = betas[offset+i]
		}
	}

	return outliers, cur, nil
}

// buildOutlierRegressor builds the original-y-space exog column for an
// outlier intervention at time t. delta is the TC decay rate (used only
// for OutlierTC; ignored otherwise). psi is the ψ MA(∞) coefficient
// vector of the current ARIMA model (used only for OutlierIO; ignored
// otherwise).
//
//   - AO: impulse — col[t] = 1, 0 elsewhere
//   - LS: step — col[i] = 1 for i ≥ t
//   - TC: geometric decay — col[i] = δ^(i-t) for i ≥ t
//   - IO: ψ-weighted impulse — col[i] = ψ_(i-t) for i ≥ t (the model's
//     MA(∞) impulse response)
func buildOutlierRegressor(n, t int, typ OutlierType, delta float64, psi []float64) []float64 {
	col := make([]float64, n)
	switch typ {
	case OutlierAO:
		if t >= 0 && t < n {
			col[t] = 1
		}
	case OutlierLS:
		for i := t; i < n; i++ {
			col[i] = 1
		}
	case OutlierTC:
		// δ^k recursion: col[t]=1, col[t+1]=δ, col[t+2]=δ², …
		v := 1.0
		for i := t; i < n; i++ {
			col[i] = v
			v *= delta
		}
	case OutlierIO:
		// ψ-weighted impulse response (model's MA(∞)). For lags past
		// len(psi) we leave the regressor at 0 — the truncation matches
		// the model's psi-cache truncation.
		for i := t; i < n; i++ {
			k := i - t
			if k < len(psi) {
				col[i] = psi[k]
			}
		}
	}
	return col
}

func appendOutlierCol(X [][]float64, col []float64) [][]float64 {
	if len(X) == 0 {
		out := make([][]float64, len(col))
		for i, v := range col {
			out[i] = []float64{v}
		}
		return out
	}
	for i := range X {
		X[i] = append(X[i], col[i])
	}
	return X
}

// computePiWeights returns the first n coefficients of π(L) = A(L)/B(L)
// where:
//
//	A(L) = φ(L) · Φ(L^M) · (1-L)^d · (1-L^M)^D
//	B(L) = θ(L) · Θ(L^M)
//
// using the model's storage convention: m.phi[i] holds +φ_{i+1} (so the
// AR polynomial coefficient at L^{i+1} is -m.phi[i]); m.theta[j] holds
// +θ_{j+1} (already with the polynomial sign).
//
// π_k is recovered from A(L) = π(L)·B(L) by matching coefficients:
// A_k = π_k + Σ_{j≥1} B_j π_{k-j}, i.e., π_k = A_k - Σ_{j≥1} B_j π_{k-j}
// (with π_0 = 1 since A_0 = B_0 = 1).
func computePiWeights(m *ARIMA, n int) []float64 {
	A := []float64{1}
	if len(m.phi) > 0 {
		coef := make([]float64, len(m.phi)+1)
		coef[0] = 1
		for i, p := range m.phi {
			coef[i+1] = -p
		}
		A = polyMul(A, coef)
	}
	if len(m.Phi) > 0 && m.Seasonal.M > 1 {
		coef := make([]float64, m.Seasonal.M*len(m.Phi)+1)
		coef[0] = 1
		for k, P := range m.Phi {
			coef[(k+1)*m.Seasonal.M] = -P
		}
		A = polyMul(A, coef)
	}
	for k := 0; k < m.Order.D; k++ {
		A = polyMul(A, []float64{1, -1})
	}
	if m.Seasonal.M > 1 {
		for k := 0; k < m.Seasonal.D; k++ {
			coef := make([]float64, m.Seasonal.M+1)
			coef[0] = 1
			coef[m.Seasonal.M] = -1
			A = polyMul(A, coef)
		}
	}

	B := []float64{1}
	if len(m.theta) > 0 {
		coef := make([]float64, len(m.theta)+1)
		coef[0] = 1
		for j, q := range m.theta {
			coef[j+1] = q
		}
		B = polyMul(B, coef)
	}
	if len(m.Theta) > 0 && m.Seasonal.M > 1 {
		coef := make([]float64, m.Seasonal.M*len(m.Theta)+1)
		coef[0] = 1
		for k, Q := range m.Theta {
			coef[(k+1)*m.Seasonal.M] = Q
		}
		B = polyMul(B, coef)
	}

	pi := make([]float64, n)
	pi[0] = 1
	for k := 1; k < n; k++ {
		var ak float64
		if k < len(A) {
			ak = A[k]
		}
		s := ak
		for j := 1; j < len(B) && j <= k; j++ {
			s -= B[j] * pi[k-j]
		}
		pi[k] = s
	}
	return pi
}

