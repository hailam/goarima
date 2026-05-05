package arima

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// BootstrapInferenceResult holds parameter-level inference statistics
// from a parametric bootstrap. Field ordering aligns with m.Params():
// [phi..., theta..., Phi..., Theta..., intercept (if), beta...].
type BootstrapInferenceResult struct {
	// Params is the bootstrap mean of each parameter — informational
	// (equals the original estimate plus bias). Differs from
	// m.Params() under finite-sample bias.
	Params []float64

	// StdErr is the empirical bootstrap standard error per parameter
	// (sqrt of sample variance, divisor B−1). Use this for Wald-style
	// CIs that don't rely on the asymptotic-normality assumption that
	// Hessian-based SEs depend on.
	StdErr []float64

	// Lower / Upper are percentile-method CI bounds at the requested
	// alpha — Lower is the alpha/2 quantile of bootstrap samples,
	// Upper the 1-alpha/2 quantile.
	Lower []float64
	Upper []float64

	// Samples is the B × P raw bootstrap parameter matrix (B rows,
	// each a parameter vector). Provided for users who want to compute
	// custom statistics (e.g. correlations between parameters,
	// bias-corrected and accelerated CIs, joint confidence regions).
	// Setting BootstrapInferenceOpts.OmitSamples avoids this allocation.
	Samples [][]float64
}

// BootstrapInferenceOpts configures BootstrapInference.
type BootstrapInferenceOpts struct {
	// B is the number of bootstrap simulations. Required (≥10).
	// Typical: 200-1000. Higher = tighter quantile estimates,
	// proportionally more cost.
	B int

	// Alpha is the two-sided coverage error rate for Lower/Upper.
	// Required (in (0, 1)). 0.05 → 95% CI.
	Alpha float64

	// Seed pins the per-bootstrap RNG. 0 → deterministic from a
	// fixed default seed (so callers who don't set Seed still get
	// reproducible results across runs, just identical across
	// invocations).
	Seed uint64

	// OmitSamples skips populating the BootstrapInferenceResult.Samples
	// field — saves a B×P-float allocation when callers only need
	// summary statistics (mean / SE / CI bounds).
	OmitSamples bool
}

// BootstrapInference runs a parametric bootstrap to estimate per-parameter
// standard errors and confidence intervals. Useful on small n where the
// Hessian-based asymptotic SEs (Summary's StdErr column) over-cover or
// under-cover because the asymptotic-normality assumption is wrong.
//
// Algorithm:
//
//   1. For b = 1..B: simulate from the fitted model, re-fit a fresh
//      ARIMA of the same shape, record the parameter vector θ_b.
//   2. Empirical SE per parameter: sqrt of sample variance across b.
//   3. Percentile-method CI per parameter: empirical α/2 and 1-α/2
//      quantiles of the bootstrap distribution.
//
// Cost: B fresh Fits. With our post-G-NEW-2 / KAL-WORKSPACE / BURG-1
// optimizations (~2.7 ms per default Fit), B=200 is ~540 ms wallclock.
//
// Differentiation vs R/pmdarima: neither ships bootstrap inference
// built-in (R users install `tsbootstrap` separately). This is a
// gold-standard small-n inference tool that's now a one-method call.
//
// Closes the small-dataset-perf-roadmap "Bootstrap parameter CIs" item.
func (m *ARIMA) BootstrapInference(opts BootstrapInferenceOpts) (*BootstrapInferenceResult, error) {
	if !m.fitted {
		return nil, errors.New("arima: model not fitted; call Fit before BootstrapInference")
	}
	if opts.B < 10 {
		return nil, errors.New("arima: B must be ≥ 10 (typical: 200-1000)")
	}
	if opts.Alpha <= 0 || opts.Alpha >= 1 {
		return nil, errors.New("arima: Alpha must be in (0, 1)")
	}
	n := len(m.yTrain)
	if n == 0 {
		return nil, errors.New("arima: yTrain is empty")
	}

	// Same exog as training so each bootstrap fit reproduces the
	// regressor structure faithfully.
	var simExog [][]float64
	if m.nExog > 0 {
		simExog = m.xTrain
	}
	fitExog := simExog

	// Probe parameter dimension from the original fit.
	origParams := m.Params()
	P := len(origParams)
	if P == 0 {
		return nil, errors.New("arima: model has no parameters to bootstrap")
	}

	samples := make([][]float64, 0, opts.B)
	for b := 0; b < opts.B; b++ {
		simY, err := m.Simulate(n, SimulateOpts{
			Seed:       opts.Seed + uint64(b) + 1,
			FutureExog: simExog,
		})
		if err != nil {
			continue
		}
		candidate := NewARIMA(m.Order)
		candidate.Seasonal = m.Seasonal
		candidate.WithIntercept = m.WithIntercept
		candidate.Method = m.Method
		candidate.MaxIter = m.MaxIter
		candidate.NonSimpleDifferencing = m.NonSimpleDifferencing
		candidate.DiffuseConvention = m.DiffuseConvention
		if m.Lambda != nil {
			lam := *m.Lambda
			candidate.Lambda = &lam
			candidate.Lambda2 = m.Lambda2
		}
		candidate.RidgePenalty = m.RidgePenalty
		if err := candidate.Fit(simY, fitExog); err != nil {
			continue
		}
		row := candidate.Params()
		if len(row) != P {
			// Shouldn't happen — same shape — but guard anyway.
			continue
		}
		samples = append(samples, row)
	}
	if len(samples) < opts.B/2 {
		return nil, fmt.Errorf("arima: only %d/%d bootstrap fits succeeded; inference unreliable",
			len(samples), opts.B)
	}

	res := &BootstrapInferenceResult{
		Params: make([]float64, P),
		StdErr: make([]float64, P),
		Lower:  make([]float64, P),
		Upper:  make([]float64, P),
	}

	// Compute mean per parameter.
	nf := float64(len(samples))
	for j := 0; j < P; j++ {
		s := 0.0
		for _, row := range samples {
			s += row[j]
		}
		res.Params[j] = s / nf
	}
	// Sample variance (divisor n-1) → empirical SE.
	for j := 0; j < P; j++ {
		ssq := 0.0
		for _, row := range samples {
			d := row[j] - res.Params[j]
			ssq += d * d
		}
		if len(samples) > 1 {
			res.StdErr[j] = math.Sqrt(ssq / float64(len(samples)-1))
		}
	}
	// Percentile-method bounds. Reuse a column buffer per parameter.
	col := make([]float64, len(samples))
	loQ := opts.Alpha / 2
	hiQ := 1 - opts.Alpha/2
	for j := 0; j < P; j++ {
		for i, row := range samples {
			col[i] = row[j]
		}
		sort.Float64s(col)
		res.Lower[j] = quantile(col, loQ)
		res.Upper[j] = quantile(col, hiQ)
	}

	if !opts.OmitSamples {
		res.Samples = samples
	}
	return res, nil
}

// BootstrapStdErrors is a thin convenience wrapper: returns just the
// per-parameter empirical SEs, aligned with m.Params(). Use
// BootstrapInference for full CI / quantile / sample access.
func (m *ARIMA) BootstrapStdErrors(B int, seed uint64) ([]float64, error) {
	res, err := m.BootstrapInference(BootstrapInferenceOpts{
		B:           B,
		Alpha:       0.05, // unused for SE, but Alpha is required
		Seed:        seed,
		OmitSamples: true,
	})
	if err != nil {
		return nil, err
	}
	return res.StdErr, nil
}
