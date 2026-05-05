package arima

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"

	"github.com/hailam/goarima/metrics"
)

// AutoArimaOpts configures the auto_arima search.
type AutoArimaOpts struct {
	// Seasonal period; if 0 or 1, only non-seasonal models are considered.
	M int

	// AutoM, when true, overrides M with FindFrequency(y) — an AR-spectral
	// peak-detector matching R's forecast::findfrequency(). Useful for
	// daily/hourly data where the seasonal period isn't obvious. If the
	// detector finds no clear peak it returns 1 (non-seasonal). Mirrors
	// the behavior R users get from `auto.arima(y)` when frequency(y)==1.
	AutoM bool

	// Optional fixed differencing terms; -1 means "estimate".
	D     int
	Dd    int // seasonal D
	HasD  bool
	HasDd bool

	// Search ranges.
	StartP, MaxP       int
	StartQ, MaxQ       int
	StartCapP, MaxCapP int // seasonal P
	StartCapQ, MaxCapQ int // seasonal Q

	MaxOrder int // p+q+P+Q upper bound (default 5)
	MaxD     int // upper bound for non-seasonal d (default 2)
	MaxCapD  int // upper bound for seasonal D (default 1)

	// Stationary, when true, restricts the search to (p, 0, q)(P, 0, Q)
	// — d = D = 0 — and skips the unit-root tests entirely. Useful when
	// you want to model the level series rather than its differences,
	// e.g. when y is already stationary by construction (residuals from
	// another model, demeaned arrival counts, etc.). Mirrors R's
	// `auto.arima(..., stationary=TRUE)`.
	//
	// When false (default), the existing KPSS / OCSB / CH tests pick d
	// and D as before. Closes GAP-3.
	Stationary bool

	Alpha float64    // alpha for diff tests
	Test  NDiffsTest // unit-root test (default KPSS)

	// SeasonalTest selects the seasonal-differencing test. Defaults to
	// NSDiffsOCSB (zero value), matching pmdarima.auto_arima and R's
	// forecast::auto.arima. Set to NSDiffsCH for the legacy Canova-Hansen
	// test (older R / hand-rolled scripts).
	//
	// Pre-fix this field didn't exist and the search hard-coded CH, which
	// underpicks D on short seasonal series where the two tests disagree.
	SeasonalTest NSDiffsTest

	WithIntercept *bool // explicit override; nil = auto

	// AllowDrift overrides WithIntercept's behaviour when d > 0 (i.e. when
	// the model has a non-seasonal differencing operator). nil means
	// "fall through to WithIntercept logic". Set to floatPtr(true) /
	// floatPtr(false) to force drift on/off independently of the d=0
	// mean-term decision.
	//
	// AllowMean is the symmetric override for the d = 0 case.
	//
	// Mirrors R's `auto.arima(..., allowdrift=TRUE, allowmean=TRUE)`
	// which lets users control the two cases independently. goarima
	// previously collapsed both into WithIntercept; these knobs let
	// users separate them. Closes GAP-4. Helper `BoolPtr(v)` provided
	// to make the call site readable.
	AllowDrift *bool
	AllowMean  *bool

	// Information criterion to minimize.
	IC InfoCriterion

	// Maximum optimizer iterations per fit.
	MaxIter int

	// FullSearch enumerates every (p,q,P,Q) combination within the search box
	// and picks the lowest-IC (or holdout-score) fit. Slower than the default
	// stepwise mode but more thorough. Equivalent to pmdarima's stepwise=False.
	FullSearch bool

	// NFits caps the number of candidates explored in non-stepwise mode.
	// 0 means "no cap" (full enumeration). When >0 and < total combinations,
	// candidates are sampled uniformly without replacement (mirrors random=True).
	NFits int

	// Seed for the random sampler when NFits > 0 and full enumeration is
	// truncated. Default 0 (deterministic).
	Seed uint64

	// OutOfSampleSize holds out the last K observations and scores candidates
	// on them instead of by IC. 0 disables.
	OutOfSampleSize int

	// Scoring is the holdout-set scorer (when OutOfSampleSize > 0). nil → SMAPE.
	// MAY BE CALLED CONCURRENTLY from multiple goroutines (stepwise neighbor
	// fits + FullSearch worker pool). Any state captured in the closure must
	// be safe for concurrent access.
	Scoring func(yTrue, yPred []float64) (float64, error)

	// Trace receives one line per fitted candidate when non-nil. Mirrors
	// pmdarima's trace=True (the printed lines are emitted programmatically).
	//
	// In **stepwise mode** (the default) Trace is called only from the main
	// goroutine in stable neighbor order, so callbacks do NOT need to be
	// thread-safe and trace output is deterministic regardless of how the
	// parallel candidate fits scheduled.
	//
	// In **FullSearch mode** Trace is still called from the result-collector
	// goroutine on a single thread, so callbacks don't need to be
	// thread-safe there either; only the order in which lines arrive is
	// non-deterministic (worker arrival order).
	Trace func(string)

	// ErrorAction controls behavior when a candidate fit errors:
	//   "raise" (default): the first error aborts the whole search
	//   "ignore"          : skip the failing candidate, continue
	//   "warn"            : like "ignore" but call Trace if present
	ErrorAction string

	// MaxSteps caps the number of stepwise iterations (0 → 50).
	MaxSteps int

	// ParsimonyDelta requires a candidate with MORE parameters than the
	// current best to beat its IC by at least this much before being
	// adopted. Same-or-fewer-parameter candidates use the existing 1e-6
	// tolerance, so down-shifting to a simpler model is unaffected.
	//
	// Default 0.0 disables the gate (legacy behavior — any IC drop wins).
	// A common setting is 2.0, the conventional ΔAICc threshold for
	// "meaningful" improvement (Burnham & Anderson). R's auto.arima uses
	// a similar mechanism via its `ic` tolerance to keep the stepwise
	// search from drifting onto over-parameterized models on flat IC
	// landscapes; goarima now exposes the same lever explicitly.
	ParsimonyDelta float64

	// NJobs sets goroutine concurrency for both stepwise (4–8 neighbor fits
	// per iteration) and FullSearch (whole search box) modes. 0 → GOMAXPROCS.
	// Mirrors pmdarima's `n_jobs`. R/Python require pickling+IPC for parallel
	// auto.arima, but Go's goroutines are essentially free.
	NJobs int

	// Method selects the per-candidate fitting estimator (CSS / ML / CSS-ML).
	// Default 0 (MethodCSSML) — same as a hand-rolled `Fit`. Mirrors
	// pmdarima's `method=` and R's `method=` arguments to auto.arima.
	Method Method

	// Approximation, when true, runs the candidate search with MethodCSS
	// (fast, biased likelihood) regardless of the Method field, then
	// refits the picked (Order, Seasonal) once with the user's Method
	// (default MethodCSSML). Mirrors R's `auto.arima(..., approximation=TRUE)`.
	// Empirically 5–6× faster on the search phase with negligible loss
	// in model quality — R uses this by default when n>150 or m>12.
	//
	// All other opts (Lambda, NonSimpleDifferencing, DiffuseConvention,
	// WithIntercept, exog) propagate to both phases. The final fit's
	// logL / σ² / AICc are reported on the Method scale, not CSS.
	Approximation bool

	// RidgePenalty propagates to every candidate fit's `m.RidgePenalty`.
	// See ARIMA.RidgePenalty doc; default 0.0 = off (R-parity safe).
	// Useful on short series (n ≤ 50) where AutoArima's stepwise search
	// otherwise picks degenerate boundary models. Suggested starting
	// value: `1.0 / len(y)`.
	RidgePenalty float64

	// Lambda (and Lambda2 shift) enable Box-Cox transformation on every
	// candidate fit. nil → no transform. Use `floatPtr(0)` for log. Mirrors
	// pmdarima's `lambda=` arg. Each candidate's AICc / forecast is
	// computed on the back-transformed series so values are comparable
	// across the search box.
	Lambda  *float64
	Lambda2 float64

	// NonSimpleDifferencing routes every candidate through the integrated
	// state-space form (R's `stats::arima` / statsmodels SARIMAX path).
	// Slower than the default simple-differencing path but required when
	// you need 1:1 R or statsmodels parity within the search.
	NonSimpleDifferencing bool

	// DiffuseConvention selects the likelihood-convention flag passed to
	// each candidate. Only effective when NonSimpleDifferencing is true.
	// DiffuseR (default) matches R's stats::arima; DiffuseStatsmodels
	// matches SARIMAX(simple_differencing=False).
	DiffuseConvention DiffuseConv
}

// BoolPtr returns a pointer to the given bool. Convenience helper for
// AutoArimaOpts fields that need explicit override (AllowDrift, AllowMean,
// WithIntercept) — distinguishes "nil = use default" from "false = force off".
func BoolPtr(v bool) *bool { return &v }

// AutoArima runs model selection over ARIMA(p,d,q)(P,D,Q,m).
// exog is optional (nil for none); if provided, every fitted candidate
// includes the same regressors and the chosen IC compares like-for-like.
//
// Mirrors pmdarima.auto_arima: chooses d/D via unit-root tests, then
// either explores neighbors stepwise (default) or enumerates the search box.
func AutoArima(y []float64, exog [][]float64, opts AutoArimaOpts) (*ARIMA, error) {
	if len(y) < 10 {
		return nil, fmt.Errorf("series too short: %d", len(y))
	}
	if opts.AutoM {
		opts.M = FindFrequency(y)
	}
	// GAP-2: Approximation mode runs the candidate search with MethodCSS
	// (fast, biased) and then refits the picked order with the user's
	// Method (default MethodCSSML). This mirrors R's `auto.arima(...,
	// approximation=TRUE)`. We capture the user's "real" method here,
	// override to MethodCSS for the search, then restore-and-refit at
	// the end of this function. The implementation lives in autoArimaCore;
	// this wrapper handles the two-stage orchestration.
	if opts.Approximation {
		finalMethod := opts.Method
		searchOpts := opts
		searchOpts.Method = MethodCSS
		searchOpts.Approximation = false // prevent recursive Approximation
		mSearch, err := AutoArima(y, exog, searchOpts)
		if err != nil {
			return nil, err
		}
		// Refit the picked order with the user's actual Method, preserving
		// every other configuration knob from the search.
		m := NewARIMA(mSearch.Order)
		m.Seasonal = mSearch.Seasonal
		m.Method = finalMethod
		m.MaxIter = opts.MaxIter
		if m.MaxIter == 0 {
			m.MaxIter = 100
		}
		m.WithIntercept = mSearch.WithIntercept
		m.Lambda = opts.Lambda
		m.Lambda2 = opts.Lambda2
		m.NonSimpleDifferencing = opts.NonSimpleDifferencing
		m.DiffuseConvention = opts.DiffuseConvention
		m.GradientWorkers = mSearch.GradientWorkers
		m.RidgePenalty = opts.RidgePenalty
		if err := m.Fit(y, exog); err != nil {
			return nil, fmt.Errorf("approximation refit failed: %w", err)
		}
		return m, nil
	}
	// Defaults
	if opts.MaxOrder == 0 {
		opts.MaxOrder = 5
	}
	if opts.MaxD == 0 {
		opts.MaxD = 2
	}
	if opts.MaxCapD == 0 {
		opts.MaxCapD = 1
	}
	// GAP-3: Stationary forces d = D = 0 — applied AFTER the default-fill
	// block so it overrides MaxD=2 / MaxCapD=1. We also pin manual
	// D=0/Dd=0 so the KPSS / OCSB / CH unit-root tests don't run.
	if opts.Stationary {
		opts.MaxD = 0
		opts.MaxCapD = 0
		opts.D = 0
		opts.HasD = true
		opts.Dd = 0
		opts.HasDd = true
	}
	if opts.Alpha == 0 {
		opts.Alpha = 0.05
	}
	if opts.MaxP == 0 {
		opts.MaxP = 5
	}
	if opts.MaxQ == 0 {
		opts.MaxQ = 5
	}
	if opts.MaxCapP == 0 {
		opts.MaxCapP = 2
	}
	if opts.MaxCapQ == 0 {
		opts.MaxCapQ = 2
	}
	if opts.MaxIter == 0 {
		opts.MaxIter = 100
	}
	if opts.MaxSteps == 0 {
		opts.MaxSteps = 50
	}
	if opts.ErrorAction == "" {
		opts.ErrorAction = "raise"
	}
	switch opts.ErrorAction {
	case "raise", "ignore", "warn":
	default:
		return nil, fmt.Errorf("invalid error_action %q", opts.ErrorAction)
	}

	// Validate exog up front: row count must match y, and rows must be
	// non-ragged (mirrors Fit's checks). Pre-fix this only happened inside
	// the per-candidate Fit call, which meant the holdout slice below
	// (`exog[:split]`) panicked with an opaque slice-out-of-range when
	// exog was shorter than y.
	if exog != nil {
		if len(exog) != len(y) {
			return nil, fmt.Errorf("exog rows (%d) must match len(y) (%d)", len(exog), len(y))
		}
		if len(exog[0]) == 0 {
			return nil, errors.New("exog has zero columns; pass nil for no regressors")
		}
		k0 := len(exog[0])
		for i, row := range exog {
			if len(row) != k0 {
				return nil, fmt.Errorf("exog row %d has %d cols, want %d (ragged exog)", i, len(row), k0)
			}
		}
	}

	// Optional out-of-sample holdout.
	yFit := y
	var yHoldout []float64
	xFit := exog
	var xHoldout [][]float64
	if opts.OutOfSampleSize > 0 {
		if opts.OutOfSampleSize >= len(y) {
			return nil, fmt.Errorf("out_of_sample_size (%d) >= len(y) (%d)", opts.OutOfSampleSize, len(y))
		}
		split := len(y) - opts.OutOfSampleSize
		yFit = y[:split]
		yHoldout = y[split:]
		if exog != nil {
			xFit = exog[:split]
			xHoldout = exog[split:]
		}
	}

	scoring := opts.Scoring
	if scoring == nil {
		scoring = metrics.SMAPE
	}

	// Pre-regress exog out of y before running differencing tests. Mirrors
	// pmdarima.auto_arima (auto.py:491-494) and forecast::auto.arima which
	// both fit `y ~ X` via OLS and pass the residuals to nsdiffs/ndiffs.
	// Without this step, when exog explains a chunk of the seasonal pattern
	// (e.g. calendar regressors like Ramadan/holidays), the raw-y test
	// over-detects seasonal differencing and selects D=1 where pmdarima/R
	// (correctly) settle on D=0.
	//
	// When Lambda is set, the differencing tests should see the same
	// transformed series the candidate fits will work on internally, so
	// apply Box-Cox here (only for the diff-test path; yFit / yHoldout
	// stay on original scale so candidate Fit can apply its own Box-Cox).
	yForDiff := yFit
	if opts.Lambda != nil {
		bc, err := boxCoxApply(yFit, *opts.Lambda, opts.Lambda2)
		if err != nil {
			return nil, fmt.Errorf("Box-Cox transform: %w", err)
		}
		yForDiff = bc
	}
	xx := yForDiff
	if xFit != nil && len(xFit) == len(yForDiff) && len(xFit[0]) > 0 {
		if beta, err := olsFit(xFit, yForDiff, true); err == nil && len(beta) == len(xFit[0])+1 {
			adjusted := make([]float64, len(yForDiff))
			for i, v := range yForDiff {
				pred := beta[0] // intercept
				for j, b := range beta[1:] {
					pred += b * xFit[i][j]
				}
				adjusted[i] = v - pred
			}
			xx = adjusted
		}
	}

	// Determine D first (matches pmdarima/R order), then d on the
	// seasonally-differenced residuals. Pre-fix d was determined from raw
	// xx with no seasonal-diff awareness, which can over-detect d=2 on a
	// series whose seasonality hasn't been removed yet.
	D := 0
	if opts.M > 1 {
		if opts.HasDd {
			D = opts.Dd
		} else {
			Dx, err := NSDiffs(xx, NSDiffsOpts{
				M: opts.M, MaxD: opts.MaxCapD, Test: opts.SeasonalTest, MaxLag: 3,
			})
			if err == nil {
				D = Dx
			}
		}
	}

	// Difference by D before estimating d, matching pmdarima auto.py:516-519
	// and R's auto.arima.
	dx := xx
	if D > 0 && opts.M > 1 {
		dx = applyDiff(xx, opts.M, D)
	}

	d := opts.D
	if !opts.HasD {
		nd, err := NDiffs(dx, NDiffsOpts{
			Alpha: opts.Alpha, Test: opts.Test, MaxD: opts.MaxD,
		})
		if err != nil {
			return nil, fmt.Errorf("ndiffs: %w", err)
		}
		d = nd
	}

	// Initial seed.
	p := minInt(opts.StartP, opts.MaxP)
	q := minInt(opts.StartQ, opts.MaxQ)
	if p == 0 && opts.StartP == 0 {
		p = minInt(2, opts.MaxP)
	}
	if q == 0 && opts.StartQ == 0 {
		q = minInt(2, opts.MaxQ)
	}
	P := minInt(opts.StartCapP, opts.MaxCapP)
	Q := minInt(opts.StartCapQ, opts.MaxCapQ)
	if opts.M <= 1 {
		P, Q = 0, 0
	}
	// Clamp initial seed to satisfy MaxOrder.
	for p+q+P+Q > opts.MaxOrder {
		switch {
		case Q > 0:
			Q--
		case P > 0:
			P--
		case q > 0:
			q--
		case p > 0:
			p--
		default:
			// every term already 0; nothing more to drop
			p, q, P, Q = 0, 0, 0, 0
		}
		if p+q+P+Q == 0 {
			break
		}
	}

	// GAP-4: AllowDrift / AllowMean give independent control over the
	// d > 0 (drift) vs d = 0 (mean) case. AllowDrift = nil and
	// AllowMean = nil falls through to the existing WithIntercept logic.
	withIntercept := false
	switch {
	case (d+D) > 0 && opts.AllowDrift != nil:
		// Drift case: explicit AllowDrift wins.
		withIntercept = *opts.AllowDrift
	case (d+D) == 0 && opts.AllowMean != nil:
		// Mean case: explicit AllowMean wins.
		withIntercept = *opts.AllowMean
	case opts.WithIntercept != nil:
		// Either case, no per-case override: WithIntercept applies.
		withIntercept = *opts.WithIntercept
	default:
		// Auto: default-on for d=0 (mean), default-off for d>0 (drift) —
		// matching R's auto.arima default of allowmean=TRUE, allowdrift=FALSE.
		withIntercept = (d + D) == 0
	}

	// Default to stepwise unless caller opted into full enumeration.
	stepwise := !opts.FullSearch

	type orderKey struct{ p, q, P, Q int }
	cache := map[orderKey]*ARIMA{}
	cacheScore := map[orderKey]float64{}
	var cacheMu sync.Mutex // protects cache, cacheScore (stepwise parallel access)

	emit := func(s string) {
		if opts.Trace != nil {
			opts.Trace(s)
		}
	}

	// gradientBudget computes the GradientWorkers cap for each candidate
	// fit when AutoArima dispatches K parallel candidates. Splits cores
	// evenly so the total goroutines (outer K × inner GradientWorkers)
	// stays at GOMAXPROCS instead of K × GOMAXPROCS.
	gradientBudget := func(nParallelFits int) int {
		if nParallelFits <= 1 {
			return 0 // 0 → use full GOMAXPROCS in minimize
		}
		gp := runtime.GOMAXPROCS(0)
		w := gp / nParallelFits
		if w < 1 {
			w = 1
		}
		return w
	}

	// independentFit fits a single candidate without touching the cache.
	// Used both by the FullSearch worker pool and by the stepwise parallel
	// neighbor evaluation. No emit() calls — callers emit on the main
	// goroutine after the fit completes (Trace func may not be thread-safe).
	//
	// gradWorkers is the per-fit BFGS gradient concurrency budget; pass 0
	// to use full GOMAXPROCS (sequential fit context) or a smaller value
	// when this fit is one of K running concurrently (so outer×inner ≤ GOMAXPROCS).
	independentFit := func(k orderKey, gradWorkers int) (*ARIMA, float64, error) {
		if k.p < 0 || k.q < 0 || k.P < 0 || k.Q < 0 {
			return nil, 0, fmt.Errorf("negative orders")
		}
		if k.p > opts.MaxP || k.q > opts.MaxQ || k.P > opts.MaxCapP || k.Q > opts.MaxCapQ {
			return nil, 0, fmt.Errorf("exceeds max order")
		}
		if k.p+k.q+k.P+k.Q > opts.MaxOrder {
			return nil, 0, fmt.Errorf("exceeds max_order sum")
		}
		ord := Order{P: k.p, D: d, Q: k.q}
		var ssn SeasonalOrder
		if opts.M > 1 && (k.P+k.Q+D > 0) {
			ssn = SeasonalOrder{P: k.P, D: D, Q: k.Q, M: opts.M}
		}
		method := opts.Method // zero value = MethodCSSML (post-GAP-1 fix)
		mdl := &ARIMA{
			Order:                 ord,
			Seasonal:              ssn,
			WithIntercept:         withIntercept,
			Method:                method,
			MaxIter:               opts.MaxIter,
			GradientWorkers:       gradWorkers,
			Lambda:                opts.Lambda,
			RidgePenalty:          opts.RidgePenalty,
			Lambda2:               opts.Lambda2,
			NonSimpleDifferencing: opts.NonSimpleDifferencing,
			DiffuseConvention:     opts.DiffuseConvention,
		}
		if err := mdl.Fit(yFit, xFit); err != nil {
			return nil, 0, err
		}
		var score float64
		if opts.OutOfSampleSize > 0 {
			fc, _, _, perr := mdl.Predict(opts.OutOfSampleSize, 0, xHoldout)
			if perr != nil {
				return nil, 0, perr
			}
			s, serr := scoring(yHoldout, fc)
			if serr != nil {
				return nil, 0, serr
			}
			score = s
		} else {
			score = mdl.IC(opts.IC)
		}
		return mdl, score, nil
	}

	// tryOrderTraced fits one candidate (cached) and returns the trace
	// string instead of calling emit() directly. Lets the caller decide
	// WHEN to emit — sequential paths emit immediately, parallel paths
	// collect strings and emit from the main goroutine after wg.Wait()
	// so trace output is deterministic regardless of scheduler order
	// AND user-supplied Trace callbacks don't need to be thread-safe.
	tryOrderTraced := func(p, q, capP, capQ, gradWorkers int) (*ARIMA, float64, string, error) {
		k := orderKey{p, q, capP, capQ}
		cacheMu.Lock()
		if cached, ok := cache[k]; ok {
			score := cacheScore[k]
			cacheMu.Unlock()
			if cached == nil {
				return nil, 0, "", fmt.Errorf("cached failure")
			}
			return cached, score, "", nil // cache hit — no new trace line
		}
		cacheMu.Unlock()
		mdl, score, err := independentFit(k, gradWorkers)
		cacheMu.Lock()
		if err != nil {
			cache[k] = nil
		} else {
			cache[k] = mdl
			cacheScore[k] = score
		}
		cacheMu.Unlock()
		if err != nil {
			line := fmt.Sprintf("ARIMA(%d,%d,%d)(%d,%d,%d)[%d] : fit failed: %v",
				p, d, q, capP, D, capQ, opts.M, err)
			return nil, 0, line, err
		}
		line := fmt.Sprintf("ARIMA(%d,%d,%d)(%d,%d,%d)[%d] : score=%.4f",
			p, d, q, capP, D, capQ, opts.M, score)
		return mdl, score, line, nil
	}

	// tryOrder is the sequential variant: emits the trace line immediately.
	// Used by the initial stepwise fit and by the sequential nJobs==1 path.
	tryOrder := func(p, q, capP, capQ, gradWorkers int) (*ARIMA, float64, error) {
		mdl, score, line, err := tryOrderTraced(p, q, capP, capQ, gradWorkers)
		if line != "" {
			emit(line)
		}
		return mdl, score, err
	}

	handleErr := func(err error) error {
		if err == nil {
			return nil
		}
		switch opts.ErrorAction {
		case "ignore":
			return nil
		case "warn":
			emit(fmt.Sprintf("warning: %v", err))
			return nil
		default:
			return err
		}
	}

	// Full enumeration mode (a.k.a. stepwise=False).
	if !stepwise {
		var combos []orderKey
		for pp := 0; pp <= opts.MaxP; pp++ {
			for qq := 0; qq <= opts.MaxQ; qq++ {
				if opts.M > 1 {
					for cP := 0; cP <= opts.MaxCapP; cP++ {
						for cQ := 0; cQ <= opts.MaxCapQ; cQ++ {
							if pp+qq+cP+cQ > opts.MaxOrder {
								continue
							}
							combos = append(combos, orderKey{pp, qq, cP, cQ})
						}
					}
				} else {
					if pp+qq > opts.MaxOrder {
						continue
					}
					combos = append(combos, orderKey{pp, qq, 0, 0})
				}
			}
		}
		if opts.NFits > 0 && opts.NFits < len(combos) {
			rng := rand.New(rand.NewPCG(opts.Seed, opts.Seed+1))
			rng.Shuffle(len(combos), func(i, j int) { combos[i], combos[j] = combos[j], combos[i] })
			combos = combos[:opts.NFits]
		}

		// Build a per-candidate fit function that doesn't share the order
		// cache (each goroutine fits independently). After fits complete,
		// we merge results.
		nJobs := opts.NJobs
		if nJobs <= 0 {
			nJobs = runtime.GOMAXPROCS(0)
		}
		if nJobs > len(combos) {
			nJobs = len(combos)
		}
		if nJobs < 1 {
			nJobs = 1
		}

		type result struct {
			key   orderKey
			model *ARIMA
			score float64
			err   error
		}
		jobs := make(chan orderKey, len(combos))
		results := make(chan result, len(combos))
		var wg sync.WaitGroup
		gradBudget := gradientBudget(nJobs)

		for w := 0; w < nJobs; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for k := range jobs {
					mdl, score, err := independentFit(k, gradBudget)
					results <- result{key: k, model: mdl, score: score, err: err}
				}
			}()
		}
		for _, c := range combos {
			jobs <- c
		}
		close(jobs)
		go func() { wg.Wait(); close(results) }()

		var best *ARIMA
		bestScore := math.Inf(1)
		var firstErr error
		for r := range results {
			if r.err != nil {
				if e := handleErr(r.err); e != nil && firstErr == nil {
					firstErr = e
				}
				emit(fmt.Sprintf("ARIMA(%d,%d,%d)(%d,%d,%d)[%d] : fit failed: %v",
					r.key.p, d, r.key.q, r.key.P, D, r.key.Q, opts.M, r.err))
				continue
			}
			emit(fmt.Sprintf("ARIMA(%d,%d,%d)(%d,%d,%d)[%d] : score=%.4f",
				r.key.p, d, r.key.q, r.key.P, D, r.key.Q, opts.M, r.score))
			if r.score < bestScore {
				bestScore = r.score
				best = r.model
			}
		}
		// "raise" semantics: any errored candidate aborts the whole search,
		// even if other candidates succeeded. Pre-fix this branch only
		// surfaced firstErr when best was still nil — silently swallowing
		// fit failures whenever any other candidate produced a model.
		if firstErr != nil {
			return nil, firstErr
		}
		if best == nil {
			return nil, errors.New("no candidate fit succeeded")
		}
		return best, nil
	}

	// Stepwise neighbor exploration.
	best, bestScore, err := tryOrder(p, q, P, Q, 0)
	if err != nil {
		if e := handleErr(err); e != nil {
			return nil, fmt.Errorf("initial fit failed: %w", e)
		}
		// fall through with bestScore=Inf; subsequent neighbors may succeed.
		bestScore = math.Inf(1)
	}
	bestKey := orderKey{p, q, P, Q}

	improved := true
	for iter := 0; iter < opts.MaxSteps && improved; iter++ {
		improved = false
		neighbors := []orderKey{
			{bestKey.p - 1, bestKey.q, bestKey.P, bestKey.Q},
			{bestKey.p + 1, bestKey.q, bestKey.P, bestKey.Q},
			{bestKey.p, bestKey.q - 1, bestKey.P, bestKey.Q},
			{bestKey.p, bestKey.q + 1, bestKey.P, bestKey.Q},
		}
		if opts.M > 1 {
			neighbors = append(neighbors,
				orderKey{bestKey.p, bestKey.q, bestKey.P - 1, bestKey.Q},
				orderKey{bestKey.p, bestKey.q, bestKey.P + 1, bestKey.Q},
				orderKey{bestKey.p, bestKey.q, bestKey.P, bestKey.Q - 1},
				orderKey{bestKey.p, bestKey.q, bestKey.P, bestKey.Q + 1},
			)
		}

		// Drop neighbors that are out-of-bounds for the search box. These
		// aren't fit failures — they're a normal consequence of stepwise
		// neighbor expansion (e.g., bestKey.p=0 produces (-1,…) which is
		// just skipped). Filtering here so ErrorAction="raise" only fires
		// on real fit errors below.
		filtered := neighbors[:0]
		for _, n := range neighbors {
			if n.p < 0 || n.q < 0 || n.P < 0 || n.Q < 0 {
				continue
			}
			if n.p > opts.MaxP || n.q > opts.MaxQ || n.P > opts.MaxCapP || n.Q > opts.MaxCapQ {
				continue
			}
			if n.p+n.q+n.P+n.Q > opts.MaxOrder {
				continue
			}
			filtered = append(filtered, n)
		}
		neighbors = filtered

		// Fit neighbors in parallel. Each iteration evaluates 4–8 candidates
		// that are independent (no data dependency between them within an
		// iteration); the cross-iteration dependency comes from `bestKey` only.
		// We still need cache lookups to avoid re-fitting orders seen in
		// earlier iterations, so use the cache-aware tryOrder under cacheMu.
		nJobs := opts.NJobs
		if nJobs <= 0 {
			nJobs = runtime.GOMAXPROCS(0)
		}
		if nJobs > len(neighbors) {
			nJobs = len(neighbors)
		}
		if nJobs < 1 {
			nJobs = 1
		}

		type stepResult struct {
			model *ARIMA
			score float64
			trace string // collected trace line; emitted in stable order below
			err   error
		}
		results := make([]stepResult, len(neighbors))

		gradBudget := gradientBudget(nJobs)
		if nJobs == 1 {
			// Sequential — preserves zero goroutine overhead at small parallelism.
			for i, n := range neighbors {
				cand, score, line, err := tryOrderTraced(n.p, n.q, n.P, n.Q, 0)
				results[i] = stepResult{cand, score, line, err}
			}
		} else {
			var wg sync.WaitGroup
			sem := make(chan struct{}, nJobs)
			for i, n := range neighbors {
				i, n := i, n // capture for goroutine
				wg.Add(1)
				sem <- struct{}{}
				go func() {
					defer wg.Done()
					defer func() { <-sem }()
					cand, score, line, err := tryOrderTraced(n.p, n.q, n.P, n.Q, gradBudget)
					results[i] = stepResult{cand, score, line, err}
				}()
			}
			wg.Wait()
		}

		// Process results in stable original order so ties resolve identically
		// to the sequential implementation (preserves determinism). Under
		// ErrorAction="raise" any neighbor failure aborts the whole search,
		// matching the documented contract. Trace lines collected from the
		// workers are also emitted here in stable order — so a user's Trace
		// callback (which previously had to be thread-safe per the
		// documentation warning) now sees deterministic output and runs only
		// on the main goroutine.
		for i, r := range results {
			if r.trace != "" {
				emit(r.trace)
			}
			if r.err != nil || r.model == nil {
				if e := handleErr(r.err); e != nil {
					return nil, e
				}
				continue
			}
			candParams := neighbors[i].p + neighbors[i].q + neighbors[i].P + neighbors[i].Q
			bestParams := bestKey.p + bestKey.q + bestKey.P + bestKey.Q
			threshold := 1e-6
			if candParams > bestParams && opts.ParsimonyDelta > threshold {
				threshold = opts.ParsimonyDelta
			}
			if r.score < bestScore-threshold {
				best = r.model
				bestScore = r.score
				bestKey = neighbors[i]
				improved = true
			}
		}
	}
	if best == nil {
		return nil, errors.New("no successful fit")
	}
	return best, nil
}

// minInt returns the smaller of two ints.
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
