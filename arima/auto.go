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
	// MAY BE CALLED CONCURRENTLY in stepwise/FullSearch modes; protect any
	// shared state.
	Trace func(string)

	// ErrorAction controls behavior when a candidate fit errors:
	//   "raise" (default): the first error aborts the whole search
	//   "ignore"          : skip the failing candidate, continue
	//   "warn"            : like "ignore" but call Trace if present
	ErrorAction string

	// MaxSteps caps the number of stepwise iterations (0 → 50).
	MaxSteps int

	// NJobs sets goroutine concurrency for both stepwise (4–8 neighbor fits
	// per iteration) and FullSearch (whole search box) modes. 0 → GOMAXPROCS.
	// Mirrors pmdarima's `n_jobs`. R/Python require pickling+IPC for parallel
	// auto.arima, but Go's goroutines are essentially free.
	NJobs int

	// Method selects the per-candidate fitting estimator (CSS / ML / CSS-ML).
	// Default 0 (MethodCSSML) — same as a hand-rolled `Fit`. Mirrors
	// pmdarima's `method=` and R's `method=` arguments to auto.arima.
	Method Method

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

	withIntercept := false
	if opts.WithIntercept != nil {
		withIntercept = *opts.WithIntercept
	} else {
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
		method := opts.Method // zero value = MethodCSSML, same as before
		mdl := &ARIMA{
			Order:                 ord,
			Seasonal:              ssn,
			WithIntercept:         withIntercept,
			Method:                method,
			MaxIter:               opts.MaxIter,
			GradientWorkers:       gradWorkers,
			Lambda:                opts.Lambda,
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

	// Fit one candidate; cached. Returns model and the chosen score (lower=better).
	// Sequential entry point (used by the initial stepwise fit). Thread-safe
	// against concurrent tryOrderCached / fitNeighborParallel via cacheMu.
	// gradWorkers is the BFGS gradient concurrency budget — pass via
	// closure-bound variable that callers update before parallel dispatch.
	tryOrder := func(p, q, capP, capQ, gradWorkers int) (*ARIMA, float64, error) {
		k := orderKey{p, q, capP, capQ}
		cacheMu.Lock()
		if cached, ok := cache[k]; ok {
			score := cacheScore[k]
			cacheMu.Unlock()
			if cached == nil {
				return nil, 0, fmt.Errorf("cached failure")
			}
			return cached, score, nil
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
			emit(fmt.Sprintf("ARIMA(%d,%d,%d)(%d,%d,%d)[%d] : fit failed: %v",
				p, d, q, capP, D, capQ, opts.M, err))
			return nil, 0, err
		}
		emit(fmt.Sprintf("ARIMA(%d,%d,%d)(%d,%d,%d)[%d] : score=%.4f",
			p, d, q, capP, D, capQ, opts.M, score))
		return mdl, score, nil
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
			err   error
		}
		results := make([]stepResult, len(neighbors))

		gradBudget := gradientBudget(nJobs)
		if nJobs == 1 {
			// Sequential — preserves zero goroutine overhead at small parallelism.
			for i, n := range neighbors {
				cand, score, err := tryOrder(n.p, n.q, n.P, n.Q, 0)
				results[i] = stepResult{cand, score, err}
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
					cand, score, err := tryOrder(n.p, n.q, n.P, n.Q, gradBudget)
					results[i] = stepResult{cand, score, err}
				}()
			}
			wg.Wait()
		}

		// Process results in stable original order so ties resolve identically
		// to the sequential implementation (preserves determinism). Under
		// ErrorAction="raise" any neighbor failure aborts the whole search,
		// matching the documented contract.
		for i, r := range results {
			if r.err != nil || r.model == nil {
				if e := handleErr(r.err); e != nil {
					return nil, e
				}
				continue
			}
			if r.score < bestScore-1e-6 {
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
