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

	// Determine d on the fitting set.
	d := opts.D
	if !opts.HasD {
		nd, err := NDiffs(yFit, NDiffsOpts{
			Alpha: opts.Alpha, Test: opts.Test, MaxD: opts.MaxD,
		})
		if err != nil {
			return nil, fmt.Errorf("ndiffs: %w", err)
		}
		d = nd
	}

	// Determine D if seasonal.
	D := 0
	if opts.M > 1 {
		if opts.HasDd {
			D = opts.Dd
		} else {
			Dx, err := NSDiffs(yFit, NSDiffsOpts{
				M: opts.M, MaxD: opts.MaxCapD, Test: NSDiffsCH, MaxLag: 3,
			})
			if err == nil {
				D = Dx
			}
		}
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

	// independentFit fits a single candidate without touching the cache.
	// Used both by the FullSearch worker pool and by the stepwise parallel
	// neighbor evaluation. No emit() calls — callers emit on the main
	// goroutine after the fit completes (Trace func may not be thread-safe).
	independentFit := func(k orderKey) (*ARIMA, float64, error) {
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
		mdl := &ARIMA{
			Order:         ord,
			Seasonal:      ssn,
			WithIntercept: withIntercept,
			Method:        MethodCSSML,
			MaxIter:       opts.MaxIter,
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
	tryOrder := func(p, q, capP, capQ int) (*ARIMA, float64, error) {
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
		mdl, score, err := independentFit(k)
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

		for w := 0; w < nJobs; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for k := range jobs {
					mdl, score, err := independentFit(k)
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
		if best == nil {
			if firstErr != nil {
				return nil, firstErr
			}
			return nil, errors.New("no candidate fit succeeded")
		}
		return best, nil
	}

	// Stepwise neighbor exploration.
	best, bestScore, err := tryOrder(p, q, P, Q)
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

		if nJobs == 1 {
			// Sequential — preserves zero goroutine overhead at small parallelism.
			for i, n := range neighbors {
				cand, score, err := tryOrder(n.p, n.q, n.P, n.Q)
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
					cand, score, err := tryOrder(n.p, n.q, n.P, n.Q)
					results[i] = stepResult{cand, score, err}
				}()
			}
			wg.Wait()
		}

		// Process results in stable original order so ties resolve identically
		// to the sequential implementation (preserves determinism).
		for i, r := range results {
			if r.err != nil || r.model == nil {
				_ = handleErr(r.err)
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
