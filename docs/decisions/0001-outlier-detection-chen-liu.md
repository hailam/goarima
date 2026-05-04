# 0001 — Outlier detection: Chen-Liu residual-projection over per-candidate refit

**Status**: Accepted (2026-05-04)
**Implements**: G-NEW-3a (verified gap from the multi-review backlog).
**Code**: `arima/outliers.go`, `arima/outliers_test.go`.

## Context

R's `tsoutliers::tso()` builds outlier intervention regressors (AO, LS, TC, IO)
and refits an ARIMA iteratively to mask out shocks before forecasting. pmdarima
has no direct equivalent (users compose it via statsmodels). goarima had zero
outlier-detection support, which was a real gap for users with anomaly-prone
data — retail series with COVID drops, Eid spikes, Friday-closure patterns,
strikes — i.e., the kukuru-ml workload.

We needed a port that: (a) matches the algorithmic spirit of `tsoutliers::tso`,
(b) is usable as a building block by AutoArima callers, (c) doesn't blow up
fit time on long series, and (d) keeps the public API small.

## Decision

### Algorithm: simplified Chen-Liu (1993) residual projection

For each detection round:

1. Fit a base ARIMA on `y` with the current outlier-regressor matrix.
2. Compute combined-π weights of the inverse ARMA filter:

   ```
   A(L) = φ(L) · Φ(L^M) · (1-L)^d · (1-L^M)^D
   B(L) = θ(L) · Θ(L^M)
   π(L) = A(L)/B(L)            (recovered by polynomial division)
   ```

3. For each candidate `(τ, type)`, score the t-statistic of the residual
   projection onto the candidate's inverse-filtered signature:

   ```
   sig_AO(t, τ) = π_{t-τ}                        for t ≥ τ, else 0
   sig_LS(t, τ) = Σ_{k=0..t-τ} π_k                for t ≥ τ, else 0
   t_stat(τ, type) = Σ_t sig(t)·ε_t / (σ̂ · √Σ_t sig(t)²)
   ```

4. If the largest |t_stat| exceeds `CritVal`, append the corresponding
   regressor (impulse for AO, step for LS) to the exog matrix and refit.
5. Repeat ≤ `MaxIter` times.

### Why not brute-force per-candidate refit

A natural-feeling alternative is "for each candidate, augment exog with that
single regressor and refit; pick the best |t-stat| from the refitted β."
That's `O(n × types × maxIter × fit_cost)`. For `n=200`, `types=2`,
`maxIter=5`, `fit_cost ≈ 50ms` → **100s** per call. Unacceptable.

The residual-projection approach is `O(n × types × maxIter)` plus
`maxIter` refits, i.e., ~5× a normal Fit on n=200 — a few hundred ms.

### Chosen scope: AO + LS only

R's `tsoutliers` supports four types: AO, LS, TC (temporary change, geometric
decay), IO (innovational outlier). We implement only AO and LS because:

- They cover the dominant real-world contamination shapes: one-off shocks
  and regime changes.
- TC adds a `δ ∈ (0,1)` decay parameter that requires a separate inner
  search (or a pmdarima-style fixed `δ=0.7`); deferred.
- IO is rarely useful in practice — it's mathematically equivalent to a
  shock that propagates through the ARMA filter and is hard to interpret
  for users who think in terms of "what happened on date X."
- The API leaves room: `DetectOutliersOpts.Types []OutlierType` accepts
  a list, so adding TC/IO later is non-breaking.

### Outlier indices reported in original-time scale

The π-weights include the full differencing operator `(1-L)^d (1-L^M)^D`,
so the projection naturally maps original-scale impulse/step regressors
through to the residual signature. Users get back `Outlier.Index` in the
same indexing as the input `y`, not the differenced series.

The alternative (report indices in differenced space) would have been
simpler to implement but worse UX — users read "outlier at y[42]"
directly without doing arithmetic against `d + D·M`.

### Default model: ARIMA(0,1,1)

`tsoutliers` uses an "approximation model" when the user doesn't specify
one. ARIMA(0,1,1) is the default because random-walk-plus-MA1 is the
most robust shape across diverse series — non-stationary trend handled
by `d=1`, residual autocorrelation handled by `q=1`. Users who know
their data should pass `Order` and `Seasonal` explicitly.

### CritVal default 3.5 / 4.0 by n

Matches `tsoutliers`'s default `cval` — 3.5 for moderate series,
4.0 for `n ≥ 200`. The threshold is on |t-stat| of a normal-ish
projection, so these correspond to ~Bonferroni-corrected ~0.05 family-wise
error rates over `n × types` candidates.

### One outlier per round, not multi-add

Each round picks the single largest |t-stat| above `CritVal` and refits.
The "multi-add" alternative (declare every candidate above `CritVal`
in one round) is faster but contaminates: a strong AO at t=50 distorts
nearby residuals, producing spurious AO ghosts at t=49/51 that the
multi-add would also accept. The single-add discipline lets the refit
absorb the strong one before the next round re-evaluates everything.

### Coefficients refreshed from final β, not the projection estimate

Each `Outlier.Coef` returned to the caller is taken from the final
fitted model's `Beta()` slice (offset to skip user-supplied exog).
The projection-time coefficient is a one-shot OLS estimate that
ignores the simultaneous refit of φ/θ/Φ/Θ; the final β is the proper
joint estimate.

## Consequences

### Positive

- ~330 LOC, one file (`arima/outliers.go`) plus 5 tests.
- Detection on `n=200` with planted outliers runs in ~5 ms (effectively
  ~5 ARIMA fits).
- Public API symmetric with rest of `arima/`: `Outlier`, `OutlierType`,
  `DetectOutliersOpts`, `DetectOutliers`. No stutter naming.

### Negative / known limitations

- **No TC, no IO.** Series dominated by exponentially-decaying shocks
  (e.g. response to a marketing campaign that fades over weeks) won't
  be cleanly captured.
- **No simultaneous-outlier refinement.** The "Stage 3" of full Chen-Liu
  joint-refits all detected outliers and drops any that lose
  significance. We rely on the per-round refit to keep the list clean,
  which is weaker on series with many co-located outliers.
- **No backward-pass elimination.** R's `tso` runs a final pruning pass
  to drop outliers that became insignificant after later additions.
  Skipped — `MaxIter=5` keeps the list short enough that this rarely
  matters in practice.
- **Detection sensitive to the assumed Order.** If the user passes a
  badly-misspecified `Order`, the π-weights are wrong and detection
  degrades. Mitigated by the ARIMA(0,1,1) default which is robust.

### Open questions for future work

- Should AutoArima gain a `DetectOutliers bool` flag that runs the
  detector before model selection? Currently users compose it manually:
  `outs, m, _ := DetectOutliers(...)`, then call `AutoArima(y, regs, ...)`
  with the detected regressors as exog. The composition is fine; a
  one-flag convenience could come if requested.
- Should we expose the π-weights computation as a public helper? It's
  useful for users who want to filter their own series. Currently
  unexported (`computePiWeights`).
