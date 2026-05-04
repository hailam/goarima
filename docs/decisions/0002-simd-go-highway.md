# 0002 — SIMD via go-highway: targeted adoption, point-win only

**Status**: Accepted (2026-05-04)
**Code**: `arima/arima.go` (residOf / residOfFull), `arima/simd_bench_test.go`.

## Context

User asked whether to adopt `github.com/ajroetker/go-highway` as a SIMD
acceleration layer. The library is a deliberate stop-gap: it provides
slice-level vector ops (`Add`, `Sub`, `Mul`, `Dot`, `Sum`,
`MulConstAddTo`, etc.) over `[]float64`/`[]float32` until Go's native
SIMD lands as GA, at which point the dependency is intended to be
removed.

A prior CGO-based SIMD library was rejected. go-highway is pure Go +
generated assembly (no `import "C"` anywhere in the tree).

## Investigation

### Architecture & build-flag landscape

- **ARM64** (Apple silicon, Graviton, etc.): NEON kernels via GoAT-
  generated assembly. **On by default**, no flags.
- **AMD64**: routes through Go 1.26's `simd/archsimd`, which is gated by
  `goexperiment.simd`. Without `GOEXPERIMENT=simd` at build time, the
  AMD64 path silently falls back to scalar Go. Documented as a setup
  step in the README.
- **All other targets**: pure-Go scalar fallback (`_other.go` build
  tag). Correct, just not faster.

### Micro-bench results (Apple M4 Max, ARM64 NEON, n = realistic ARIMA)

| Op | n | Scalar | SIMD | Speedup |
|---|---|---|---|---|
| `Sum` | 144 | 40 ns | 21 ns | 1.9× |
| `Sum` | 1000 | 241 ns | 230 ns | 1.05× (compiler auto-vec wins) |
| `Dot` | 144–1000 | 39–238 ns | 16–97 ns | **2.5×** consistently |
| `MulConstAddTo` | 144 | 52 ns | 34 ns | 1.5× |
| `MulConstAddTo` | 1000 | 345 ns | 210 ns | 1.6× |
| `MulConstAddTo` | **14** | **5.7 ns** | **6.7 ns** | **0.85× (slower)** |
| `MulConstAddTo` | **28** | **10.5 ns** | **11.3 ns** | **0.93× (slower)** |
| `residOf` (k=2) | 144 | 235 ns | 104 ns | **2.26×** |
| `residOf` (k=5) | 144 | 439 ns | 173 ns | **2.53×** |
| `residOf` (k=5) | 1000 | 3054 ns | 1065 ns | **2.87×** |

Two clear regimes:
1. **n ≥ ~50, structured AXPY/dot**: SIMD wins 1.5–3×.
2. **n ≤ ~30**: SIMD loses to scalar (per-call dispatch overhead exceeds
   the saved arithmetic). Stay scalar.

### End-to-end bench

Conversion of `residOf` + `residOfFull` to use `vec.MulConstAddTo` over a
column-transposed exog matrix. A/B on the same hardware:

| Bench | Scalar | SIMD | Δ |
|---|---|---|---|
| `BenchmarkFitARIMA011_Airline` (k=0 — path not entered) | 14.22 ms | 14.36 ms | within noise |
| `BenchmarkFitARIMA011_AirlineWithExog` (k=5) | 17.39 ms | 17.68 ms | within noise |
| `BenchmarkFitNonSimple_Airline` (k=0) | 132.3 ms | 136.1 ms | within noise |
| `BenchmarkAutoArima_AirPassengers` (k=0) | 4.57 ms | 4.53 ms | within noise |

End-to-end is flat. CPU profile (`go test -cpuprofile`) explains why:

- **97% of CPU during Fit is goroutine scheduler overhead** (`pthread_cond_signal`, `wakep`, `findRunnable`) from `parallelGradient` dispatching workers for finite-difference gradient computation. The per-worker work is so small that scheduler dispatch dominates.
- **User code is 2.04% of total CPU**:
  - `kalmanARMALikelihood` 1.07% (the dominant user-side bottleneck)
  - `vec.MulConstAddTo` 0.097% (SIMD is being called, but the call is tiny relative to scheduler)
  - everything else <0.5%

So the SIMD micro-win is real but invisible end-to-end because `residOf`
isn't on the critical path. Improving `residOf` from 439 ns to 173 ns
saves ~270 ns × ~500 calls/fit = 135 µs on a 17 ms fit = **0.8% of fit
time** — well below noise.

### Sites considered and rejected

- **`kalmanARMALikelihood` rank-1 P update / TP = T·P inner loops**:
  the row size is `r ≤ p + q·M` ≈ 14 for ARIMA(0,1,1)(0,1,1)[12]. At
  n=14 SIMD is **slower** than scalar (6.7 ns vs 5.7 ns). Below
  break-even. Stay scalar.
- **`armaCSS` / bootstrap simulation recurrences**: sequential
  dependency `res[t] = f(res[t-1..t-q])`. Cannot vectorize the outer
  loop. Inner sums are length `p+q ≤ 5–10`, also below break-even.
- **`buildPsi` ψ-recursion**: same sequential dependency.
- **`olsFit` / Hannan-Rissanen**: already routed through `gonum/mat` →
  BLAS → SIMD. Re-vectorizing on top would just bypass BLAS.
- **Sum reductions** (n ≥ 500): compiler auto-vectorization already
  matches go-highway. No win.

### Sites converted

`residOf` and `residOfFull` in `arima/arima.go`:

```go
// Pre-transpose wX to column-major once (invariant across optimizer calls).
var wXT [][]float64
if k > 0 {
    wXT = make([][]float64, k)
    for j := 0; j < k; j++ {
        col := make([]float64, len(ws))
        for i := 0; i < len(ws); i++ { col[i] = wX[i][j] }
        wXT[j] = col
    }
}

residOf := func(params []float64) []float64 {
    _, _, _, _, c, beta := unpackParamsX(...)
    out := make([]float64, len(ws))
    for i, v := range ws { out[i] = v - c }
    for j := 0; j < k; j++ {
        vec.MulConstAddTo(out, -beta[j], wXT[j])  // dst[i] += -beta[j] * wXT[j][i]
    }
    return out
}
```

The pre-transpose pays off after the first ~3 invocations of `residOf`;
the closure is called ~500 times per fit.

## Decision

**Adopt go-highway for `residOf` / `residOfFull` only.** Do not
bulk-convert. The micro-win on these specific loops is real and stable
across n=144–1000 with k≥2 exog cols.

The end-to-end metric is flat for now because of the dominant
goroutine-scheduler overhead in `parallelGradient` — that's a separate
PERF item, tracked in the lockless audit (the actual answer there will
likely be: switch to serial gradient for small problems, since the
parallelism has negative ROI when per-worker work is microseconds).

When the parallelGradient overhead is fixed, residOf will become a
larger fraction of the remaining cost and the SIMD conversion will
start showing up on end-to-end benches. Until then, the conversion is
"free correct code" with a 2.5× win on its own loop.

## Consequences

### Positive

- A point optimization that's already paying off in its niche
  (residual computation with non-trivial exog).
- The conversion makes the code arguably clearer: `vec.MulConstAddTo(out, -beta[j], wXT[j])`
  has explicit AXPY semantics. No hand-rolled inner loop.
- Pre-transpose pattern is reusable: any future column-wise ops on
  exog matrices already have `wXT` / `xUndiffT` available.

### Negative

- Adds a transitive dependency on `golang.org/x/sys/cpu` (pulled in by
  go-highway's runtime CPU feature detection).
- AMD64 users without `GOEXPERIMENT=simd` see no speedup. Documented
  in README, but it's still a setup gotcha.
- Hard floor on Go 1.26+ (already in `go.mod`, so not a regression).

### Migration plan

When Go's native SIMD lands as GA (target: Go 1.27 or 1.28 based on
proposal status):
1. Replace `vec.MulConstAddTo(...)` with the equivalent stdlib call.
2. `go mod tidy` removes the go-highway and `x/sys/cpu` deps.
3. Drop the `GOEXPERIMENT=simd` note from the README.

This is what go-highway is explicitly designed for — it's a transition
adapter, not a permanent dep.

## Re-investigation 2026-05-05 (post CSS-1)

After CSS-1 (5× speedup on default Fit), re-profiled to identify any
remaining vectorizable hot loops. Findings:

### Profile of post-CSS-1 Fit

`go test -bench=FitARIMA011_Airline -cpuprofile`:

- 70% of CPU is still `pthread_cond_signal` etc. from `parallelGradient`
  (that's PG-1, separate concern — investigated, current threshold is
  correct, not actionable).
- User-code share: ~2% of total CPU, dominated by:
  - `kalmanARMALikelihood`: 1.31% flat
  - `armaCSS`: 0.22% cum
  - `expandSMA` / `maTransparams`: <0.15% each

### Kalman inner-loop break-even

`BenchmarkTwoDot_*` in `arima/simd_bench_test.go` measures the matvec/
dot pattern at actual ARIMA shapes:

| n | Pattern | Scalar | SIMD | Verdict |
|---|---|---|---|---|
| 14 | monthly SARIMA simple-diff (r) | 4.89 ns | 9.13 ns | **SIMD 1.87× slower** |
| 27 | monthly NonSimple (rd = r + d + D·M) | 8.79 ns | 11.59 ns | **SIMD 1.32× slower** |
| 50 | high-frequency seasonal (M ≥ 24) | 16.15 ns | 15.83 ns | break-even |

**SIMD break-even on this hardware is around n=50.** Below that, the
per-call dispatch cost in `vec.Dot` / `vec.MulConstAddTo` exceeds the
saved arithmetic.

### Conclusion: no further SIMD wins for typical ARIMA shapes

For the dominant use case (monthly data, M=12, r ≤ 14, rd ≤ 27), every
remaining hot loop is below SIMD break-even. Converting them would
**regress** performance.

**SIMD only pays off for high-frequency seasonal models**: M=24 quarterly
(rd ≈ 50), M=52 weekly (rd ≈ 100+), M=365 daily (rd ≈ 400+). When such
a workload arrives — re-bench the same loops at the larger r and apply
`vec.MulConstAddTo` / `vec.Dot` selectively. Until then, the existing
residOf conversion is the only payoff site and the rest stays scalar.

The compiler already auto-vectorizes simple AXPY loops well at small n
(see Sum_Scalar at n=1000 matching Sum_SIMD), so we're not leaving
performance on the table — Go's optimizer is handling the small cases.

### Other vectorizable patterns ruled out

| Site | Why not |
|---|---|
| `armaCSS` recursion | Sequential dep on `res[t-1-j]` — not vectorizable. Inner sum length p+q ≤ 14 also below break-even. |
| `simulateOne` (bootstrap) | Same sequential dep as armaCSS. |
| `buildPsi` ψ-recursion | Same sequential dep. |
| `expandSMA` / `expandSARMA` | Polynomial mul ~3 × ~13 — tiny. |
| `stationaryCovGardner` | Many r-sized inner loops, r=14. Below break-even. |
| `olsFit` / Hannan-Rissanen | Already routed through gonum/BLAS. |
| `kalmanARIMAFullConv` rd-sized loops | rd=27 for airline NonSimple, below break-even. Would help if rd > 50. |

## Open follow-ups

- **parallelGradient overhead** (97% of Fit's CPU): convert to serial
  gradient when problem size is below a threshold, or use a static
  worker pool with much lower wakeup cost. This is the highest-leverage
  remaining optimization in goarima Fit, dwarfing anything SIMD can do.
- **Convert any future hot loops** that surface above ~50-element
  AXPY/dot work: same pattern (column-transpose + `vec.*` call).
- **Reconsider on AMD64-with-AVX-512**: if a target deployment routinely
  builds with `GOEXPERIMENT=simd`, the win might be larger; re-bench
  there before generalizing.
