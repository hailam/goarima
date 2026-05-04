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
