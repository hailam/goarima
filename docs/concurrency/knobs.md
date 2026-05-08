# Tuning knobs

## `GradientWorkers`

Defaults to `runtime.NumCPU()`. Lower it on shared hosts; raise it
for very large parameter counts (rare). For very small n (≤ 50) the
goroutine dispatch overhead can outweigh savings — set to 1 if
profiling says so.

## `NJobs` (AutoArima)

Controls candidate-fit parallelism. Default is NumCPU. Set to 1 to
make stepwise deterministic — useful for reproducing exact
AICc-tied picks across runs (PG-91 fixed the underlying determinism
issue but ordered scheduling can still differ between runs at NJobs>1).

## `numWorkers` (PredictBoot)

Should be small relative to NumCPU when called inside a parallel CV
loop, otherwise nested parallelism oversubscribes. A common pattern:
NumCPU at the CV-fold layer, `numWorkers=1` inside the per-fold
PredictBoot.
