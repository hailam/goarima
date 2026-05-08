# What hasn't been parallelised, and why

- **Kalman time-loop** (`for t := 0; t < n; t++`) — sequential by
  construction (each step uses `P_{t-1}`). PSCAN-1 (parallel
  prefix-sum reformulation) is deferred — speed gain only at
  T > ~2000 and the workload doesn't surface there in practice.
- **BFGS itself** — gonum's optimizer is sequential. We parallelise
  the gradient evaluation inside it via `parallelGradient`.
- **Stepwise neighbour enumeration** — generating candidates is
  cheap; the work is the fits, which already run in parallel.
- **Differencing tests** (KPSS / OCSB / SEAS / HEGY) — run once
  per AutoArima call; not on the per-fit hot path.
