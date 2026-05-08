# Concurrency

Goroutines, pools, and locks. The library is "concurrent where it
helps and serial where it doesn't" — guided by profile evidence,
not by reflex.

## Contents

- [`goroutines.md`](goroutines.md) — where parallelism fires
- [`pools.md`](pools.md) — `sync.Pool` layout, workspace lifecycle
- [`locks.md`](locks.md) — what's locked, what's lockless
- [`knobs.md`](knobs.md) — how to tune `GradientWorkers` / `NJobs`
- [`not-parallel.md`](not-parallel.md) — what's deliberately serial
