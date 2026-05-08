# Locks and lockless patterns

The codebase has very few locks. Where they exist:

| Lock | Where | What it guards |
|---|---|---|
| `sync.Mutex` on stepwise visit cache | `arima/auto.go` | Memoisation of (Order,Seasonal) → fit result during stepwise; NJobs>1 needs it |
| `sync.RWMutex` in `*Summary` | `arima/summary.go` | Lazy SE / z-stat computation on first access |
| Pool internals | `sync.Pool` runtime | (not user-visible) |

There's deliberately **no lock on the Kalman workspace** — each
`parallelGradient` worker pulls a fresh `paramScratch` from the pool
and owns it for the duration of one call before returning it. The
Kalman hot path is lockless.

## "Lockless caches"

Two patterns in the code are commonly described as lockless:

1. **Per-goroutine workspaces via sync.Pool** — concurrent
   likelihood evaluations don't share buffers. See
   [`pools.md`](pools.md).
2. **Flat backing storage for bootstrap paths** (CDX-4): one
   contiguous `pathsFlat` array in `arima/bootstrap.go`; each worker
   writes its own rows by index. No mutex; writes don't overlap.

The stepwise visit cache *does* hold a mutex; it's the only contended
shared state and it's brief (map insert / lookup).
