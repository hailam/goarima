# Stepwise search

At each step, propose 4-8 neighbours from the current
`(p, d, q, P, D, Q)`:

- ±1 on p, q, P, Q
- Toggle constant inclusion (intercept on/off)
- Optionally diagonals (`StepwiseDiagonals=true` — opt-in, off by
  default per the PG-4a empirical study)

Visited models cache hit → skip refit. The cache is a
`sync.Mutex`-guarded map; that's the only contended state during
stepwise (see [`../concurrency/locks.md`](../concurrency/locks.md)).

## Determinism

PG-91 fixed a determinism bug where parallel-gradient ordering could
change which model wins on AICc ties. Set `NJobs=1` if you need
bit-exact reproducibility across runs. With NJobs>1 the *picked
model* is deterministic but the *order of visits* (and the trace
output) can vary.
