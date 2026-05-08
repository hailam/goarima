# Architecture

How the codebase is organised. For per-topic deep dives, see the
sibling folders.

## Contents

- [`packages.md`](packages.md) — package map, what lives where
- [`fit-flow.md`](fit-flow.md) — top-level `Fit()` flow
- [`state.md`](state.md) — where state lives during a Fit
- [`subsystems.md`](subsystems.md) — quick "where to look" index

## Cross-references

- [`../concurrency/`](../concurrency/) — goroutines, pools, locks
- [`../kalman/`](../kalman/) — Kalman variants and dispatch
- [`../auto-arima/`](../auto-arima/) — search modes
- [`../numerical-stability/`](../numerical-stability/) — Joseph, transparams, Smith
- [`../api-mapping/`](../api-mapping/) — pmdarima/R → goarima
- [`../policies/`](../policies/) — divergence policy, parity modes
