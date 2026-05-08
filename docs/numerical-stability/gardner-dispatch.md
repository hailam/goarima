# Stationary-covariance dispatch (GARD-OPT-1)

Initial state covariance `P_0` is the solution of the discrete
Lyapunov equation `P = T·P·Tᵀ + R·Rᵀ`. Three implementations live in
`arima/getq0.go`, dispatched by `(p, r)`:

```mermaid
flowchart TD
    A[stationaryCovGardnerInto] --> B{p == 0?}
    B -->|yes| C[pure-MA closed form<br/>O(r²)]
    B -->|no| D{r >= 20?}
    D -->|yes| E[Smith doubling iteration<br/>O(r³)]
    D -->|no| F[inclu2 Givens rotations<br/>O(r⁴)]
    E -->|fails to converge| F
```

| Path | Complexity | Wins at | Rounding behaviour |
|---|---|---|---|
| Pure-MA closed form | O(r²) | always when p=0 | exact |
| inclu2 (R's `getQ0bis` port) | O(r⁴) | small r (≤ 19) — tight inner loop, cache-friendly | ~1e-15 typical, but Givens chain accumulates ~1e-3 worst-case at r=100 |
| Smith doubling | O(r³) with small constants | r ≥ 20 with p > 0 — dominant on weekly/hourly seasonal models | matmuls only; ~1e-15 even at r=100 |

## Smith convergence

Smith's iteration squares T at each step — geometric convergence when
ρ(T) < 1. The convergence test requires *both* the T-norm to have
shrunk to 1e-15 of original AND the increment to be small relative
to P (the second guard catches the pure-AR pitfall where Q has only
one nonzero and the very first increment looks small before
propagating). On non-stationary T (BFGS off the manifold) Smith
returns false and the dispatcher falls back to inclu2 — see
[`fallbacks.md`](fallbacks.md).

## Tests

`arima/getq0_test.go` has two tests: `TestSmithVsInclu2_Parity`
(mutual agreement at moderate r) and `TestSmithDenseLyapunov` (Smith
vs the dense O(r⁶) reference at small r — the strongest correctness
check, independent of inclu2).
