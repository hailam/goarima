# Optimisations layered on the Kalman inner loop

| Layer | What | Win |
|---|---|---|
| Joseph form (KAL-1) | Symmetric, PSD-preserving update vs the broken rank-1 `P -= K·row0` | Numerical stability near unit circle |
| Hoist `coef = K[i]·F − row0[i]` (PG-113) | 2-AXPY shape, FMA-friendly inner | ~10% on Joseph |
| Companion-shift fusion (PG-113) | TP intermediate eliminated; predict step is `P[i+1,k+1]` shift + phi-broadcast | 2.25× on r=27 case |
| Upper-triangle storage (PG-113) | P symmetric — only j ≥ i computed; `K[i] = P[0,i]/F` by symmetry | ~half the writes |
| Pooled `kalmanWorkspace` (KAL-WORKSPACE) | 9 buffers reused across calls | zero allocs in the hot loop |
| Sparse `zRow` in diffuse path (CDX-3) | `kalman_full.go` matvecs iterate only nonzero z-entries (typically 2-3 of rd=27) | −20% on `FitNonSimple_Airline` |
| Smith doubling for stationary cov (GARD-OPT-1) | O(r³) Lyapunov solver replaces inclu2's O(r⁴) for AR-containing models with r ≥ 20 | −25% on weekly seasonal AutoArima |

Each is documented under its own ID — search the codebase comments
for `PG-113`, `KAL-1`, `KAL-WORKSPACE`, `CDX-3`, `GARD-OPT-1`.
