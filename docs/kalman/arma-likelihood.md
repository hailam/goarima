# `kalmanARMALikelihoodInto` — the default hot path

Lives at `arima/kalman.go:kalmanARMALikelihoodInto`. Hamilton form;
state dim r = max(p, q+1). Per timestep:

1. Innovation `v = y[t] − a[0]`; `F = P[0,0]`; gain `K = P[:,0] / F`
2. **Joseph P-update**:
   `P[i,j] += -K[i]·row0[j] + (K[i]·F − row0[i])·K[j]`
3. **Predict** via T's companion structure:
   `newP[i,k] = P[i+1, k+1] + RRt[i,k] + phi-corrections`

Initial state covariance `P_0` from `stationaryCovGardnerInto` —
dispatch (Smith vs inclu2 vs pure-MA) at
[`../numerical-stability/gardner-dispatch.md`](../numerical-stability/gardner-dispatch.md).

Joseph form (vs the simpler rank-1 `P -= K·row0`) preserves
PSD/symmetry over many steps. See
[`../numerical-stability/joseph.md`](../numerical-stability/joseph.md)
for the KAL-1 history.

The companion-shift fusion + upper-triangle storage (PG-113) is what
makes this 2.25× faster than the previous sparse-T inner loop —
[`optimisations.md`](optimisations.md).
