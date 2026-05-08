# Joseph-form Kalman update (KAL-1)

The covariance update in `arima/kalman.go` is the *Joseph* form:

```
P[i,j] += -K[i]·row0[j] + (K[i]·F − row0[i])·K[j]
```

not the simpler rank-1 `P -= K·row0`. They're algebraically
equivalent in exact arithmetic, but the rank-1 form loses
symmetry/PSD over many steps once BFGS pushes φ near the unit
circle — F = P[0,0] then drifts wildly, the Kalman gain explodes,
and the likelihood is garbage.

Joseph trades ~30% extra inner-loop work for "doesn't blow up". The
hoisted-coef variant (PG-113) keeps the Joseph algebra but
restructures it as 2-AXPY for FMA.

KAL-1 probe tests (`arima/kalman_logf_probe_test.go`) pin
`sum(log F)` into the expected band so any future regression is
caught.
