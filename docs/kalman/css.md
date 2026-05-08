# `armaCSS` — the warm-start path

Conditional sum-of-squares: a plain difference-equation forward-pass
on the centred series. Profile likelihood, not Gaussian. Fast (no
state covariance) but biased on small n.

Lives at `arima/kalman.go:armaCSS`.

## When it fires

- Standalone via `Method=MethodCSS` — when speed beats statistical
  rigour (screening many series).
- As a warm-start before ML under `Method=MethodCSSML` (the default).
- During the candidate search when `AutoArimaOpts.Approximation = true`,
  with a final CSSML refit at the picked order — see
  [`../policies/approximation.md`](../policies/approximation.md).

## What it returns

`(negLogLik, sigma2, residuals)`. The `negLogLik` is on a different
scale than the Gaussian Kalman likelihood (CSS drops the constant
term and uses `(n/2) log(SSE/n)`), so don't compare CSS AICc directly
to ML AICc — refit at the same Method first.
