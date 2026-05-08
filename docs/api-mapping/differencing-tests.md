# Differencing-test mapping

| Concern | pmdarima | R | goarima |
|---|---|---|---|
| Seasonal differencing test | `nsdiffs(test='ocsb')` (default) | `nsdiffs(test='seas')` (default in `auto.arima`); `'ocsb'` and `'hegy'` available | `AutoArimaOpts.SeasonalTest` defaults to `NSDiffsSEAS` (matches R `auto.arima`); set to `NSDiffsOCSB` for pmdarima's default; `NSDiffsHEGY` for full hegy.test parity |
| Non-seasonal differencing test | `ndiffs(test='kpss')` (default) | `ndiffs(test='kpss')` (default) | `AutoArimaOpts.Test` defaults to `NDiffsKPSS`; alternatives `NDiffsADF`, `NDiffsPP` |

For per-frequency HEGY p-values use `HEGYTestFull(x, m, opts)` —
returns t-stats for π_1 / Nyquist, pair F-stats for each harmonic,
and the joint stats. Backed by 60 response-surface tables ported
from uroot 2.1.3 (PG-114).

Pipeline-level explanation: [`../auto-arima/differencing.md`](../auto-arima/differencing.md).
