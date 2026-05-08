# Differencing pipeline

`d` (non-seasonal) and `D` (seasonal) are picked by separate tests
*before* any model is fit. Defaults match R `forecast::auto.arima`.

| Diff | Default test | Switch | Notes |
|---|---|---|---|
| `d` | KPSS | `Test = NDiffsKPSS / NDiffsADF / NDiffsPP` | KPSS tests stationarity (large stat ⇒ diff); ADF/PP test unit roots (small stat ⇒ diff). |
| `D` | SEAS | `SeasonalTest = NSDiffsSEAS / NSDiffsOCSB / NSDiffsHEGY / NSDiffsCH` | SEAS (Wang-Smith-Hyndman seasonal-strength) is R `auto.arima`'s default. OCSB matches pmdarima's default. HEGY exposes per-frequency stats. |

**Behaviour change 2026-05-07**: the seasonal-test default was OCSB
prior; it's now SEAS to match R. Picks may differ on series where
SEAS and OCSB disagree (notably daily M5-shape). Set
`SeasonalTest: NSDiffsOCSB` to restore the old behaviour.

Each test returns a count (0/1/2) capped at `MaxD` / `MaxCapD`.
Differencing is then applied; the search runs on the differenced
series.

## HEGY

HEGY has the deepest API surface — `HEGYTest` returns just the verdict;
`HEGYTestFull(x, m, opts)` returns per-frequency t-stats and pair
F-stats with p-values. 60 response-surface tables ported from
uroot 2.1.3 cover all (deterministic, lag.method) combos. PG-114
shipped this; PG-115 exposed the per-frequency view.
