# Approximation — R-style two-stage search

`AutoArimaOpts.Approximation = true` runs the candidate search in
CSS (cheap), then refits the picked order at the user's actual
Method (default CSSML).

Mirrors R's `approximation=TRUE`, which R sets by default when
n>150 or m>12. Off by default in goarima so zero-config behaviour
is "always use the documented Method" — explicit opt-in matches the
documented Method (no surprise speedups that change estimates).

| Bench (AirPassengers, MaxIter=50) | Wall |
|---|---:|
| Default (full CSSML stepwise) | 75 ms |
| `Approximation=true` (CSS search + CSSML refit) | 46 ms |

GAP-2 (2026-05-05) is the underlying implementation; the option
was missing prior to that.
