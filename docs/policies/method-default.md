# Method default — why CSSML

GAP-1 (2026-05-05) reordered the `Method` iota so the zero value is
`MethodCSSML`, not `MethodCSS`. Pre-fix, `AutoArimaOpts{}` silently
gave the fast-but-biased CSS estimator, then users compared AICc
against R/pmdarima and got mysteriously different picks. Now the
zero value matches R's default — slower but correct.

| Method | Default? | When to override |
|---|---|---|
| `MethodCSSML` | yes | — |
| `MethodCSS` | no | Want speed and don't need ML-grade estimates (screening many series) |
| `MethodML` | no | Have a specific reason to skip the CSS warm-start |

For the per-Method Kalman path, see
[`../kalman/dispatch.md`](../kalman/dispatch.md).
