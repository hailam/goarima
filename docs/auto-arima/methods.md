# Estimator methods

| Method | What | When you'd pick it |
|---|---|---|
| `MethodCSS` | Conditional sum-of-squares | Warm-start; Approximation; not for final estimates |
| `MethodML` | Pure Kalman ML | Want ML directly without warm-start |
| `MethodCSSML` | CSS warm-start → ML refine | **Default.** Best accuracy and conditioning |

GAP-1 (2026-05-05) reordered the iota so `Method(0) == MethodCSSML`
matches the documented default. See
[`../policies/method-default.md`](../policies/method-default.md).

For the per-Method Kalman path, see [`../kalman/dispatch.md`](../kalman/dispatch.md).
