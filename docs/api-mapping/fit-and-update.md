# Fit, Update, Refit, Predict

| Concern | pmdarima | R | goarima |
|---|---|---|---|
| Confidence interval | `alpha=0.05` (kwarg) | `level=95` | `Predict(n, alpha, futureExog)`. Pass `alpha=0` to skip CIs (returns `nil` for `lower` / `upper`). |
| Update / refresh | `model.update(y, X)` warm-starts MLE on existing params | `Arima(model = existing, x = new_y, xreg = new_X)` warm-starts | `m.Update(y, x)` warm-starts (fast); `m.Refit(y, x)` does a full cold re-fit. Neither re-searches orders — call `AutoArima` fresh for that. |
| Estimator method | `method='lbfgs'/'css-mle'` | `method='CSS'/'ML'/'CSS-ML'` | `Method` enum: `MethodCSS`, `MethodML`, `MethodCSSML` (default — same as R). See [`../policies/method-default.md`](../policies/method-default.md). |
| Forecast variance for integrated models | grows with horizon ✓ | grows with horizon ✓ | `Predict` CI bands grow correctly (cumulative-sum ψ for each unit-root factor). |
