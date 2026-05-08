# Orders and field naming

| Concern | pmdarima | R | goarima |
|---|---|---|---|
| AR/MA/seasonal orders | `order=(p,d,q), seasonal_order=(P,D,Q,m)` | `order=c(p,d,q), seasonal=list(order=c(P,D,Q), period=m)` | `Order{P,D,Q}` (non-seasonal) and `Seasonal{P,D,Q,M}` |

**`Order.P` is the non-seasonal AR order** (lowercase `p` in the
math); `Seasonal.P` is the seasonal AR order. The duplicate
field name is forced by Go visibility rules — every exported field
must start with an uppercase letter.

| Concern | pmdarima | R | goarima |
|---|---|---|---|
| Fitted values | `predict_in_sample(...)` returns `len(y)` array | `fitted(model)` returns `ts` of length `n` (NA in warmup) | `m.FittedValues()` returns `len(yTrain)` slice with `math.NaN()` in the first `d + D·m` warmup entries |
| Residuals | `arima_res_.resid` length `len(y)` | `residuals(model)` length `n` (NA in warmup) | `m.Resid()` returns `len(yTrain)` slice with `math.NaN()` in warmup |

Strip warmup NaNs before passing to ACF / Ljung-Box — neither is
NaN-aware. Users do this with a simple `math.IsNaN` filter pass on
the slice.
