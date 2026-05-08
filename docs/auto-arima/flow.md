# End-to-end flow

```mermaid
flowchart TD
    A[AutoArima(y, exog, opts)] --> B[FindFrequency if M=0]
    B --> C[NDiffs - non-seasonal d via KPSS/ADF/PP]
    C --> D[NSDiffs - seasonal D via SEAS/OCSB/HEGY/CH]
    D --> E[apply differencing to y]
    E --> F{search mode}
    F -->|stepwise| G[stepwise neighbour walk]
    F -->|full| H[parallel cartesian fits]
    G --> I[pick best AICc]
    H --> I
    I --> J{Approximation?}
    J -->|true| K[refit at user Method]
    J -->|false| L[done]
    K --> L
```

Top-level entry: `arima/auto.go:AutoArima`. Differencing pipeline:
`arima/seasonality.go:NDiffs` and `:NSDiffs`. See
[`differencing.md`](differencing.md) for which test fires when.
