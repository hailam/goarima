# Save / load

| Concern | pmdarima | R | goarima |
|---|---|---|---|
| Save / load | pickle | `saveRDS` / `readRDS` | `m.Save(io.Writer)` / `arima.LoadARIMA(io.Reader)` write versioned JSON. `*ARIMA` also implements `json.Marshaler` / `json.Unmarshaler`. |

Saved-model JSON is forward-compatible with iota reorders (e.g.
GAP-1 reordering `Method`) — the version field handles
deserialisation. See [`../policies/versioning.md`](../policies/versioning.md).
