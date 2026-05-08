# Versioning

## Public API

Pre-1.0 we reserve the right to break things, but breaking changes
are recorded in commit messages with `!` (e.g.
`feat(arima)!: …`). Search `git log --oneline | grep '!'` for the
running list.

## Saved-model JSON

`m.Save(io.Writer)` and `arima.LoadARIMA(io.Reader)` write versioned
JSON. The version field in `arima/serialize.go` lets old saves keep
loading after refactors.

`Method` is JSON-serialised as a *string* via the `methodToString`
map, so reordering the iota (as GAP-1 did) doesn't invalidate
previously-saved models — see
[`method-default.md`](method-default.md).

## Dataset attributions

Bare MIT in `LICENSE`; per-dataset attributions in `NOTICE`. Built-in
series come from public/government sources — see the per-dataset Go
files and the `NOTICE` summary.
