# Divergence-decision policy

When R `forecast::auto.arima`, pmdarima, and goarima disagree:

1. **R first.** R's `forecast::auto.arima` is the canonical
   Hyndman-Khandakar reference. Behaviour aligns with R unless
   empirical evidence on canonical datasets shows R is genuinely
   worse.
2. **pmdarima second.** pmdarima models its API on R closely; we
   use it as a tiebreaker reference and to keep the option / field
   names familiar to Python users. We do not chase pmdarima
   compatibility when it conflicts with R.
3. **Empirical winner third.** When goarima's behaviour beats R on
   the canonical `threeway-tests-goarima` grid (verified AICc / fit
   diagnostics), we keep the divergence and document it.

Currently-shipped divergences from R are tracked internally as
PG-92 through PG-100 — search the codebase comments for those IDs.
