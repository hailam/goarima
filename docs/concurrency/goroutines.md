# Where goroutines fire

| Site | What's parallel | When it pays off | Knob |
|---|---|---|---|
| `parallelGradient` (`arima/arima.go`) | Finite-difference gradient — one worker per parameter coordinate | Always, for ≥2 parameters; serial otherwise | `ARIMA.GradientWorkers`, `AutoArimaOpts.GradientWorkers` |
| `AutoArima` full search (`arima/auto.go`) | Candidate (Order, Seasonal) fits | When `FullSearch=true` | `AutoArimaOpts.NJobs` |
| `AutoArima` stepwise (`arima/auto.go`) | Neighbour fits at each step | Default-on; sequential if `NJobs=1` | `AutoArimaOpts.NJobs` |
| `PredictBoot` (`arima/bootstrap.go`) | Per-simulation forecast paths | Always when `nSim ≥ ~few hundred` | `numWorkers` arg |
| `CrossVal*` (`modelselection/`) | CV folds | Always | implicit (one goroutine per fold) |

PG-91 (the parallelGradient determinism fix) verified empirically that
parallel speedup on AutoArima is near-linear in worker count up to
NumCPU.
