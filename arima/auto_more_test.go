package arima

import (
	"errors"
	"math"
	"math/rand/v2"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/hailam/goarima/datasets"
)

// FullSearch must enumerate every combination and find at least the same or
// better IC than stepwise.
func TestAutoArimaFullSearch(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	y := make([]float64, 200)
	for i := 1; i < len(y); i++ {
		y[i] = 0.5*y[i-1] + rng.NormFloat64()
	}
	stepMdl, err := AutoArima(y, nil, AutoArimaOpts{
		MaxP: 3, MaxQ: 3, MaxOrder: 4, IC: AICc, MaxIter: 60,
	})
	if err != nil {
		t.Fatal(err)
	}
	fullMdl, err := AutoArima(y, nil, AutoArimaOpts{
		MaxP: 3, MaxQ: 3, MaxOrder: 4, IC: AICc, MaxIter: 60, FullSearch: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	// Full search visits at least as many combinations as stepwise — its IC
	// should be ≤ stepwise's (within a tiny tolerance for optimizer noise).
	if fullMdl.AICc() > stepMdl.AICc()+0.5 {
		t.Errorf("FullSearch IC=%v worse than stepwise=%v", fullMdl.AICc(), stepMdl.AICc())
	}
}

// FullSearch with NFits caps candidates and uses random sampling.
func TestAutoArimaNFits(t *testing.T) {
	y := datasets.LoadAusbeer()
	// drop trailing NaN
	y = y[:len(y)-1]
	mdl, err := AutoArima(y, nil, AutoArimaOpts{
		M: 4, MaxP: 3, MaxQ: 3, MaxCapP: 1, MaxCapQ: 1, MaxOrder: 4,
		IC: AICc, MaxIter: 30,
		FullSearch: true, NFits: 5, Seed: 42,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mdl == nil {
		t.Fatal("nil model")
	}
}

// Trace receives one line per candidate.
func TestAutoArimaTrace(t *testing.T) {
	y := simulateAR1(150, 0.4, 1.0, 1)
	var lines []string
	_, err := AutoArima(y, nil, AutoArimaOpts{
		MaxP: 2, MaxQ: 2, MaxOrder: 3, IC: AIC, MaxIter: 30,
		FullSearch: true, // ensures multiple candidates
		Trace:      func(s string) { lines = append(lines, s) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(lines) == 0 {
		t.Error("trace received no lines")
	}
	for _, l := range lines {
		if !strings.Contains(l, "ARIMA") {
			t.Errorf("trace line missing ARIMA: %q", l)
		}
	}
}

// out_of_sample_size scoring uses SMAPE on the holdout.
func TestAutoArimaOutOfSample(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	y := make([]float64, 200)
	for i := 1; i < len(y); i++ {
		y[i] = 0.6*y[i-1] + rng.NormFloat64()
	}
	mdl, err := AutoArima(y, nil, AutoArimaOpts{
		MaxP: 2, MaxQ: 2, MaxOrder: 3, IC: AIC, MaxIter: 30,
		OutOfSampleSize: 20,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mdl == nil {
		t.Fatal("nil model")
	}
	// Model should still be a valid AR-ish fit after holdout-based selection.
	if mdl.Order.P > 2 {
		t.Errorf("p=%d should be <=2", mdl.Order.P)
	}
}

// error_action=ignore: a candidate that fails should not abort the whole run.
func TestAutoArimaErrorActionIgnore(t *testing.T) {
	y := simulateAR1(80, 0.5, 1.0, 1)
	mdl, err := AutoArima(y, nil, AutoArimaOpts{
		MaxP: 5, MaxQ: 5, MaxOrder: 8, IC: AICc, MaxIter: 20,
		ErrorAction: "ignore", FullSearch: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if mdl == nil {
		t.Fatal("nil model")
	}
}

// Codex #C12: ErrorAction="raise" (default) must propagate fit errors even
// when other candidates succeed. Pre-fix the FullSearch path returned
// (best != nil, err == nil) when the best candidate happened to succeed,
// silently swallowing failures from sibling candidates.
//
// We can't easily inject a fit failure (every well-formed candidate succeeds
// on synthetic data), so we test the error-PROPAGATION wiring directly:
// a custom Scoring callback that returns an error on the first call. With
// "raise", the search must surface that error; with "ignore", it must not.
func TestAutoArimaErrorActionRaisePropagates(t *testing.T) {
	y := simulateAR1(150, 0.5, 1.0, 3)
	scoringFails := func(yt, yp []float64) (float64, error) {
		return 0, errors.New("synthetic scoring error")
	}
	// FullSearch path
	_, err := AutoArima(y, nil, AutoArimaOpts{
		MaxP: 1, MaxQ: 1, MaxOrder: 2, IC: AICc, MaxIter: 20,
		OutOfSampleSize: 10, Scoring: scoringFails,
		ErrorAction: "raise", FullSearch: true,
	})
	if err == nil {
		t.Error("FullSearch + raise: expected error to propagate, got nil")
	}
	// Stepwise path
	_, err = AutoArima(y, nil, AutoArimaOpts{
		MaxP: 1, MaxQ: 1, MaxOrder: 2, IC: AICc, MaxIter: 20,
		OutOfSampleSize: 10, Scoring: scoringFails,
		ErrorAction: "raise",
	})
	if err == nil {
		t.Error("Stepwise + raise: expected error to propagate, got nil")
	}
	// Same setup with ignore: must NOT propagate the synthetic scoring
	// error. (Search may still error with "no candidate succeeded" since
	// every candidate's scoring fails, but the error must not be the
	// synthetic-scoring one.)
	_, err = AutoArima(y, nil, AutoArimaOpts{
		MaxP: 1, MaxQ: 1, MaxOrder: 2, IC: AICc, MaxIter: 20,
		OutOfSampleSize: 10, Scoring: scoringFails,
		ErrorAction: "ignore", FullSearch: true,
	})
	if err != nil && strings.Contains(err.Error(), "synthetic scoring error") {
		t.Errorf("ignore mode propagated scoring error: %v", err)
	}
}

// Custom scoring function.
func TestAutoArimaCustomScoring(t *testing.T) {
	y := simulateAR1(150, 0.4, 1.0, 1)
	// AutoArima may invoke Scoring concurrently across neighbor candidates
	// in stepwise mode (and across the search box in FullSearch mode), so
	// any state captured in the closure must be safe for concurrent access.
	var called atomic.Bool
	scoring := func(yt, yp []float64) (float64, error) {
		called.Store(true)
		if len(yt) != len(yp) {
			return 0, errors.New("length mismatch")
		}
		s := 0.0
		for i := range yt {
			d := yt[i] - yp[i]
			s += math.Abs(d)
		}
		return s / float64(len(yt)), nil
	}
	mdl, err := AutoArima(y, nil, AutoArimaOpts{
		MaxP: 2, MaxQ: 2, MaxOrder: 3, IC: AIC, MaxIter: 30,
		OutOfSampleSize: 15,
		Scoring:         scoring,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !called.Load() {
		t.Error("custom scorer was never called")
	}
	if mdl == nil {
		t.Fatal("nil model")
	}
}

// MaxSteps caps the stepwise iterations.
func TestAutoArimaMaxSteps(t *testing.T) {
	y := simulateAR1(100, 0.3, 1.0, 1)
	mdl, err := AutoArima(y, nil, AutoArimaOpts{
		MaxP: 3, MaxQ: 3, MaxOrder: 4, IC: AIC, MaxIter: 30,
		MaxSteps: 1, // very tight
	})
	if err != nil {
		t.Fatal(err)
	}
	if mdl == nil {
		t.Fatal("nil model")
	}
}
