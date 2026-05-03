package pipeline

import (
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/hailam/goarima/arima"
	"github.com/hailam/goarima/preprocessing"
)

// PipelineDateFeaturizer should plug into the pipeline as an exog step:
// derive day-of-week / day-of-month features from a date index at Fit
// time, then auto-extend the date index forward at Predict time so the
// caller doesn't need to pass futureExog manually.
func TestPipelineDateFeaturizerEndToEnd(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 32))
	n := 120
	start := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	dates := make([]time.Time, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		dates[i] = start.AddDate(0, 0, i)
		// Add a weekly seasonal effect (weekend bump) plus noise.
		weekend := 0.0
		if dates[i].Weekday() == time.Saturday || dates[i].Weekday() == time.Sunday {
			weekend = 5.0
		}
		var prev float64
		if i > 0 {
			prev = y[i-1]
		}
		y[i] = 0.4*prev + weekend + rng.NormFloat64()
	}

	feat := preprocessing.NewPipelineDateFeaturizer(dates, preprocessing.DailyStep)
	model := arima.NewARIMA(arima.Order{P: 1, D: 0, Q: 0})
	model.WithIntercept = true
	pl, err := NewPipeline([]Step{
		{Name: "dates", Exog: feat},
	}, model)
	if err != nil {
		t.Fatal(err)
	}

	if err := pl.Fit(y, nil); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	// Predict 14 days ahead with no manually-supplied futureExog — the
	// featurizer should auto-extend the date index by Step and emit the
	// matching exog rows.
	fc, _, _, err := pl.Predict(14, 0, nil)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if len(fc) != 14 {
		t.Fatalf("got %d forecasts, want 14", len(fc))
	}
	for i, v := range fc {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("fc[%d] = %v (NaN/Inf)", i, v)
		}
	}
}

// Validate that Fit rejects mismatched date / y lengths.
func TestPipelineDateFeaturizerFitMismatch(t *testing.T) {
	dates := []time.Time{
		time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
		time.Date(2024, 1, 2, 0, 0, 0, 0, time.UTC),
	}
	feat := preprocessing.NewPipelineDateFeaturizer(dates, preprocessing.DailyStep)
	if err := feat.Fit([]float64{1, 2, 3}); err == nil {
		t.Error("expected length-mismatch error")
	}
}

// Step helpers should produce the expected calendar increments.
func TestDateStepHelpers(t *testing.T) {
	t0 := time.Date(2024, 1, 31, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		name string
		step preprocessing.DateStep
		want time.Time
	}{
		{"Daily", preprocessing.DailyStep, t0.AddDate(0, 0, 1)},
		{"Weekly", preprocessing.WeeklyStep, t0.AddDate(0, 0, 7)},
		{"Monthly", preprocessing.MonthlyStep, time.Date(2024, 3, 2, 12, 0, 0, 0, time.UTC)}, // Jan 31 + 1 month → Mar 2 (Go's normalisation)
		{"Quarterly", preprocessing.QuarterlyStep, t0.AddDate(0, 3, 0)},
		{"Yearly", preprocessing.YearlyStep, t0.AddDate(1, 0, 0)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.step(t0)
			if !got.Equal(tc.want) {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}
