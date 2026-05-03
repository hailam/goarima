package preprocessing

import (
	"errors"
	"fmt"
	"time"
)

// DateFeaturizer derives day-of-week (one-hot) and day-of-month (ordinal)
// features from a time.Time per row. Mirrors pmdarima.preprocessing.DateFeaturizer.
type DateFeaturizer struct {
	WithDayOfWeek  bool   // emits 7 dummy columns (Mon..Sun)
	WithDayOfMonth bool   // emits 1 ordinal column (1..31)
	Prefix         string // column-name prefix for FeatureNames; defaults to "DATE"

	fitted bool
}

// NewDateFeaturizer creates a featurizer with both flags enabled.
func NewDateFeaturizer() *DateFeaturizer {
	return &DateFeaturizer{WithDayOfWeek: true, WithDayOfMonth: true}
}

// Fit is a no-op (validates parameters).
func (f *DateFeaturizer) Fit(_ []float64) error {
	if !f.WithDayOfWeek && !f.WithDayOfMonth {
		return errors.New("DateFeaturizer must enable at least one feature")
	}
	f.fitted = true
	return nil
}

// Transform builds the date-feature matrix for the supplied dates.
// Each output row corresponds to one date in the same order; if existing
// is supplied, the new columns are concatenated to its right.
//
// nPeriods is unused (dates are explicit); kept for interface symmetry.
func (f *DateFeaturizer) Transform(dates []time.Time, existing [][]float64) ([][]float64, error) {
	if !f.fitted {
		return nil, errors.New("transformer not fitted")
	}
	if len(dates) == 0 {
		return nil, errors.New("dates must be non-empty")
	}
	if existing != nil && len(existing) != len(dates) {
		return nil, fmt.Errorf("existing rows (%d) != len(dates) (%d)", len(existing), len(dates))
	}

	n := len(dates)
	out := make([][]float64, n)
	for i, d := range dates {
		extra := f.dateColumns(d)
		var row []float64
		if existing != nil {
			row = make([]float64, len(existing[i])+len(extra))
			copy(row, existing[i])
			copy(row[len(existing[i]):], extra)
		} else {
			row = extra
		}
		out[i] = row
	}
	return out, nil
}

// FeatureNames returns the new column names appended by this featurizer.
func (f *DateFeaturizer) FeatureNames() []string {
	pfx := f.Prefix
	if pfx == "" {
		pfx = "DATE"
	}
	var out []string
	if f.WithDayOfWeek {
		// 0 = Monday in pmdarima (matches Python weekday()).
		for i := 0; i < 7; i++ {
			out = append(out, fmt.Sprintf("%s-WEEKDAY-%d", pfx, i))
		}
	}
	if f.WithDayOfMonth {
		out = append(out, fmt.Sprintf("%s-DAY-OF-MONTH", pfx))
	}
	return out
}

// DateStep returns the date for period n+1 given the date at period n.
// Used by PipelineDateFeaturizer to extend the date index forward when the
// pipeline asks for nPeriods of future exog rows.
type DateStep func(prev time.Time) time.Time

// Common date-step helpers covering pmdarima's typical frequency cases.
var (
	DailyStep     DateStep = func(t time.Time) time.Time { return t.AddDate(0, 0, 1) }
	WeeklyStep    DateStep = func(t time.Time) time.Time { return t.AddDate(0, 0, 7) }
	MonthlyStep   DateStep = func(t time.Time) time.Time { return t.AddDate(0, 1, 0) }
	QuarterlyStep DateStep = func(t time.Time) time.Time { return t.AddDate(0, 3, 0) }
	YearlyStep    DateStep = func(t time.Time) time.Time { return t.AddDate(1, 0, 0) }
)

// PipelineDateFeaturizer adapts DateFeaturizer to the pipeline's
// ExogFeaturizer interface. It owns the date series the model is being fit
// against and a Step function describing the sampling frequency.
//
// Construction:
//
//	p := preprocessing.NewPipelineDateFeaturizer(dates, preprocessing.MonthlyStep)
//	p.Inner.WithDayOfWeek = false  // optional: customise inner featurizer
//	step := pipeline.Step{Name: "dates", Exog: p}
//
// During Fit, len(dates) must match len(y). Predict's nPeriods argument
// drives forward-extension via Step.
type PipelineDateFeaturizer struct {
	Inner DateFeaturizer

	// FitDates holds the timestamp for each y observation at Fit time.
	// Required; len(FitDates) must equal len(y) when Fit is called.
	FitDates []time.Time

	// Step describes how to advance one period forward. nil → DailyStep.
	Step DateStep

	fitted bool
}

// NewPipelineDateFeaturizer constructs a featurizer with both day-of-week
// and day-of-month features enabled, bound to the supplied date index and
// step function. Pass nil for step to default to daily.
func NewPipelineDateFeaturizer(dates []time.Time, step DateStep) *PipelineDateFeaturizer {
	if step == nil {
		step = DailyStep
	}
	return &PipelineDateFeaturizer{
		Inner:    DateFeaturizer{WithDayOfWeek: true, WithDayOfMonth: true},
		FitDates: dates,
		Step:     step,
	}
}

// Fit binds the inner featurizer and validates the date index lines up
// with y. Implements pipeline.ExogFeaturizer.
func (p *PipelineDateFeaturizer) Fit(y []float64) error {
	if len(p.FitDates) != len(y) {
		return fmt.Errorf("PipelineDateFeaturizer: FitDates len %d != y len %d",
			len(p.FitDates), len(y))
	}
	if p.Step == nil {
		p.Step = DailyStep
	}
	if err := p.Inner.Fit(y); err != nil {
		return err
	}
	p.fitted = true
	return nil
}

// Transform builds exog rows for the requested horizon by extending the
// fit-time date index forward via Step. When nPeriods is 0 it returns
// in-sample features for the existing FitDates instead — matching the
// pipeline's calling convention for "rebuild features from y".
//
// The optional `existing` (x in pipeline parlance) is concatenated on the
// right of the date columns, mirroring DateFeaturizer.Transform.
func (p *PipelineDateFeaturizer) Transform(y []float64, x [][]float64, nPeriods int) ([][]float64, error) {
	if !p.fitted {
		return nil, errors.New("PipelineDateFeaturizer: not fitted")
	}
	if nPeriods <= 0 {
		// In-sample / featurize-with-existing-dates path.
		dates := p.FitDates
		if len(y) != len(dates) {
			return nil, fmt.Errorf("PipelineDateFeaturizer: y len %d != FitDates len %d",
				len(y), len(dates))
		}
		return p.Inner.Transform(dates, x)
	}
	if len(p.FitDates) == 0 {
		return nil, errors.New("PipelineDateFeaturizer: no FitDates; cannot extend")
	}
	future := make([]time.Time, nPeriods)
	cur := p.FitDates[len(p.FitDates)-1]
	for i := 0; i < nPeriods; i++ {
		cur = p.Step(cur)
		future[i] = cur
	}
	return p.Inner.Transform(future, x)
}

// UpdateAndTransform handles the case where new observations have been
// appended to y (pipeline.Update flow). New dates are auto-generated via
// Step and appended to FitDates so subsequent Transform calls extend from
// the new tail.
func (p *PipelineDateFeaturizer) UpdateAndTransform(y []float64, x [][]float64) ([][]float64, error) {
	if !p.fitted {
		return nil, errors.New("PipelineDateFeaturizer: not fitted")
	}
	nNew := len(y) - len(p.FitDates)
	if nNew < 0 {
		return nil, fmt.Errorf("PipelineDateFeaturizer: y shrank (%d → %d)",
			len(p.FitDates), len(y))
	}
	if nNew > 0 {
		cur := p.FitDates[len(p.FitDates)-1]
		for i := 0; i < nNew; i++ {
			cur = p.Step(cur)
			p.FitDates = append(p.FitDates, cur)
		}
	}
	return p.Inner.Transform(p.FitDates, x)
}

// dateColumns returns one row of derived features.
//
// Day-of-week index follows Python convention (Monday=0, Sunday=6), which
// matches pmdarima's `pd.Timestamp.weekday()`. Go's time.Weekday has Sunday=0.
func (f *DateFeaturizer) dateColumns(d time.Time) []float64 {
	var row []float64
	if f.WithDayOfWeek {
		ww := make([]float64, 7)
		// Convert Go's weekday (Sun=0..Sat=6) to Python's (Mon=0..Sun=6).
		gw := int(d.Weekday())
		py := (gw + 6) % 7
		ww[py] = 1
		row = append(row, ww...)
	}
	if f.WithDayOfMonth {
		row = append(row, float64(d.Day()))
	}
	return row
}
