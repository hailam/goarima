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
