package preprocessing

import (
	"testing"
	"time"
)

// 2022-01-03 was a Monday → weekday=0; day of month = 3.
func TestDateFeaturizerBasic(t *testing.T) {
	f := NewDateFeaturizer()
	if err := f.Fit(nil); err != nil {
		t.Fatal(err)
	}
	dates := []time.Time{
		time.Date(2022, 1, 3, 0, 0, 0, 0, time.UTC),  // Monday
		time.Date(2022, 1, 4, 0, 0, 0, 0, time.UTC),  // Tuesday
		time.Date(2022, 1, 9, 0, 0, 0, 0, time.UTC),  // Sunday
		time.Date(2022, 1, 31, 0, 0, 0, 0, time.UTC), // Monday, day=31
	}
	out, err := f.Transform(dates, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 4 {
		t.Fatalf("len=%d", len(out))
	}
	// Each row: 7 weekday dummies + 1 day-of-month = 8 columns
	for i, row := range out {
		if len(row) != 8 {
			t.Errorf("row %d cols=%d want 8", i, len(row))
		}
	}
	// Monday → index 0
	if out[0][0] != 1 {
		t.Errorf("Monday weekday[0]=%v want 1", out[0][0])
	}
	if out[0][7] != 3 {
		t.Errorf("Day-of-month[0]=%v want 3", out[0][7])
	}
	// Tuesday → index 1
	if out[1][1] != 1 {
		t.Errorf("Tuesday weekday[1]=%v want 1", out[1][1])
	}
	// Sunday → index 6
	if out[2][6] != 1 {
		t.Errorf("Sunday weekday[6]=%v want 1", out[2][6])
	}
	if out[3][7] != 31 {
		t.Errorf("Day-of-month[3]=%v want 31", out[3][7])
	}
}

func TestDateFeaturizerHstack(t *testing.T) {
	f := &DateFeaturizer{WithDayOfMonth: true}
	if err := f.Fit(nil); err != nil {
		t.Fatal(err)
	}
	dates := []time.Time{time.Date(2022, 6, 15, 0, 0, 0, 0, time.UTC)}
	existing := [][]float64{{0.1, 0.2}}
	out, err := f.Transform(dates, existing)
	if err != nil {
		t.Fatal(err)
	}
	if len(out[0]) != 3 || out[0][0] != 0.1 || out[0][2] != 15 {
		t.Errorf("hstack got %v want [0.1 0.2 15]", out[0])
	}
}

func TestDateFeaturizerErrors(t *testing.T) {
	f := &DateFeaturizer{}
	if err := f.Fit(nil); err == nil {
		t.Error("expected error: both flags off")
	}
	f = NewDateFeaturizer()
	if _, err := f.Transform(nil, nil); err == nil {
		t.Error("expected error: not fitted")
	}
	_ = f.Fit(nil)
	if _, err := f.Transform(nil, nil); err == nil {
		t.Error("expected error: empty dates")
	}
}

func TestDateFeaturizerNames(t *testing.T) {
	f := NewDateFeaturizer()
	names := f.FeatureNames()
	want := []string{
		"DATE-WEEKDAY-0", "DATE-WEEKDAY-1", "DATE-WEEKDAY-2", "DATE-WEEKDAY-3",
		"DATE-WEEKDAY-4", "DATE-WEEKDAY-5", "DATE-WEEKDAY-6", "DATE-DAY-OF-MONTH",
	}
	if len(names) != len(want) {
		t.Fatalf("got %v want %v", names, want)
	}
	for i, n := range names {
		if n != want[i] {
			t.Errorf("name[%d]=%v want %v", i, n, want[i])
		}
	}
}
