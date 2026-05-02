package modelselection

import (
	"reflect"
	"testing"
)

// Mirrors the docstring example for RollingForecastCV (default h=1, step=1).
// wineind has 176 obs; first split: train=[0..57], test=[58].
func TestRollingForecastCVDefault(t *testing.T) {
	cv, err := NewRollingForecastCV(1, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	splits, err := cv.Split(176)
	if err != nil {
		t.Fatal(err)
	}
	if len(splits) == 0 {
		t.Fatal("no splits")
	}
	wantTrain0 := makeRange(0, 58)
	wantTest0 := []int{58}
	if !reflect.DeepEqual(splits[0].Train, wantTrain0) {
		t.Errorf("split[0].Train front=%v back=%v", splits[0].Train[:5], splits[0].Train[len(splits[0].Train)-3:])
	}
	if !reflect.DeepEqual(splits[0].Test, wantTest0) {
		t.Errorf("split[0].Test = %v want %v", splits[0].Test, wantTest0)
	}

	// next fold: train=[0..58], test=[59]
	if !reflect.DeepEqual(splits[1].Test, []int{59}) {
		t.Errorf("split[1].Test = %v", splits[1].Test)
	}
	if splits[1].Train[len(splits[1].Train)-1] != 58 {
		t.Errorf("split[1].Train last = %d", splits[1].Train[len(splits[1].Train)-1])
	}
}

// Mirrors RollingForecastCV(step=2, h=4) example.
func TestRollingForecastCVStepHorizon(t *testing.T) {
	cv, err := NewRollingForecastCV(4, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	splits, err := cv.Split(176)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(splits[0].Test, []int{58, 59, 60, 61}) {
		t.Errorf("split[0].Test = %v", splits[0].Test)
	}
	if !reflect.DeepEqual(splits[1].Test, []int{60, 61, 62, 63}) {
		t.Errorf("split[1].Test = %v", splits[1].Test)
	}
}

// Mirrors SlidingWindowForecastCV() default example: window_size = max(3, n/5)
// wineind (n=176) -> 35; first split train=[0..34], test=[35].
func TestSlidingWindowDefault(t *testing.T) {
	cv, err := NewSlidingWindowForecastCV(1, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	splits, err := cv.Split(176)
	if err != nil {
		t.Fatal(err)
	}
	wantTrain0 := makeRange(0, 35)
	if !reflect.DeepEqual(splits[0].Train, wantTrain0) {
		t.Errorf("Train[0] mismatch")
	}
	if !reflect.DeepEqual(splits[0].Test, []int{35}) {
		t.Errorf("Test[0] = %v", splits[0].Test)
	}
	// next: shift by 1
	if !reflect.DeepEqual(splits[1].Test, []int{36}) {
		t.Errorf("Test[1] = %v", splits[1].Test)
	}
}

// Mirrors SlidingWindowForecastCV(step=4, h=6, window_size=12).
func TestSlidingWindowParams(t *testing.T) {
	cv, err := NewSlidingWindowForecastCV(6, 4, 12)
	if err != nil {
		t.Fatal(err)
	}
	splits, err := cv.Split(176)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(splits[0].Train, makeRange(0, 12)) ||
		!reflect.DeepEqual(splits[0].Test, []int{12, 13, 14, 15, 16, 17}) {
		t.Errorf("split[0] mismatch: %v / %v", splits[0].Train, splits[0].Test)
	}
	if !reflect.DeepEqual(splits[1].Train, makeRange(4, 16)) ||
		!reflect.DeepEqual(splits[1].Test, []int{16, 17, 18, 19, 20, 21}) {
		t.Errorf("split[1] mismatch: %v / %v", splits[1].Train, splits[1].Test)
	}
}

func TestTrainTestSplitInt(t *testing.T) {
	tr, te, err := TrainTestSplit(100, 50, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(tr) != 50 || len(te) != 50 {
		t.Errorf("got tr=%d te=%d", len(tr), len(te))
	}
	if tr[0] != 0 || tr[49] != 49 || te[0] != 50 || te[49] != 99 {
		t.Errorf("indices wrong: tr=%v..%v te=%v..%v", tr[0], tr[49], te[0], te[49])
	}
}

func TestTrainTestSplitFrac(t *testing.T) {
	tr, te, err := TrainTestSplit(100, 0, 0.25, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(tr) != 75 || len(te) != 25 {
		t.Errorf("got tr=%d te=%d", len(tr), len(te))
	}
}

func TestRollingErrors(t *testing.T) {
	if _, err := NewRollingForecastCV(0, 1, 0); err == nil {
		t.Error("expected error for h=0")
	}
	if _, err := NewRollingForecastCV(1, 0, 0); err == nil {
		t.Error("expected error for step=0")
	}
	cv, _ := NewRollingForecastCV(5, 1, 100)
	if _, err := cv.Split(50); err == nil {
		t.Error("expected error: initial+h > n")
	}
}
