package arima

import (
	"math"
	"sync"
	"testing"
)

// Linear interpolation on a simple monotonic grid.
func TestApprox_Linear(t *testing.T) {
	xRef := []float64{0, 1, 2, 3, 4}
	yRef := []float64{0, 10, 20, 30, 40}
	xOut := []float64{0.5, 1.5, 2.5, 3.5}
	got, err := Approx(xRef, yRef, xOut, Linear, RuleNaN)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{5, 15, 25, 35}
	for i := range got {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Errorf("got[%d]=%g want %g", i, got[i], want[i])
		}
	}
}

// RuleNaN: out-of-range queries → NaN.
func TestApprox_RuleNaN(t *testing.T) {
	got, err := Approx([]float64{0, 1}, []float64{0, 10},
		[]float64{-1, 0.5, 2}, Linear, RuleNaN)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(got[0]) {
		t.Errorf("got[0]=%g, want NaN", got[0])
	}
	if got[1] != 5 {
		t.Errorf("got[1]=%g, want 5", got[1])
	}
	if !math.IsNaN(got[2]) {
		t.Errorf("got[2]=%g, want NaN", got[2])
	}
}

// RuleClip: out-of-range queries → endpoint value.
func TestApprox_RuleClip(t *testing.T) {
	got, err := Approx([]float64{0, 1}, []float64{2, 8},
		[]float64{-1, 0.5, 2}, Linear, RuleClip)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != 2 {
		t.Errorf("got[0]=%g, want 2 (left clip)", got[0])
	}
	if got[1] != 5 {
		t.Errorf("got[1]=%g, want 5", got[1])
	}
	if got[2] != 8 {
		t.Errorf("got[2]=%g, want 8 (right clip)", got[2])
	}
}

// Duplicate-x handling: ties collapsed by mean (matches R's approx with ties='mean').
func TestApprox_DuplicateXTiesByMean(t *testing.T) {
	xRef := []float64{1, 1, 2}
	yRef := []float64{10, 20, 30} // mean of duplicates at x=1 → 15
	got, err := Approx(xRef, yRef, []float64{1, 1.5, 2}, Linear, RuleNaN)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != 15 {
		t.Errorf("got[0]=%g, want 15", got[0])
	}
	if got[1] != 22.5 {
		t.Errorf("got[1]=%g, want 22.5", got[1])
	}
	if got[2] != 30 {
		t.Errorf("got[2]=%g, want 30", got[2])
	}
}

// L-4: NewApproxTable + reused Approx must produce the same result as the
// one-shot Approx wrapper.
func TestApproxTable_MatchesOneShot(t *testing.T) {
	xRef := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	yRef := []float64{0, 1, 4, 9, 16, 25, 36, 49, 64, 81}
	xOut := []float64{0.3, 1.5, 2.7, 3.9, 5.1, 6.3, 7.5, 8.7}

	want, err := Approx(xRef, yRef, xOut, Linear, RuleNaN)
	if err != nil {
		t.Fatal(err)
	}
	tab, err := NewApproxTable(xRef, yRef)
	if err != nil {
		t.Fatal(err)
	}
	got := tab.Approx(xOut, Linear, RuleNaN)
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("got[%d]=%g, want %g", i, got[i], want[i])
		}
	}
	// Reuse — second call should produce identical results without re-sort.
	got2 := tab.Approx(xOut, Linear, RuleNaN)
	for i := range got2 {
		if got2[i] != want[i] {
			t.Errorf("reuse got[%d]=%g, want %g", i, got2[i], want[i])
		}
	}
}

// L-4: Len reports the dedupe'd grid size.
func TestApproxTable_LenAfterDedup(t *testing.T) {
	xRef := []float64{1, 1, 2, 2, 3} // 3 unique x's after collapse
	yRef := []float64{10, 20, 30, 40, 50}
	tab, err := NewApproxTable(xRef, yRef)
	if err != nil {
		t.Fatal(err)
	}
	if tab.Len() != 3 {
		t.Errorf("Len()=%d, want 3", tab.Len())
	}
}

// L-4: ApproxTable is immutable + concurrency-safe (audit's promise).
// Hammer it with many goroutines doing concurrent Approx calls under -race.
func TestApproxTable_ConcurrentSafe(t *testing.T) {
	xRef := make([]float64, 100)
	yRef := make([]float64, 100)
	for i := range xRef {
		xRef[i] = float64(i)
		yRef[i] = math.Sin(float64(i) * 0.1)
	}
	tab, err := NewApproxTable(xRef, yRef)
	if err != nil {
		t.Fatal(err)
	}
	xOut := []float64{0.5, 10.5, 50.5, 90.5}
	want := tab.Approx(xOut, Linear, RuleNaN)

	const goroutines = 32
	const callsPerGoroutine = 100
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for c := 0; c < callsPerGoroutine; c++ {
				got := tab.Approx(xOut, Linear, RuleNaN)
				for i := range got {
					if got[i] != want[i] {
						t.Errorf("concurrent mismatch [%d]: %g vs %g", i, got[i], want[i])
						return
					}
				}
			}
		}()
	}
	wg.Wait()
}

// L-4 perf: ApproxTable amortises sort+dedupe across many calls. Bench
// the same reference grid through 100 interpolation calls — old API
// re-sorts each time, ApproxTable does it once.
func BenchmarkApprox_OneShot(b *testing.B) {
	const n = 1000
	xRef := make([]float64, n)
	yRef := make([]float64, n)
	for i := range xRef {
		xRef[i] = float64(i)
		yRef[i] = math.Sin(float64(i) * 0.05)
	}
	xOut := []float64{1.5, 25.5, 100.5, 500.5, 999.5}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 100 calls per iteration, mimicking a workload that interpolates
		// the same grid against many query batches.
		for c := 0; c < 100; c++ {
			_, _ = Approx(xRef, yRef, xOut, Linear, RuleNaN)
		}
	}
}

func BenchmarkApprox_TableReused(b *testing.B) {
	const n = 1000
	xRef := make([]float64, n)
	yRef := make([]float64, n)
	for i := range xRef {
		xRef[i] = float64(i)
		yRef[i] = math.Sin(float64(i) * 0.05)
	}
	xOut := []float64{1.5, 25.5, 100.5, 500.5, 999.5}
	tab, _ := NewApproxTable(xRef, yRef)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for c := 0; c < 100; c++ {
			_ = tab.Approx(xOut, Linear, RuleNaN)
		}
	}
}

// L-4: validation errors flow through the new constructor.
func TestNewApproxTable_Validation(t *testing.T) {
	if _, err := NewApproxTable(nil, nil); err == nil {
		t.Error("expected error for empty xRef")
	}
	if _, err := NewApproxTable([]float64{1, 2}, []float64{1}); err == nil {
		t.Error("expected error for length mismatch")
	}
}
