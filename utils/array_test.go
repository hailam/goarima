package utils

import (
	"math"
	"reflect"
	"testing"
)

func sliceClose(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
}

func sliceEq(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] && !(math.IsNaN(a[i]) && math.IsNaN(b[i])) {
			return false
		}
	}
	return true
}

func matEq(a, b [][]float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !sliceEq(a[i], b[i]) {
			return false
		}
	}
	return true
}

func mustDiff(t *testing.T, x []float64, lag, d int) []float64 {
	t.Helper()
	out, err := Diff(x, lag, d)
	if err != nil {
		t.Fatalf("Diff: %v", err)
	}
	return out
}

func mustDiffMat(t *testing.T, x [][]float64, lag, d int) [][]float64 {
	t.Helper()
	out, err := DiffMatrix(x, lag, d)
	if err != nil {
		t.Fatalf("DiffMatrix: %v", err)
	}
	return out
}

// Mirrors test_diff() in test_array.py.
func TestDiff(t *testing.T) {
	x := []float64{0, 1, 2, 3, 4}

	// vector cases
	if got := mustDiff(t, x, 1, 1); !sliceEq(got, []float64{1, 1, 1, 1}) {
		t.Errorf("diff(x,1,1) = %v", got)
	}
	if got := mustDiff(t, x, 1, 2); !sliceEq(got, []float64{0, 0, 0}) {
		t.Errorf("diff(x,1,2) = %v", got)
	}
	if got := mustDiff(t, x, 2, 1); !sliceEq(got, []float64{2, 2, 2}) {
		t.Errorf("diff(x,2,1) = %v", got)
	}
	if got := mustDiff(t, x, 2, 2); !sliceEq(got, []float64{0}) {
		t.Errorf("diff(x,2,2) = %v", got)
	}

	// matrix m built like Python: np.array([10,5,12,23,18,3,2,0,12]).reshape(3,3).T
	// reshape(3,3) row-major: [[10,5,12],[23,18,3],[2,0,12]]
	// .T transposes -> [[10,23,2],[5,18,0],[12,3,12]]
	m := [][]float64{
		{10, 23, 2},
		{5, 18, 0},
		{12, 3, 12},
	}
	if got := mustDiffMat(t, m, 1, 1); !matEq(got, [][]float64{{-5, -5, -2}, {7, -15, 12}}) {
		t.Errorf("diff(m,1,1) = %v", got)
	}
	if got := mustDiffMat(t, m, 1, 2); !matEq(got, [][]float64{{12, -10, 14}}) {
		t.Errorf("diff(m,1,2) = %v", got)
	}
	if got := mustDiffMat(t, m, 2, 1); !matEq(got, [][]float64{{2, -20, 10}}) {
		t.Errorf("diff(m,2,1) = %v", got)
	}
	if got := mustDiffMat(t, m, 2, 2); len(got) != 0 {
		t.Errorf("diff(m,2,2) should be empty, got %v", got)
	}
}

// Mirrors test_diff_inv parametrized cases.
func TestDiffInv(t *testing.T) {
	x := []float64{0, 1, 2, 3, 4}

	cases := []struct {
		name string
		arr  []float64
		lag  int
		diff int
		xi   []float64
		want []float64
	}{
		{"x lag1 d1 nil", x, 1, 1, nil, []float64{0, 0, 1, 3, 6, 10}},
		{"x lag1 d2 nil", x, 1, 2, nil, []float64{0, 0, 0, 1, 4, 10, 20}},
		{"x lag2 d1 nil", x, 2, 1, nil, []float64{0, 0, 0, 1, 2, 4, 6}},
		{"x lag2 d2 nil", x, 2, 2, nil, []float64{0, 0, 0, 0, 0, 1, 2, 5, 8}},
		{"intermediate", []float64{1, 0, 3, 2}, 1, 1, []float64{0}, []float64{0, 1, 1, 4, 6}},
		{"x lag1 d1 xi[0]", x, 1, 1, []float64{0}, []float64{0, 0, 1, 3, 6, 10}},
	}
	for _, c := range cases {
		got, err := DiffInv(c.arr, c.lag, c.diff, c.xi)
		if err != nil {
			t.Errorf("%s: err %v", c.name, err)
			continue
		}
		if !sliceEq(got, c.want) {
			t.Errorf("%s: got %v want %v", c.name, got, c.want)
		}
	}
}

func TestDiffInvMatrix(t *testing.T) {
	// xMat is np.arange(1,10).reshape(3,3).T -> [[1,4,7],[2,5,8],[3,6,9]]
	xMat := [][]float64{
		{1, 4, 7},
		{2, 5, 8},
		{3, 6, 9},
	}
	cases := []struct {
		name string
		lag  int
		diff int
		want [][]float64
	}{
		{"lag1 d1", 1, 1, [][]float64{
			{0, 0, 0}, {1, 4, 7}, {3, 9, 15}, {6, 15, 24},
		}},
		{"lag1 d2", 1, 2, [][]float64{
			{0, 0, 0}, {0, 0, 0}, {1, 4, 7}, {4, 13, 22}, {10, 28, 46},
		}},
		{"lag2 d1", 2, 1, [][]float64{
			{0, 0, 0}, {0, 0, 0}, {1, 4, 7}, {2, 5, 8}, {4, 10, 16},
		}},
		{"lag2 d2", 2, 2, [][]float64{
			{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
			{1, 4, 7}, {2, 5, 8}, {5, 14, 23},
		}},
	}
	for _, c := range cases {
		got, err := DiffInvMatrix(xMat, c.lag, c.diff, nil)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if !matEq(got, c.want) {
			t.Errorf("%s: got %v want %v", c.name, got, c.want)
		}
	}
}

// Mirrors test_corner.
func TestDiffCornerErrors(t *testing.T) {
	x := []float64{0, 1, 2, 3, 4}
	if _, err := Diff(x, 0, 1); err == nil {
		t.Error("expected error for lag=0")
	}
	if _, err := DiffInv(x, 0, 1, nil); err == nil {
		t.Error("expected error for lag=0 (inv)")
	}
	if _, err := Diff(x, 1, 0); err == nil {
		t.Error("expected error for differences=0")
	}
	if _, err := DiffInv(x, 1, 0, nil); err == nil {
		t.Error("expected error for differences=0 (inv)")
	}
	// wrong xi shape on 2D
	bad := [][]float64{{1, 1}, {1, 1}}
	if _, err := DiffInvMatrix(bad, 1, 1, [][]float64{{1}}); err == nil {
		t.Error("expected error on bad xi shape")
	}
}

func TestConcat(t *testing.T) {
	got := Concat(V(1), VS([]float64{0, 0, 0}))
	if !sliceEq(got, []float64{1, 0, 0, 0}) {
		t.Errorf("Concat scalar+slice: %v", got)
	}
	got = Concat(VS([]float64{1}), VS([]float64{0, 0, 0}))
	if !sliceEq(got, []float64{1, 0, 0, 0}) {
		t.Errorf("Concat slice+slice: %v", got)
	}
	got = Concat(V(1))
	if !sliceEq(got, []float64{1}) {
		t.Errorf("Concat single scalar: %v", got)
	}
	if Concat() != nil {
		t.Error("Concat() should be nil")
	}
}

func TestCheckEndog(t *testing.T) {
	x := []float64{0, 1, 2, 3, 4}
	out, err := CheckEndog(x, false, true)
	if err != nil {
		t.Fatal(err)
	}
	if !sliceEq(out, x) {
		t.Errorf("CheckEndog: got %v want %v", out, x)
	}
	// finite-required failure
	bad := []float64{1, math.NaN(), 3}
	if _, err := CheckEndog(bad, true, true); err == nil {
		t.Error("expected NaN to fail with forceAllFinite")
	}
}

func TestCheckExog(t *testing.T) {
	x := [][]float64{{1, 2}, {3, 4}}
	out, err := CheckExog(x, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if !matEq(out, x) {
		t.Errorf("CheckExog passthrough failed")
	}
	// non-finite
	bad := [][]float64{{1, math.Inf(1)}, {3, 4}}
	if _, err := CheckExog(bad, true, false); err == nil {
		t.Error("expected error on inf with finite required")
	}
	// uneven rows
	uneven := [][]float64{{1, 2}, {3}}
	if _, err := CheckExog(uneven, false, false); err == nil {
		t.Error("expected error on uneven rows")
	}
}

func TestIsConstant(t *testing.T) {
	if !IsConstant([]float64{5, 5, 5}) {
		t.Error("constant slice should be constant")
	}
	if IsConstant([]float64{1, 2, 3}) {
		t.Error("non-constant slice should not be constant")
	}
	if !IsConstant([]float64{}) {
		t.Error("empty should be constant")
	}
	if !IsConstant([]float64{42}) {
		t.Error("singleton should be constant")
	}
}

// Sanity: integrateVec invariant — diff(integrate(x)) == x[lag:]?
// Actually diff_inv inverts diff; verify Diff(DiffInv(x)) == x.
func TestDiffInverts(t *testing.T) {
	x := []float64{2.5, -1.0, 3.3, 7.7, 0.5}
	for _, lag := range []int{1, 2, 3} {
		for _, d := range []int{1, 2} {
			inv, err := DiffInv(x, lag, d, nil)
			if err != nil {
				t.Fatalf("DiffInv: %v", err)
			}
			back, err := Diff(inv, lag, d)
			if err != nil {
				t.Fatalf("Diff: %v", err)
			}
			if !sliceClose(back, x, 1e-9) {
				t.Errorf("lag=%d d=%d: round trip got %v want %v", lag, d, back, x)
			}
		}
	}
	_ = reflect.DeepEqual
}
