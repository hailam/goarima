// Package utils provides array utilities ported from pmdarima/utils/array.py.
package utils

import (
	"errors"
	"fmt"
	"math"
)

// ErrBadLagOrDifferences is returned when lag or differences < 1.
var ErrBadLagOrDifferences = errors.New("lag and differences must be positive (> 0) integers")

// ErrXiShape is returned when xi has the wrong shape for diff_inv.
var ErrXiShape = errors.New(`"xi" does not have the right shape`)

// Diff computes lag differences of a 1D vector, like R's diff() and pmdarima's diff().
//
// If x has length n with lag=1 differences=1, returns x[lag:n] - x[:n-lag].
// Recursively applies for differences > 1. Returns empty slice if lag > len(x).
func Diff(x []float64, lag, differences int) ([]float64, error) {
	if lag < 1 || differences < 1 {
		return nil, ErrBadLagOrDifferences
	}
	res := make([]float64, len(x))
	copy(res, x)
	for i := 0; i < differences; i++ {
		res = diffVec(res, lag)
		if len(res) == 0 {
			return res, nil
		}
	}
	return res, nil
}

func diffVec(x []float64, lag int) []float64 {
	n := len(x)
	if lag > n {
		lag = n
	}
	if n-lag <= 0 {
		return []float64{}
	}
	out := make([]float64, n-lag)
	for i := 0; i < n-lag; i++ {
		out[i] = x[i+lag] - x[i]
	}
	return out
}

// DiffMatrix computes lag differences row-wise on a matrix represented as
// row-major [][]float64 (rows of equal length).
func DiffMatrix(x [][]float64, lag, differences int) ([][]float64, error) {
	if lag < 1 || differences < 1 {
		return nil, ErrBadLagOrDifferences
	}
	res := cloneMatrix(x)
	for i := 0; i < differences; i++ {
		res = diffMat(res, lag)
		if len(res) == 0 {
			return res, nil
		}
	}
	return res, nil
}

func cloneMatrix(x [][]float64) [][]float64 {
	out := make([][]float64, len(x))
	for i, row := range x {
		r := make([]float64, len(row))
		copy(r, row)
		out[i] = r
	}
	return out
}

func diffMat(x [][]float64, lag int) [][]float64 {
	m := len(x)
	if lag > m {
		lag = m
	}
	if m-lag <= 0 {
		return [][]float64{}
	}
	cols := 0
	if m > 0 {
		cols = len(x[0])
	}
	out := make([][]float64, m-lag)
	for i := 0; i < m-lag; i++ {
		row := make([]float64, cols)
		for j := 0; j < cols; j++ {
			row[j] = x[i+lag][j] - x[i][j]
		}
		out[i] = row
	}
	return out
}

// DiffInv inverts a lag-differences operation on a 1D vector.
//
// xi is the initial values; if nil, zeros of length lag*differences are used.
func DiffInv(x []float64, lag, differences int, xi []float64) ([]float64, error) {
	if lag < 1 || differences < 1 {
		return nil, ErrBadLagOrDifferences
	}
	return diffInvVec(x, lag, differences, xi)
}

// diffInvVec implements R's diffinv.vector recursively.
func diffInvVec(x []float64, lag, differences int, xi []float64) ([]float64, error) {
	if xi == nil {
		xi = make([]float64, lag*differences)
	} else if len(xi) != lag*differences {
		return nil, fmt.Errorf(`"xi" does not have the right length`)
	}

	if differences == 1 {
		return integrateVec(x, xi, lag), nil
	}
	// recursive case (matches R's diffinv)
	innerXi, err := Diff(xi, lag, 1)
	if err != nil {
		return nil, err
	}
	inner, err := diffInvVec(x, lag, differences-1, innerXi)
	if err != nil {
		return nil, err
	}
	head := make([]float64, lag)
	copy(head, xi[:lag])
	return diffInvVec(inner, lag, 1, head)
}

// integrateVec replicates the C_intgrt_vec helper used by pmdarima.
// Output length = len(x) + lag.
// y[0:lag] = xi; y[i+lag] = x[i] + y[i] for i in [0, len(x)).
func integrateVec(x, xi []float64, lag int) []float64 {
	n := len(x)
	out := make([]float64, n+lag)
	copy(out[:lag], xi)
	for i := 0; i < n; i++ {
		out[i+lag] = x[i] + out[i]
	}
	return out
}

// DiffInvMatrix inverts diff column-wise on a row-major matrix.
func DiffInvMatrix(x [][]float64, lag, differences int, xi [][]float64) ([][]float64, error) {
	if lag < 1 || differences < 1 {
		return nil, ErrBadLagOrDifferences
	}
	n := len(x)
	if n == 0 {
		return [][]float64{}, nil
	}
	cols := len(x[0])
	rows := n + lag*differences
	if xi == nil {
		xi = make([][]float64, lag*differences)
		for i := range xi {
			xi[i] = make([]float64, cols)
		}
	} else if len(xi) != lag*differences || (len(xi) > 0 && len(xi[0]) != cols) {
		return nil, ErrXiShape
	}

	// transpose to columns, run diffInvVec per column, restitch
	colsData := make([][]float64, cols)
	for c := 0; c < cols; c++ {
		colVec := make([]float64, n)
		for i := 0; i < n; i++ {
			colVec[i] = x[i][c]
		}
		xiCol := make([]float64, len(xi))
		for i := range xi {
			xiCol[i] = xi[i][c]
		}
		out, err := diffInvVec(colVec, lag, differences, xiCol)
		if err != nil {
			return nil, err
		}
		colsData[c] = out
	}

	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		row := make([]float64, cols)
		for c := 0; c < cols; c++ {
			row[c] = colsData[c][i]
		}
		result[i] = row
	}
	return result, nil
}

// CheckEndog ensures y is a 1D float64 slice with finite values (if forceAllFinite).
// Returns a (possibly copied) []float64. Mirrors check_endog.
func CheckEndog(y []float64, forceAllFinite bool, copyArr bool) ([]float64, error) {
	if y == nil {
		return nil, errors.New("endog is nil")
	}
	if forceAllFinite {
		for i, v := range y {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, fmt.Errorf("non-finite value at index %d", i)
			}
		}
	}
	if copyArr {
		out := make([]float64, len(y))
		copy(out, y)
		return out, nil
	}
	return y, nil
}

// CheckExog ensures X is 2D, finite (when required), and returns a copy if asked.
func CheckExog(x [][]float64, forceAllFinite bool, copyArr bool) ([][]float64, error) {
	if x == nil {
		return nil, errors.New("exog is nil")
	}
	if len(x) > 0 {
		ncol := len(x[0])
		for i, row := range x {
			if len(row) != ncol {
				return nil, fmt.Errorf("row %d has %d cols; expected %d", i, len(row), ncol)
			}
			if forceAllFinite {
				for j, v := range row {
					if math.IsNaN(v) || math.IsInf(v, 0) {
						return nil, fmt.Errorf("non-finite at [%d,%d]", i, j)
					}
				}
			}
		}
	}
	if copyArr {
		return cloneMatrix(x), nil
	}
	return x, nil
}

// IsConstant returns true if every value of x equals the first.
func IsConstant(x []float64) bool {
	if len(x) <= 1 {
		return true
	}
	first := x[0]
	for _, v := range x[1:] {
		if v != first {
			return false
		}
	}
	return true
}

// Val carries either a scalar or a slice for use with Concat (R's c()).
// Pass any combination of float64 or []float64 via V or VS.
type Val struct {
	scalar float64
	slice  []float64
	isSlc  bool
}

// V wraps a scalar.
func V(x float64) Val { return Val{scalar: x} }

// VS wraps a slice.
func VS(x []float64) Val { return Val{slice: x, isSlc: true} }

// Concat concatenates Val arguments, mimicking R's c() for numeric inputs.
func Concat(args ...Val) []float64 {
	if len(args) == 0 {
		return nil
	}
	total := 0
	for _, a := range args {
		if a.isSlc {
			total += len(a.slice)
		} else {
			total++
		}
	}
	out := make([]float64, 0, total)
	for _, a := range args {
		if a.isSlc {
			out = append(out, a.slice...)
		} else {
			out = append(out, a.scalar)
		}
	}
	return out
}

// SeqInt returns [start, start+1, ..., stop-1] as a []float64.
func SeqInt(start, stop int) []float64 {
	if stop <= start {
		return []float64{}
	}
	out := make([]float64, stop-start)
	for i := range out {
		out[i] = float64(start + i)
	}
	return out
}
