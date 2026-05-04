package arima

import (
	"math/rand/v2"
	"testing"

	"github.com/ajroetker/go-highway/hwy/contrib/vec"
)

// Bench harness measuring whether SIMD via go-highway beats the scalar
// loop at realistic ARIMA slice sizes. Each candidate is run at three
// lengths (144 = AirPassengers, 500 = medium, 1000 = long).
//
// Decision rule (per audit): only convert call sites where SIMD beats
// scalar by >20% on the smallest realistic n. Anything else stays
// scalar — the call overhead, branch mispredict, and code-path
// duplication isn't worth a sub-10% improvement.

func makeF64(n int, seed uint64) []float64 {
	rng := rand.New(rand.NewPCG(seed, seed^0x9E3779B97F4A7C15))
	out := make([]float64, n)
	for i := range out {
		out[i] = rng.NormFloat64()
	}
	return out
}

// Sum reduction — show up in residual SSE, log-likelihood, sigma².
func benchSumScalar(b *testing.B, n int) {
	x := makeF64(n, 1)
	b.ResetTimer()
	var s float64
	for i := 0; i < b.N; i++ {
		s = 0
		for _, v := range x {
			s += v
		}
	}
	_ = s
}

func benchSumSIMD(b *testing.B, n int) {
	x := makeF64(n, 1)
	b.ResetTimer()
	var s float64
	for i := 0; i < b.N; i++ {
		s = vec.Sum(x)
	}
	_ = s
}

func BenchmarkSum_Scalar_n144(b *testing.B)  { benchSumScalar(b, 144) }
func BenchmarkSum_SIMD_n144(b *testing.B)    { benchSumSIMD(b, 144) }
func BenchmarkSum_Scalar_n500(b *testing.B)  { benchSumScalar(b, 500) }
func BenchmarkSum_SIMD_n500(b *testing.B)    { benchSumSIMD(b, 500) }
func BenchmarkSum_Scalar_n1000(b *testing.B) { benchSumScalar(b, 1000) }
func BenchmarkSum_SIMD_n1000(b *testing.B)   { benchSumSIMD(b, 1000) }

// Dot product — shows up in OLS residual computation, exog adjustments.
func benchDotScalar(b *testing.B, n int) {
	x := makeF64(n, 1)
	y := makeF64(n, 2)
	b.ResetTimer()
	var s float64
	for i := 0; i < b.N; i++ {
		s = 0
		for j, v := range x {
			s += v * y[j]
		}
	}
	_ = s
}

func benchDotSIMD(b *testing.B, n int) {
	x := makeF64(n, 1)
	y := makeF64(n, 2)
	b.ResetTimer()
	var s float64
	for i := 0; i < b.N; i++ {
		s = vec.Dot(x, y)
	}
	_ = s
}

func BenchmarkDot_Scalar_n144(b *testing.B)  { benchDotScalar(b, 144) }
func BenchmarkDot_SIMD_n144(b *testing.B)    { benchDotSIMD(b, 144) }
func BenchmarkDot_Scalar_n500(b *testing.B)  { benchDotScalar(b, 500) }
func BenchmarkDot_SIMD_n500(b *testing.B)    { benchDotSIMD(b, 500) }
func BenchmarkDot_Scalar_n1000(b *testing.B) { benchDotScalar(b, 1000) }
func BenchmarkDot_SIMD_n1000(b *testing.B)   { benchDotSIMD(b, 1000) }

// AXPY (MulConstAddTo) — the residOf inner loop after column-transpose:
//   for j: dst[i] -= beta[j] * wXT[j][i]   →   vec.MulConstAddTo(dst, -beta[j], wXT[j])
func benchAxpyScalar(b *testing.B, n int) {
	x := makeF64(n, 1)
	dst := makeF64(n, 2)
	scratch := append([]float64{}, dst...)
	const a = 0.7
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(dst, scratch)
		for j, v := range x {
			dst[j] += a * v
		}
	}
}

func benchAxpySIMD(b *testing.B, n int) {
	x := makeF64(n, 1)
	dst := makeF64(n, 2)
	scratch := append([]float64{}, dst...)
	const a = 0.7
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(dst, scratch)
		vec.MulConstAddTo(dst, a, x)
	}
}

func BenchmarkAxpy_Scalar_n144(b *testing.B)  { benchAxpyScalar(b, 144) }
func BenchmarkAxpy_SIMD_n144(b *testing.B)    { benchAxpySIMD(b, 144) }
func BenchmarkAxpy_Scalar_n500(b *testing.B)  { benchAxpyScalar(b, 500) }
func BenchmarkAxpy_SIMD_n500(b *testing.B)    { benchAxpySIMD(b, 500) }
func BenchmarkAxpy_Scalar_n1000(b *testing.B) { benchAxpyScalar(b, 1000) }
func BenchmarkAxpy_SIMD_n1000(b *testing.B)   { benchAxpySIMD(b, 1000) }

// residOf shape: dst[i] = ws[i] - c - sum_j(beta[j] * wXT[j][i])
// This is the most realistic ARIMA hot loop. We compare the existing
// row-major scalar implementation vs a column-transposed SIMD version.
func benchResidOfScalar(b *testing.B, n, k int) {
	ws := makeF64(n, 1)
	beta := makeF64(k, 2)
	wX := make([][]float64, n)
	for i := range wX {
		wX[i] = makeF64(k, uint64(i)+10)
	}
	const c = 0.123
	dst := make([]float64, n)
	b.ResetTimer()
	for it := 0; it < b.N; it++ {
		for i, v := range ws {
			r := v - c
			for j := 0; j < k; j++ {
				r -= beta[j] * wX[i][j]
			}
			dst[i] = r
		}
	}
}

func benchResidOfSIMD(b *testing.B, n, k int) {
	ws := makeF64(n, 1)
	beta := makeF64(k, 2)
	// Pre-transpose wX to column-major (this happens once outside the
	// residOf hot path in the real implementation).
	wXT := make([][]float64, k)
	for j := range wXT {
		wXT[j] = makeF64(n, uint64(j)+10)
	}
	const c = 0.123
	dst := make([]float64, n)
	b.ResetTimer()
	for it := 0; it < b.N; it++ {
		// dst[i] = ws[i] - c
		copy(dst, ws)
		// dst[i] += -c (broadcast)
		// AddConst with -c
		// (or just inline the c into the loop start; vec.AddConst is fine)
		// Using a loop here for the -c since it's a single pass.
		for i := range dst {
			dst[i] -= c
		}
		// dst[i] -= beta[j] * wXT[j][i]
		for j := 0; j < k; j++ {
			vec.MulConstAddTo(dst, -beta[j], wXT[j])
		}
	}
}

func BenchmarkResidOf_Scalar_n144_k0(b *testing.B) { benchResidOfScalar(b, 144, 0) }
func BenchmarkResidOf_SIMD_n144_k0(b *testing.B)   { benchResidOfSIMD(b, 144, 0) }
func BenchmarkResidOf_Scalar_n144_k2(b *testing.B) { benchResidOfScalar(b, 144, 2) }
func BenchmarkResidOf_SIMD_n144_k2(b *testing.B)   { benchResidOfSIMD(b, 144, 2) }
func BenchmarkResidOf_Scalar_n144_k5(b *testing.B) { benchResidOfScalar(b, 144, 5) }
func BenchmarkResidOf_SIMD_n144_k5(b *testing.B)   { benchResidOfSIMD(b, 144, 5) }
func BenchmarkResidOf_Scalar_n500_k5(b *testing.B) { benchResidOfScalar(b, 500, 5) }
func BenchmarkResidOf_SIMD_n500_k5(b *testing.B)   { benchResidOfSIMD(b, 500, 5) }
func BenchmarkResidOf_Scalar_n1000_k5(b *testing.B) { benchResidOfScalar(b, 1000, 5) }
func BenchmarkResidOf_SIMD_n1000_k5(b *testing.B)   { benchResidOfSIMD(b, 1000, 5) }

// Kalman inner-loop is r×r where r ≤ p+q*M for SARIMA. For (0,1,1)(0,1,1)[12]
// r=14. Confirm whether SIMD even pays off at this size.
func benchSmallAxpy(b *testing.B, n int) {
	x := makeF64(n, 1)
	dst := makeF64(n, 2)
	scratch := append([]float64{}, dst...)
	const a = 0.7
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(dst, scratch)
		for j, v := range x {
			dst[j] += a * v
		}
	}
}

func benchSmallAxpySIMD(b *testing.B, n int) {
	x := makeF64(n, 1)
	dst := makeF64(n, 2)
	scratch := append([]float64{}, dst...)
	const a = 0.7
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(dst, scratch)
		vec.MulConstAddTo(dst, a, x)
	}
}

func BenchmarkAxpy_Scalar_n14(b *testing.B) { benchSmallAxpy(b, 14) }
func BenchmarkAxpy_SIMD_n14(b *testing.B)   { benchSmallAxpySIMD(b, 14) }
func BenchmarkAxpy_Scalar_n28(b *testing.B) { benchSmallAxpy(b, 28) }
func BenchmarkAxpy_SIMD_n28(b *testing.B)   { benchSmallAxpySIMD(b, 28) }

// Kalman inner-loop shapes from real ARIMA fits — verify SIMD
// break-even at r=14 (typical monthly SARIMA simple-diff) and rd=27
// (NonSimpleDifferencing for the same model).
//
// Two-vec dot pattern from kalman_full.go:178-189 (Pinf and Pstar
// matvecs share the same z multiplier per step).
func benchTwoDot(b *testing.B, n int, simd bool) {
	x := makeF64(n, 1)
	y1 := makeF64(n, 2)
	y2 := makeF64(n, 3)
	b.ResetTimer()
	var s1, s2 float64
	for i := 0; i < b.N; i++ {
		if simd {
			s1 = vec.Dot(x, y1)
			s2 = vec.Dot(x, y2)
		} else {
			s1 = 0
			s2 = 0
			for j, v := range x {
				s1 += v * y1[j]
				s2 += v * y2[j]
			}
		}
	}
	_, _ = s1, s2
}

func BenchmarkTwoDot_Scalar_n14(b *testing.B) { benchTwoDot(b, 14, false) }
func BenchmarkTwoDot_SIMD_n14(b *testing.B)   { benchTwoDot(b, 14, true) }
func BenchmarkTwoDot_Scalar_n27(b *testing.B) { benchTwoDot(b, 27, false) }
func BenchmarkTwoDot_SIMD_n27(b *testing.B)   { benchTwoDot(b, 27, true) }
func BenchmarkTwoDot_Scalar_n50(b *testing.B) { benchTwoDot(b, 50, false) }
func BenchmarkTwoDot_SIMD_n50(b *testing.B)   { benchTwoDot(b, 50, true) }
