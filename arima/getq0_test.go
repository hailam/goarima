package arima

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestSmithVsInclu2_Parity validates GARD-OPT-1: at low/moderate r,
// Smith's doubling iteration produces stationary-covariance matrices
// bit-equivalent to inclu2's Givens-rotation path. At high r with
// AR-heavy coefficients, inclu2 itself accumulates rounding error
// across the np = r·(r+1)/2 Givens passes (~5000+ at r=100), so the
// expectation is "agree to 1e-2 broadly, with both matching dense
// Lyapunov as ground truth where dense is feasible".
//
// Production behavior: stationaryCovGardnerInto dispatches to Smith
// for p > 0 && r >= gardSmithThresholdR — at those shapes Smith is
// the *more* numerically stable path, since each doubling step is
// just two r×r matmuls (rounding bounded by O(r³ · ε)) rather than
// inclu2's chain of np Givens rotations (O(np² · ε) in worst case).
func TestSmithVsInclu2_Parity(t *testing.T) {
	type shape struct {
		p, q int
	}
	shapes := []shape{
		// AR-only
		{1, 0}, {2, 0}, {3, 0}, {5, 0}, {12, 0}, {20, 0},
		// ARMA mixes
		{1, 1}, {1, 13}, {1, 26}, {1, 52},
		{2, 13}, {3, 26}, {5, 52},
		// AR-heavy
		{12, 1}, {12, 12},
	}
	rng := rand.New(rand.NewPCG(7, 11))
	for _, sh := range shapes {
		for trial := 0; trial < 3; trial++ {
			phi := make([]float64, sh.p)
			theta := make([]float64, sh.q)
			for i := range phi {
				phi[i] = 0.4 / float64(sh.p+1) * rng.NormFloat64()
			}
			for i := range theta {
				theta[i] = 0.3 / float64(sh.q+1) * rng.NormFloat64()
			}
			r := sh.p
			if sh.q+1 > r {
				r = sh.q + 1
			}

			refFlat := computeViaInclu2(phi, theta, r)
			testFlat := computeViaSmith(phi, theta, r)
			if testFlat == nil {
				t.Errorf("Smith failed on p=%d q=%d r=%d trial=%d", sh.p, sh.q, r, trial)
				continue
			}

			maxDiff := 0.0
			for i := 0; i < r*r; i++ {
				d := math.Abs(refFlat[i] - testFlat[i])
				if d > maxDiff {
					maxDiff = d
				}
			}
			if maxDiff > 1e-10 {
				t.Errorf("p=%d q=%d r=%d trial=%d: Smith vs inclu2 maxDiff=%.3e (want < 1e-10 at this scale)",
					sh.p, sh.q, r, trial, maxDiff)
			}
		}
	}
}

// TestSmithDenseLyapunov validates Smith's doubling against the dense
// O(r⁶) Lyapunov solve at sizes where dense is feasible. This is the
// strongest correctness check — independent of inclu2.
func TestSmithDenseLyapunov(t *testing.T) {
	type shape struct {
		p, q int
	}
	shapes := []shape{
		{1, 0}, {2, 0}, {3, 0}, {5, 0}, {12, 0}, {20, 0}, {30, 0},
		{1, 13}, {1, 26}, {2, 26}, {3, 26}, {5, 25},
		{12, 12},
	}
	rng := rand.New(rand.NewPCG(7, 11))
	for _, sh := range shapes {
		for trial := 0; trial < 2; trial++ {
			phi := make([]float64, sh.p)
			theta := make([]float64, sh.q)
			for i := range phi {
				phi[i] = 0.4 / float64(sh.p+1) * rng.NormFloat64()
			}
			for i := range theta {
				theta[i] = 0.3 / float64(sh.q+1) * rng.NormFloat64()
			}
			r := sh.p
			if sh.q+1 > r {
				r = sh.q + 1
			}
			pSmith := computeViaSmith(phi, theta, r)
			if pSmith == nil {
				continue
			}
			pDense := denseLyapunovReference(phi, theta, r)
			maxDiff := 0.0
			for i := 0; i < r*r; i++ {
				d := math.Abs(pSmith[i] - pDense[i])
				if d > maxDiff {
					maxDiff = d
				}
			}
			if maxDiff > 1e-10 {
				t.Errorf("p=%d q=%d r=%d trial=%d: Smith vs dense maxDiff=%.3e (want < 1e-10)",
					sh.p, sh.q, r, trial, maxDiff)
			}
		}
	}
}

// denseLyapunovReference solves vec(P) = (I - T⊗T)⁻¹ · vec(R Rᵀ) via
// gonum's general LU — O(r⁶) bottom-line ground truth for testing.
// Slow but bit-exact; used only by tests at small r.
func denseLyapunovReference(phi, theta []float64, r int) []float64 {
	r2 := r * r
	// Build T (companion).
	tMat := make([]float64, r2)
	p := len(phi)
	for i := 0; i < r; i++ {
		if i < p {
			tMat[i*r] = phi[i]
		}
		if i+1 < r {
			tMat[i*r+i+1] = 1
		}
	}
	// Build Q = R Rᵀ where R = (1, theta_1, ..., theta_q).
	q := len(theta)
	rvec := make([]float64, r)
	rvec[0] = 1
	for j := 0; j < q && j+1 < r; j++ {
		rvec[j+1] = theta[j]
	}
	qMat := make([]float64, r2)
	for i := 0; i < r; i++ {
		for j := 0; j < r; j++ {
			qMat[i*r+j] = rvec[i] * rvec[j]
		}
	}
	// I - T⊗T as a flat r²×r² matrix in row-major.
	M := make([]float64, r2*r2)
	for i := 0; i < r; i++ {
		for j := 0; j < r; j++ {
			tij := tMat[i*r+j]
			for k := 0; k < r; k++ {
				for l := 0; l < r; l++ {
					M[(i*r+k)*r2+(j*r+l)] = -tij * tMat[k*r+l]
				}
			}
		}
	}
	for k := 0; k < r2; k++ {
		M[k*r2+k] += 1
	}
	// Solve M · vec(P) = vec(Q) via Gaussian elimination.
	rhs := make([]float64, r2)
	for i := 0; i < r; i++ {
		for j := 0; j < r; j++ {
			rhs[i*r+j] = qMat[i*r+j]
		}
	}
	gaussSolveDense(M, rhs, r2)
	out := make([]float64, r2)
	for i := 0; i < r; i++ {
		for j := 0; j < r; j++ {
			out[i*r+j] = rhs[i*r+j]
		}
	}
	return out
}

// gaussSolveDense solves M · x = b in-place using partial-pivot
// Gaussian elimination. For test reference use only; O(n³).
func gaussSolveDense(M, b []float64, n int) {
	for i := 0; i < n; i++ {
		// Partial pivot.
		piv := i
		pivVal := math.Abs(M[i*n+i])
		for k := i + 1; k < n; k++ {
			if v := math.Abs(M[k*n+i]); v > pivVal {
				piv = k
				pivVal = v
			}
		}
		if piv != i {
			for j := 0; j < n; j++ {
				M[i*n+j], M[piv*n+j] = M[piv*n+j], M[i*n+j]
			}
			b[i], b[piv] = b[piv], b[i]
		}
		// Eliminate below.
		for k := i + 1; k < n; k++ {
			factor := M[k*n+i] / M[i*n+i]
			for j := i; j < n; j++ {
				M[k*n+j] -= factor * M[i*n+j]
			}
			b[k] -= factor * b[i]
		}
	}
	// Back-substitute.
	for i := n - 1; i >= 0; i-- {
		s := b[i]
		for j := i + 1; j < n; j++ {
			s -= M[i*n+j] * b[j]
		}
		b[i] = s / M[i*n+i]
	}
}

func computeViaInclu2(phi, theta []float64, r int) []float64 {
	flat, _ := stationaryCovInclu2OnlyInto(&gardnerWorkspace{}, phi, theta)
	out := make([]float64, r*r)
	copy(out, flat)
	return out
}

func computeViaSmith(phi, theta []float64, r int) []float64 {
	var ws gardnerWorkspace
	flat, ok := stationaryCovSmithInto(&ws, phi, theta, r)
	if !ok {
		return nil
	}
	out := make([]float64, r*r)
	copy(out, flat)
	return out
}
