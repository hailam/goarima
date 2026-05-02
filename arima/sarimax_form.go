package arima

// PublicSarimaxStateSpace exposes sarimaxStateSpace for parity diagnostics.
func PublicSarimaxStateSpace(phi, theta, sPhi, sTheta []float64, d, D, m int) (T [][]float64, Z []float64, R []float64) {
	tt, z, r, _ := sarimaxStateSpace(phi, theta, sPhi, sTheta, d, D, m)
	return tt, z, r
}

// sarimaxStateSpace builds the SARIMA state-space matrices in the form used by
// statsmodels.tsa.statespace.sarimax.SARIMAX (default, non-Hamilton, non-simple
// differencing). Mirrors `_initialize_state_space` plus `initial_design`,
// `initial_transition`, `initial_selection`.
//
// State layout for SARIMA(p,d,q)(P,D,Q,m):
//
//	[0, k_diff)                   non-seasonal-diff accumulators (= lag-1 of y)
//	[k_diff, k_states_diff)       seasonal-diff register (D blocks of m)
//	[k_states_diff, k_states)     ARMA companion (size k_order)
//
// where k_states_diff = d + D*m and k_order = max(p+P*m, q+Q*m+1).
//
// Returns T (k_states × k_states, row-major), Z (length k_states),
// R (length k_states; selection vector with MA params embedded).
func sarimaxStateSpace(phi, theta, sPhi, sTheta []float64, d, D, m int) (T [][]float64, Z []float64, R []float64, kStatesDiff int) {
	p := len(phi)
	q := len(theta)
	P := len(sPhi)
	Q := len(sTheta)

	if D > 0 && m <= 1 {
		D = 0 // seasonal differencing is a no-op when m <= 1
	}

	pFull := p + P*m
	qFull := q + Q*m

	kAR := pFull
	kMA := qFull
	kOrder := kAR
	if kMA+1 > kOrder {
		kOrder = kMA + 1
	}

	kDiff := d
	kSeasDiff := D
	kStatesDiff = kDiff + m*kSeasDiff
	kStates := kStatesDiff + kOrder

	// Z (design)
	Z = make([]float64, kStates)
	for i := 0; i < kDiff; i++ {
		Z[i] = 1
	}
	for ds := 0; ds < kSeasDiff; ds++ {
		Z[kDiff+ds*m+m-1] = 1
	}
	if kOrder > 0 {
		Z[kStatesDiff] = 1
	}

	// T (transition)
	T = make([][]float64, kStates)
	for i := range T {
		T[i] = make([]float64, kStates)
	}

	// ARMA block (bottom-right): companion_matrix(kOrder) shifted with phi_full.
	if kOrder > 0 {
		// companion: T[kStatesDiff+i][kStatesDiff+i+1] = 1 for i = 0..kOrder-2
		for i := 0; i < kOrder-1; i++ {
			T[kStatesDiff+i][kStatesDiff+i+1] = 1
		}
		// AR params (phi_full from polynomial expansion) fill column kStatesDiff
		phiFull := expandSARMA(phi, sPhi, m)
		for i := 0; i < kAR && i < kOrder; i++ {
			T[kStatesDiff+i][kStatesDiff] = phiFull[i]
		}
	}

	// Seasonal-diff blocks
	for ds := 0; ds < kSeasDiff; ds++ {
		start := kDiff + ds*m
		// seasonal_companion = companion_matrix(m).T  with [0, m-1] = 1
		T[start][start+m-1] = 1
		for i := 0; i < m-1; i++ {
			T[start+i+1][start+i] = 1
		}
		// \iota: couple to ARMA[0]
		T[start][kStatesDiff] = 1
		// link to next seasonal block (if any)
		if ds < kSeasDiff-1 {
			T[start][start+m+m-1] = 1
		}
	}

	// Non-seasonal differencing block (top-left)
	if kDiff > 0 {
		// upper-triangular ones
		for i := 0; i < kDiff; i++ {
			for j := i; j < kDiff; j++ {
				T[i][j] = 1
			}
		}
		// couple diff -> seasonal-diff register tails
		if kSeasDiff > 0 && m > 0 {
			for i := 0; i < kDiff; i++ {
				for ds := 0; ds < kSeasDiff; ds++ {
					T[i][kDiff+ds*m+m-1] = 1
				}
			}
		}
		// couple diff -> ARMA[0]
		for i := 0; i < kDiff; i++ {
			T[i][kStatesDiff] = 1
		}
	}

	// R (selection): impulse to ARMA[0] plus MA params at the next slots.
	R = make([]float64, kStates)
	if kOrder > 0 {
		R[kStatesDiff] = 1
		thetaFull := expandSMA(theta, sTheta, m)
		for i := 0; i < kMA && kStatesDiff+1+i < kStates; i++ {
			R[kStatesDiff+1+i] = thetaFull[i]
		}
	}
	return T, Z, R, kStatesDiff
}
