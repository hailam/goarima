package arima

// applyDiff differences x by lag, repeated `times` times.
// Returns a fresh slice. If times*lag > len(x), returns empty slice.
func applyDiff(x []float64, lag, times int) []float64 {
	res := make([]float64, len(x))
	copy(res, x)
	for i := 0; i < times; i++ {
		if len(res) <= lag {
			return []float64{}
		}
		next := make([]float64, len(res)-lag)
		for j := 0; j < len(next); j++ {
			next[j] = res[j+lag] - res[j]
		}
		res = next
	}
	return res
}

// integrateBack inverses applyDiff given the original head values.
// `head` are the consumed-by-differencing values (length lag*times).
// Returns y of length len(diffed) + lag*times, with head prepended.
func integrateBack(diffed []float64, head []float64, lag, times int) []float64 {
	if times == 0 {
		out := make([]float64, len(diffed))
		copy(out, diffed)
		return out
	}
	// Recursive: head has lag*times entries.
	// Apply once: take head[0:lag] as initial values, undifference once, then recurse with lag*(times-1) head.
	// We follow R's diffinv for recursive case:
	//   diffinv(x, lag, d) = diffinv(diffinv(x, lag, d-1, diff(head, lag, 1)), lag, 1, head[0:lag])
	if times == 1 {
		out := make([]float64, lag+len(diffed))
		copy(out[:lag], head[:lag])
		for i := 0; i < len(diffed); i++ {
			out[i+lag] = diffed[i] + out[i]
		}
		return out
	}
	headDiff := applyDiff(head, lag, 1)
	inner := integrateBack(diffed, headDiff, lag, times-1)
	return integrateBack(inner, head[:lag], lag, 1)
}
