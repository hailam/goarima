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

// integrateBackTail inverses applyDiff and returns ONLY the tail —
// the integrated forecast values, omitting the prepended head. Saves
// the alloc + copy that the full `integrateBack` does for the head
// region, which immediately gets discarded by every PredictBoot caller.
//
// `dst` is the destination buffer (length len(diffed)); resized via
// ensureLen if too small. `scratch` is a small carry buffer of length
// `lag*times` reused across iterations to avoid per-call allocations.
//
// CDX-2: addresses `integrateBack` being the largest alloc_objects
// site in profile (23.46% of total per codex audit).
func integrateBackTail(dst, diffed, head []float64, lag, times int) []float64 {
	dst = ensureLen(dst, len(diffed))
	if times == 0 {
		copy(dst, diffed)
		return dst
	}
	if times == 1 {
		// Single-pass: out[i] = diffed[i] + carry[i]
		// where carry[0..lag) = head[0..lag), then carry[i] = out[i-lag].
		// We can compute this in-place using `dst` as the carry tail:
		// for i < lag: dst[i] = diffed[i] + head[i]
		// for i >= lag: dst[i] = diffed[i] + dst[i-lag]
		for i := 0; i < lag && i < len(diffed); i++ {
			dst[i] = diffed[i] + head[i]
		}
		for i := lag; i < len(diffed); i++ {
			dst[i] = diffed[i] + dst[i-lag]
		}
		return dst
	}
	// times > 1: fall back to recursive integrateBack and slice off the head.
	// This path is rare (only multi-fold differencing on the same axis,
	// e.g. d=2 or D=2). The single-step path handles the common cases.
	full := integrateBack(diffed, head, lag, times)
	copy(dst, full[lag*times:])
	return dst
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
