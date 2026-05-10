package arima

import (
	"math"
	"sync"
)

// paramScratch is per-likelihood-eval reusable scratch space. The four
// hot allocator chains (arTransparams, maTransparams, expandSARMA,
// expandSMA) together account for >50% of allocations during a Fit;
// pooling their internal buffers eliminates that pressure.
//
// KAL-WORKSPACE extends this with `kalman`, the buffer set used by
// `kalmanARMALikelihoodInto` — 9 more allocations per likelihood call
// pooled, which dominates Kalman-path Fit allocations once the
// expand/transparams chain is pooled.
//
// Acquired via the package-level pool; each goroutine gets its own
// scratch in the parallelGradient case.
type paramScratch struct {
	// Transparams scratch + outputs (one pair per AR/MA family).
	transARPhi, transARPhiTmp       []float64
	transMATheta, transMAThetaTmp   []float64
	transARSPhi, transARSPhiTmp     []float64
	transMASTheta, transMASThetaTmp []float64
	beta                            []float64

	// expandSARMA scratch (a / b / c / out).
	expPhiA, expPhiB, expPhiC, expPhiOut []float64

	// expandSMA scratch.
	expThetaA, expThetaB, expThetaC, expThetaOut []float64

	// kalmanARMALikelihoodInto scratch — see KAL-WORKSPACE.
	kalman kalmanWorkspace
}

// kalmanWorkspace holds the reusable buffers that kalmanARMALikelihood
// would otherwise allocate per call. Sized lazily to fit the largest
// (n, r) seen in the current Fit; subsequent calls with smaller shapes
// reuse the leading prefix.
//
// PG-113: nzT and TP were dropped — the predict step now exploits T's
// companion structure directly (shift + phi-broadcast), eliminating the
// sparse-T iteration and the TP intermediate.
type kalmanWorkspace struct {
	Rvec []float64 // R selection vector, len r
	RRt  []float64 // RR' precomputed, len r*r (zeroed each call)
	a    []float64 // state mean, len r (zeroed each call)
	K    []float64 // Kalman gain, len r
	row0 []float64 // P[0,:] snapshot, len r
	newA []float64 // predicted state, len r
	newP []float64 // predicted covariance, len r*r
	// (innov omitted from the workspace — Fit's compute closure throws
	// it away; allocating it would be wasted. Internal callers that
	// need innovations use the legacy kalmanARMALikelihood entry point.)

	gardner gardnerWorkspace // pooled buffers for stationaryCovGardnerInto
}

// gardnerWorkspace holds the 7 reusable buffers that stationaryCovGardner
// would otherwise allocate per call. Pre-G-NEW-3 profile showed
// stationaryCovGardner as the largest single allocator in AutoArima
// (~36% of total allocs); pooling these eliminates that pressure.
type gardnerWorkspace struct {
	V      []float64 // length np = r·(r+1)/2 (zeroed each call)
	P      []float64 // length r*r (Gardner's output — caller aliases it)
	xnext  []float64 // length np
	xrow   []float64 // length np (zeroed each call by inclu2)
	rbar   []float64 // length nrbar (zeroed each call)
	thetab []float64 // length np (zeroed each call)
	Pbuf   []float64 // length np (zeroed each call by inclu2)

	// GARD-OPT-1: Smith's doubling iteration scratch. Allocated lazily
	// when the AR-large-r dispatch fires. r×r each.
	smithT  []float64
	smithT2 []float64
	smithTP []float64
	smithDP []float64
}

var paramScratchPool = sync.Pool{
	New: func() any { return &paramScratch{} },
}

func acquireParamScratch() *paramScratch { return paramScratchPool.Get().(*paramScratch) }
func releaseParamScratch(s *paramScratch) { paramScratchPool.Put(s) }

// ensureLen returns a slice with length exactly n, reusing the backing
// array of s when possible. Caller must NOT assume the returned slice is
// zeroed — see ensureLenZ for the zeroed variant.
func ensureLen(s []float64, n int) []float64 {
	if cap(s) >= n {
		return s[:n]
	}
	return make([]float64, n)
}

func ensureLenZ(s []float64, n int) []float64 {
	s = ensureLen(s, n)
	for i := range s {
		s[i] = 0
	}
	return s
}

// arTransparamsInto is the buffer-reuse variant of arTransparams. `out`
// and `tmp` should both be at least length len(params); they're grown
// via ensureLen if not. Returns out[:len(params)] populated with the
// transformed AR coefficients.
func arTransparamsInto(out, tmp, params []float64) []float64 {
	n := len(params)
	if n == 0 {
		return out[:0]
	}
	out = ensureLen(out, n)
	tmp = ensureLen(tmp, n)
	for i, p := range params {
		out[i] = math.Tanh(p)
		tmp[i] = out[i]
	}
	for j := 1; j < n; j++ {
		a := out[j]
		for k := 0; k < j; k++ {
			tmp[k] = out[k] - a*out[j-k-1]
		}
		copy(out[:j], tmp[:j])
	}
	return out
}

// maTransparamsInto: like arTransparamsInto but with the MA invertibility
// sign convention (`out[k] + a*out[j-k-1]` per inner step).
func maTransparamsInto(out, tmp, params []float64) []float64 {
	n := len(params)
	if n == 0 {
		return out[:0]
	}
	out = ensureLen(out, n)
	tmp = ensureLen(tmp, n)
	for i, p := range params {
		out[i] = math.Tanh(p)
		tmp[i] = out[i]
	}
	for j := 1; j < n; j++ {
		a := out[j]
		for k := 0; k < j; k++ {
			tmp[k] = out[k] + a*out[j-k-1]
		}
		copy(out[:j], tmp[:j])
	}
	return out
}

// expandSARMAInto: buffer-reuse variant of expandSARMA. `s` carries the
// scratch; the returned slice aliases `s.expPhiOut`, valid until the
// next call.
func expandSARMAInto(s *paramScratch, phi, Phi []float64, m int) []float64 {
	p := len(phi)
	P := len(Phi)
	if P == 0 || m <= 1 {
		s.expPhiOut = ensureLen(s.expPhiOut, p)
		copy(s.expPhiOut, phi)
		return s.expPhiOut
	}
	a := ensureLen(s.expPhiA, p+1)
	a[0] = 1
	for i, v := range phi {
		a[i+1] = -v
	}
	s.expPhiA = a
	b := ensureLenZ(s.expPhiB, P*m+1)
	b[0] = 1
	for i, v := range Phi {
		b[(i+1)*m] = -v
	}
	s.expPhiB = b
	c := ensureLenZ(s.expPhiC, len(a)+len(b)-1)
	for i, ai := range a {
		if ai == 0 {
			continue
		}
		for j, bj := range b {
			c[i+j] += ai * bj
		}
	}
	s.expPhiC = c
	out := ensureLen(s.expPhiOut, len(c)-1)
	for i := 1; i < len(c); i++ {
		out[i-1] = -c[i]
	}
	s.expPhiOut = out
	return out
}

// expandSMAInto: buffer-reuse variant of expandSMA. Returns slice
// aliasing s.expThetaOut.
func expandSMAInto(s *paramScratch, theta, Theta []float64, m int) []float64 {
	q := len(theta)
	Q := len(Theta)
	if Q == 0 || m <= 1 {
		s.expThetaOut = ensureLen(s.expThetaOut, q)
		copy(s.expThetaOut, theta)
		return s.expThetaOut
	}
	a := ensureLen(s.expThetaA, q+1)
	a[0] = 1
	for i, v := range theta {
		a[i+1] = v
	}
	s.expThetaA = a
	b := ensureLenZ(s.expThetaB, Q*m+1)
	b[0] = 1
	for i, v := range Theta {
		b[(i+1)*m] = v
	}
	s.expThetaB = b
	c := ensureLenZ(s.expThetaC, len(a)+len(b)-1)
	for i, ai := range a {
		if ai == 0 {
			continue
		}
		for j, bj := range b {
			c[i+j] += ai * bj
		}
	}
	s.expThetaC = c
	out := ensureLen(s.expThetaOut, len(c)-1)
	for i := 1; i < len(c); i++ {
		out[i-1] = c[i]
	}
	s.expThetaOut = out
	return out
}

// unpackParamsXInto: buffer-reuse variant of unpackParamsX. The phi /
// theta / sPhi / sTheta slices alias scratch buffers in `s` and are
// valid until the next call on the same scratch.
func unpackParamsXInto(s *paramScratch, params []float64, p, q, P, Q int, withIntercept bool, k int) (phi, theta, sPhi, sTheta []float64, c float64, beta []float64) {
	idx := 0
	if p > 0 {
		s.transARPhi = arTransparamsInto(s.transARPhi, s.transARPhiTmp, params[idx:idx+p])
		s.transARPhiTmp = ensureLen(s.transARPhiTmp, p)
		phi = s.transARPhi
		idx += p
	}
	if q > 0 {
		s.transMATheta = maTransparamsInto(s.transMATheta, s.transMAThetaTmp, params[idx:idx+q])
		s.transMAThetaTmp = ensureLen(s.transMAThetaTmp, q)
		theta = s.transMATheta
		idx += q
	}
	if P > 0 {
		s.transARSPhi = arTransparamsInto(s.transARSPhi, s.transARSPhiTmp, params[idx:idx+P])
		s.transARSPhiTmp = ensureLen(s.transARSPhiTmp, P)
		sPhi = s.transARSPhi
		idx += P
	}
	if Q > 0 {
		s.transMASTheta = maTransparamsInto(s.transMASTheta, s.transMASThetaTmp, params[idx:idx+Q])
		s.transMASThetaTmp = ensureLen(s.transMASThetaTmp, Q)
		sTheta = s.transMASTheta
		idx += Q
	}
	if withIntercept {
		c = params[idx]
		idx++
	}
	if k > 0 {
		s.beta = ensureLen(s.beta, k)
		copy(s.beta, params[idx:idx+k])
		beta = s.beta
	}
	return
}
