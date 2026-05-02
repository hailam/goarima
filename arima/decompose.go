package arima

import (
	"errors"
	"math"
)

// DecomposeType selects additive or multiplicative decomposition.
type DecomposeType int

const (
	// Additive: x = trend + seasonal + random
	Additive DecomposeType = iota
	// Multiplicative: x = trend * seasonal * random
	Multiplicative
)

// Decomposed holds the four components returned by Decompose.
type Decomposed struct {
	X        []float64
	Trend    []float64
	Seasonal []float64
	Random   []float64
}

// Decompose performs trend-seasonal-residual decomposition with a centered
// moving-average filter, mirroring R's `stats::decompose` and pmdarima's
// `arima.seasonality.decompose`.
//
// m must be >= 2; len(x) / m must be >= 2 (need at least two full seasonal cycles).
// filter (optional) is applied as the moving-average kernel; if nil, a length-m
// uniform mean filter is used.
func Decompose(x []float64, dtype DecomposeType, m int, filter []float64) (*Decomposed, error) {
	if m < 2 {
		return nil, errors.New("m must be >= 2")
	}
	if dtype != Additive && dtype != Multiplicative {
		return nil, errors.New("type must be Additive or Multiplicative")
	}
	if float64(len(x))/float64(m) < 2 {
		return nil, errors.New("series too short: need at least 2 periods")
	}
	if filter == nil {
		filter = make([]float64, m)
		for i := range filter {
			filter[i] = 1.0 / float64(m)
		}
	}
	isMOdd := m%2 == 1
	halfM := m / 2

	// trend = convolve(x, filter, mode='valid')
	trend := convolveValid(x, filter)
	if !isMOdd {
		// drop final index for even m, matching numpy convolve+valid behaviour
		trend = trend[:len(trend)-1]
	}

	// detrend over indices [halfM, halfM+len(trend))
	detrend := make([]float64, len(trend))
	for i := range trend {
		idx := halfM + i
		if idx >= len(x) {
			break
		}
		detrend[i] = combineInverse(dtype, x[idx], trend[i])
	}

	// pad detrend to a multiple of m
	numSeasons := int(math.Ceil(float64(len(trend)) / float64(m)))
	padLen := numSeasons*m - len(trend)
	padded := append([]float64{}, detrend...)
	for i := 0; i < padLen; i++ {
		padded = append(padded, math.NaN())
	}

	// reshape to (numSeasons, m), nanmean per column
	seasonalShort := make([]float64, m)
	for col := 0; col < m; col++ {
		sum := 0.0
		count := 0
		for row := 0; row < numSeasons; row++ {
			v := padded[row*m+col]
			if !math.IsNaN(v) {
				sum += v
				count++
			}
		}
		if count > 0 {
			seasonalShort[col] = sum / float64(count)
		}
	}
	// rotate: seasonal = seasonal[halfM:] + seasonal[:halfM]
	rotated := make([]float64, m)
	copy(rotated, seasonalShort[halfM:])
	copy(rotated[m-halfM:], seasonalShort[:halfM])

	// tile rotated `numSeasons+1` times then trim
	seasonal := make([]float64, 0, (numSeasons+1)*m)
	for i := 0; i <= numSeasons; i++ {
		seasonal = append(seasonal, rotated...)
	}
	if padLen > 0 {
		seasonal = seasonal[:len(seasonal)-padLen]
	}
	if isMOdd {
		seasonal = seasonal[:len(seasonal)-1]
	}
	if len(seasonal) > len(x) {
		seasonal = seasonal[:len(x)]
	} else {
		// pad up to len(x) with NaN
		for len(seasonal) < len(x) {
			seasonal = append(seasonal, math.NaN())
		}
	}

	// pad trend to len(x) by halfM NaNs on each end
	paddedTrend := make([]float64, halfM)
	for i := range paddedTrend {
		paddedTrend[i] = math.NaN()
	}
	paddedTrend = append(paddedTrend, trend...)
	for len(paddedTrend) < len(x) {
		paddedTrend = append(paddedTrend, math.NaN())
	}
	if len(paddedTrend) > len(x) {
		paddedTrend = paddedTrend[:len(x)]
	}

	// random = combineInverse(combineInverse(x, trend), seasonal)
	random := make([]float64, len(x))
	for i := range x {
		stage1 := combineInverse(dtype, x[i], paddedTrend[i])
		random[i] = combineInverse(dtype, stage1, seasonal[i])
	}

	return &Decomposed{
		X:        append([]float64{}, x...),
		Trend:    paddedTrend,
		Seasonal: seasonal,
		Random:   random,
	}, nil
}

func combineInverse(t DecomposeType, a, b float64) float64 {
	switch t {
	case Multiplicative:
		if b == 0 || math.IsNaN(b) {
			return math.NaN()
		}
		return a / b
	default:
		if math.IsNaN(b) {
			return math.NaN()
		}
		return a - b
	}
}

// convolveValid returns the "valid" subset of np.convolve(x, filter): output
// length = len(x) - len(filter) + 1. Filter is applied in reverse (numpy/R
// convention).
func convolveValid(x, filter []float64) []float64 {
	n := len(x)
	k := len(filter)
	if k > n {
		return nil
	}
	out := make([]float64, n-k+1)
	for i := 0; i < n-k+1; i++ {
		s := 0.0
		for j := 0; j < k; j++ {
			s += x[i+j] * filter[k-1-j]
		}
		out[i] = s
	}
	return out
}
