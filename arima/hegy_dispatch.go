package arima

import "errors"

// PG-114: dispatcher mapping (statType, deterministic, lagMethod) to a
// hegyRSTableID. Mirrors uroot::hegy.rs.pvalue's switch logic.

// hegyStatType selects which response-surface table family to consult.
type hegyStatType int

const (
	hegyStatJointSeasonal hegyStatType = iota // CFs — joint F_{2:m}
	hegyStatJointAll                          // CFt — joint F_{1:m}
	hegyStatPairF                             // CF  — pair F at one harmonic
	hegyStatZero                              // Ct1 — π_1 t-stat
	hegyStatNyquist                           // Ct2 — π_2 t-stat (m even)
)

func (t hegyStatType) prefix() string {
	switch t {
	case hegyStatJointSeasonal:
		return "CFs"
	case hegyStatJointAll:
		return "CFt"
	case hegyStatPairF:
		return "CF"
	case hegyStatZero:
		return "Ct1"
	case hegyStatNyquist:
		return "Ct2"
	}
	return ""
}

// HEGYDeterministic specifies the deterministic terms in the auxiliary
// regression. Format mirrors uroot's c(constant, trend, seasDummies).
// Only the four uroot-supported configurations are valid (the (0,*,*)
// "no constant" branch is rejected by uroot itself).
type HEGYDeterministic [3]int

// Common deterministic configurations. The Default matches
// forecast::nsdiffs(test="hegy") (uroot's c(1,1,0)).
var (
	HEGYDetConstant                 = HEGYDeterministic{1, 0, 0}
	HEGYDetConstantTrend            = HEGYDeterministic{1, 1, 0}
	HEGYDetConstantSeasDummies      = HEGYDeterministic{1, 0, 1}
	HEGYDetConstantTrendSeasDummies = HEGYDeterministic{1, 1, 1}
)

func (d HEGYDeterministic) code() (string, error) {
	switch d {
	case HEGYDetConstant:
		return "c", nil
	case HEGYDetConstantTrend:
		return "ct", nil
	case HEGYDetConstantSeasDummies:
		return "cD", nil
	case HEGYDetConstantTrendSeasDummies:
		return "cDt", nil
	}
	return "", errors.New("HEGY: deterministic must be one of " +
		"{1,0,0}, {1,1,0}, {1,0,1}, {1,1,1} (uroot rejects c(0,*,*))")
}

// hasConstant reports whether this configuration includes the
// intercept column.
func (d HEGYDeterministic) hasConstant() bool { return d[0] != 0 }

// hasTrend reports whether this configuration includes the linear
// trend column.
func (d HEGYDeterministic) hasTrend() bool { return d[1] != 0 }

// hasSeasonalDummies reports whether this configuration includes the
// (m-1) seasonal indicator columns.
func (d HEGYDeterministic) hasSeasonalDummies() bool { return d[2] != 0 }

// HEGYLagMethod selects how the lag-augmentation order is chosen.
type HEGYLagMethod int

const (
	// HEGYLagAIC selects the lag p ∈ [0, MaxLag] minimising AIC.
	// Default — matches forecast::nsdiffs(test="hegy").
	HEGYLagAIC HEGYLagMethod = iota
	// HEGYLagBIC selects the lag minimising BIC instead.
	HEGYLagBIC
	// HEGYLagFixed uses HEGYOpts.Lag directly.
	HEGYLagFixed
)

func (m HEGYLagMethod) suffix() (string, error) {
	switch m {
	case HEGYLagAIC:
		return "AIC", nil
	case HEGYLagBIC:
		return "BIC", nil
	case HEGYLagFixed:
		return "fijo", nil
	}
	return "", errors.New("HEGY: unknown lag method")
}

// hegyTableID resolves the (statType, deterministic, lagMethod) triple
// to a concrete hegyRSTableID via the uroot label.
func hegyTableID(stat hegyStatType, det HEGYDeterministic, lm HEGYLagMethod) (hegyRSTableID, error) {
	prefix := stat.prefix()
	if prefix == "" {
		return 0, errors.New("HEGY: invalid statType")
	}
	detCode, err := det.code()
	if err != nil {
		return 0, err
	}
	suf, err := lm.suffix()
	if err != nil {
		return 0, err
	}
	name := prefix + "_" + detCode + "_" + suf
	for i := 0; i < hegyRSTableCount; i++ {
		if hegyRSTableName[i] == name {
			return hegyRSTableID(i), nil
		}
	}
	return 0, errors.New("HEGY: no response-surface table named " + name)
}
