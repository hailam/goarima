package arima

import (
	"encoding/json"
	"fmt"
	"io"
)

// SerializationVersion is the current on-disk format version. Bumped on any
// change to the serialized shape that older binaries cannot read.
const SerializationVersion = 1

// arimaSnapshot is the serialized representation of a fitted ARIMA model.
//
// All fields are exported so encoding/json sees them. Method and
// DiffuseConvention are emitted as strings (not int enum values) so the
// format survives reordering of the const blocks. Lambda is emitted as a
// pointer so nil round-trips correctly.
type arimaSnapshot struct {
	Version int `json:"version"`

	// Configuration
	Order                 Order         `json:"order"`
	Seasonal              SeasonalOrder `json:"seasonal"`
	WithIntercept         bool          `json:"with_intercept"`
	Method                string        `json:"method"`
	MaxIter               int           `json:"max_iter"`
	Lambda                *float64      `json:"lambda,omitempty"`
	Lambda2               float64       `json:"lambda2,omitempty"`
	NonSimpleDifferencing bool          `json:"non_simple_differencing,omitempty"`
	DiffuseConvention     string        `json:"diffuse_convention"`

	// Fitted parameters
	Phi    []float64 `json:"phi"`     // non-seasonal AR (lowercase phi in math)
	Theta  []float64 `json:"theta"`   // non-seasonal MA
	SPhi   []float64 `json:"s_phi"`   // seasonal AR (uppercase Phi in math)
	STheta []float64 `json:"s_theta"` // seasonal MA
	C      float64   `json:"c"`
	Mean   float64   `json:"mean"`
	Beta   []float64 `json:"beta,omitempty"`

	// Fitted summary statistics
	Sigma2  float64 `json:"sigma2"`
	LogL    float64 `json:"log_l"`
	Nobs    int     `json:"nobs"`
	NExog   int     `json:"n_exog"`
	PsiInfN int     `json:"psi_inf_n"`

	// Fitted state needed for Predict / FittedValues / Resid
	Resids []float64   `json:"resids"`
	YTrain []float64   `json:"y_train"`
	XTrain [][]float64 `json:"x_train,omitempty"`
	PsiInf []float64   `json:"psi_inf,omitempty"`
}

// Method <-> string mapping. Used for forward-compat serialization so that
// reordering or extending the Method const block doesn't invalidate saved
// models.
var methodToString = map[Method]string{
	MethodCSS:   "css",
	MethodML:    "ml",
	MethodCSSML: "css-ml",
}

var stringToMethod = func() map[string]Method {
	m := make(map[string]Method, len(methodToString))
	for k, v := range methodToString {
		m[v] = k
	}
	return m
}()

var diffuseToString = map[DiffuseConv]string{
	DiffuseR:           "r",
	DiffuseStatsmodels: "statsmodels",
}

var stringToDiffuse = func() map[string]DiffuseConv {
	m := make(map[string]DiffuseConv, len(diffuseToString))
	for k, v := range diffuseToString {
		m[v] = k
	}
	return m
}()

// MarshalJSON implements encoding/json.Marshaler. Saving an unfitted model
// is a programming error and returns an error rather than producing a
// half-state snapshot.
func (m *ARIMA) MarshalJSON() ([]byte, error) {
	if !m.fitted {
		return nil, fmt.Errorf("arima: cannot serialize unfitted model — call Fit first")
	}
	methodStr, ok := methodToString[m.Method]
	if !ok {
		return nil, fmt.Errorf("arima: unknown Method %d", m.Method)
	}
	diffStr, ok := diffuseToString[m.DiffuseConvention]
	if !ok {
		return nil, fmt.Errorf("arima: unknown DiffuseConvention %d", m.DiffuseConvention)
	}
	snap := arimaSnapshot{
		Version:               SerializationVersion,
		Order:                 m.Order,
		Seasonal:              m.Seasonal,
		WithIntercept:         m.WithIntercept,
		Method:                methodStr,
		MaxIter:               m.MaxIter,
		Lambda:                m.Lambda,
		Lambda2:               m.Lambda2,
		NonSimpleDifferencing: m.NonSimpleDifferencing,
		DiffuseConvention:     diffStr,
		Phi:                   m.phi,
		Theta:                 m.theta,
		SPhi:                  m.Phi,
		STheta:                m.Theta,
		C:                     m.c,
		Mean:                  m.mean,
		Beta:                  m.beta,
		Sigma2:                m.sigma2,
		LogL:                  m.logL,
		Nobs:                  m.nobs,
		NExog:                 m.nExog,
		PsiInfN:               m.psiInfN,
		Resids:                m.resids,
		YTrain:                m.yTrain,
		XTrain:                m.xTrain,
		PsiInf:                m.psiInf,
	}
	return json.Marshal(&snap)
}

// UnmarshalJSON implements encoding/json.Unmarshaler. Validates the version,
// the Method/DiffuseConvention strings, and basic coherence (parameter slice
// lengths match Order). Regenerates the Predict caches so the loaded model
// is immediately ready to forecast.
func (m *ARIMA) UnmarshalJSON(data []byte) error {
	var snap arimaSnapshot
	if err := json.Unmarshal(data, &snap); err != nil {
		return fmt.Errorf("arima: decode snapshot: %w", err)
	}
	if snap.Version != SerializationVersion {
		return fmt.Errorf("arima: unsupported serialization version %d (want %d)",
			snap.Version, SerializationVersion)
	}
	method, ok := stringToMethod[snap.Method]
	if !ok {
		return fmt.Errorf("arima: unknown Method %q", snap.Method)
	}
	diffuse, ok := stringToDiffuse[snap.DiffuseConvention]
	if !ok {
		return fmt.Errorf("arima: unknown DiffuseConvention %q", snap.DiffuseConvention)
	}
	if snap.Order.P < 0 || snap.Order.D < 0 || snap.Order.Q < 0 {
		return fmt.Errorf("arima: negative Order field in snapshot: %+v", snap.Order)
	}
	if snap.Seasonal.P < 0 || snap.Seasonal.D < 0 || snap.Seasonal.Q < 0 || snap.Seasonal.M < 0 {
		return fmt.Errorf("arima: negative Seasonal field in snapshot: %+v", snap.Seasonal)
	}
	if len(snap.Phi) != snap.Order.P {
		return fmt.Errorf("arima: phi length %d != Order.P %d", len(snap.Phi), snap.Order.P)
	}
	if len(snap.Theta) != snap.Order.Q {
		return fmt.Errorf("arima: theta length %d != Order.Q %d", len(snap.Theta), snap.Order.Q)
	}
	if snap.Seasonal.Active() {
		if len(snap.SPhi) != snap.Seasonal.P {
			return fmt.Errorf("arima: seasonal Phi length %d != Seasonal.P %d", len(snap.SPhi), snap.Seasonal.P)
		}
		if len(snap.STheta) != snap.Seasonal.Q {
			return fmt.Errorf("arima: seasonal Theta length %d != Seasonal.Q %d", len(snap.STheta), snap.Seasonal.Q)
		}
	}
	if snap.NExog < 0 {
		return fmt.Errorf("arima: negative NExog %d", snap.NExog)
	}
	if snap.NExog > 0 && len(snap.Beta) != snap.NExog {
		return fmt.Errorf("arima: beta length %d != NExog %d", len(snap.Beta), snap.NExog)
	}
	if len(snap.YTrain) == 0 {
		return fmt.Errorf("arima: yTrain is empty")
	}
	// nobs should equal len(yTrain) - (d + D*M). Every fit path sets it that
	// way, and every Predict / FittedValues path assumes it. Reject snapshots
	// where this invariant is broken — they'd silently produce wrong forecasts.
	dHead := snap.Order.D
	if snap.Seasonal.Active() {
		dHead += snap.Seasonal.D * snap.Seasonal.M
	}
	wantNobs := len(snap.YTrain) - dHead
	if wantNobs < 1 {
		return fmt.Errorf("arima: yTrain length %d insufficient for differencing head %d",
			len(snap.YTrain), dHead)
	}
	if snap.Nobs != wantNobs {
		return fmt.Errorf("arima: nobs %d inconsistent with yTrain length %d minus diff head %d",
			snap.Nobs, len(snap.YTrain), dHead)
	}
	if len(snap.Resids) != snap.Nobs {
		return fmt.Errorf("arima: resids length %d != nobs %d", len(snap.Resids), snap.Nobs)
	}
	if snap.NExog > 0 {
		if len(snap.XTrain) != len(snap.YTrain) {
			return fmt.Errorf("arima: xTrain rows %d != yTrain length %d",
				len(snap.XTrain), len(snap.YTrain))
		}
		for i, row := range snap.XTrain {
			if len(row) != snap.NExog {
				return fmt.Errorf("arima: xTrain row %d cols %d != NExog %d",
					i, len(row), snap.NExog)
			}
		}
	} else if len(snap.XTrain) != 0 {
		return fmt.Errorf("arima: xTrain non-empty but NExog is 0")
	}

	m.Order = snap.Order
	m.Seasonal = snap.Seasonal
	m.WithIntercept = snap.WithIntercept
	m.Method = method
	m.MaxIter = snap.MaxIter
	m.Lambda = snap.Lambda
	m.Lambda2 = snap.Lambda2
	m.NonSimpleDifferencing = snap.NonSimpleDifferencing
	m.DiffuseConvention = diffuse
	m.phi = snap.Phi
	m.theta = snap.Theta
	m.Phi = snap.SPhi
	m.Theta = snap.STheta
	m.c = snap.C
	m.mean = snap.Mean
	m.beta = snap.Beta
	m.sigma2 = snap.Sigma2
	m.logL = snap.LogL
	m.nobs = snap.Nobs
	m.nExog = snap.NExog
	m.psiInfN = snap.PsiInfN
	m.resids = snap.Resids
	m.yTrain = snap.YTrain
	m.xTrain = snap.XTrain
	m.psiInf = snap.PsiInf
	m.fitted = true

	// Regenerate Predict caches from yTrain. yMSCache is yTrain after Box-Cox;
	// wsCenteredCache is the differenced+centered series equivalent to
	// residOf(best) at fit time.
	if err := m.rebuildPredictCaches(); err != nil {
		m.fitted = false
		return fmt.Errorf("arima: rebuild Predict caches: %w", err)
	}
	return nil
}

// rebuildPredictCaches regenerates yMSCache and wsCenteredCache from yTrain
// after a Load. Mirrors the cache-population path inside Fit (line ~152 +
// ~528 in arima.go).
func (m *ARIMA) rebuildPredictCaches() error {
	yMS := append([]float64(nil), m.yTrain...)
	if m.Lambda != nil {
		t, err := boxCoxApply(yMS, *m.Lambda, m.Lambda2)
		if err != nil {
			return err
		}
		yMS = t
	}
	m.yMSCache = yMS

	// Differenced training series.
	ws := yMS
	if m.Order.D > 0 {
		ws = applyDiff(ws, 1, m.Order.D)
	}
	if m.Seasonal.Active() && m.Seasonal.D > 0 {
		ws = applyDiff(ws, m.Seasonal.M, m.Seasonal.D)
	}
	// Differenced exog (column-wise).
	var wX [][]float64
	if m.xTrain != nil {
		wX = cloneMat(m.xTrain)
		if m.Order.D > 0 {
			wX = applyMatDiff(wX, 1, m.Order.D)
		}
		if m.Seasonal.Active() && m.Seasonal.D > 0 {
			wX = applyMatDiff(wX, m.Seasonal.M, m.Seasonal.D)
		}
	}
	wsCentered := make([]float64, len(ws))
	for i, v := range ws {
		r := v - m.mean - m.c
		if wX != nil {
			for j, b := range m.beta {
				r -= b * wX[i][j]
			}
		}
		wsCentered[i] = r
	}
	m.wsCenteredCache = wsCentered
	return nil
}

// Save writes the fitted model to w as JSON. Equivalent to
// `json.NewEncoder(w).Encode(m)` but without the trailing newline.
func (m *ARIMA) Save(w io.Writer) error {
	b, err := m.MarshalJSON()
	if err != nil {
		return err
	}
	_, err = w.Write(b)
	return err
}

// LoadARIMA reads a JSON-serialized ARIMA model from r and returns it ready
// for Predict. The reader is consumed in full.
func LoadARIMA(r io.Reader) (*ARIMA, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("arima: read snapshot: %w", err)
	}
	m := &ARIMA{}
	if err := m.UnmarshalJSON(data); err != nil {
		return nil, err
	}
	return m, nil
}
