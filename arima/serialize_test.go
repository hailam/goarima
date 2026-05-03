package arima

import (
	"bytes"
	"encoding/json"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/hailam/goarima/datasets"
)

// roundTripPredictParity asserts that Save → Load preserves Predict output
// exactly. Tests both the MarshalJSON/UnmarshalJSON path and Save/Load.
func roundTripPredictParity(t *testing.T, m *ARIMA, nPeriods int, futureExog [][]float64, alpha float64) {
	t.Helper()
	wantFc, wantLo, wantHi, err := m.Predict(nPeriods, alpha, futureExog)
	if err != nil {
		t.Fatalf("original Predict: %v", err)
	}

	// Path 1: encoding/json round-trip.
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	loaded := &ARIMA{}
	if err := json.Unmarshal(data, loaded); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	gotFc, gotLo, gotHi, err := loaded.Predict(nPeriods, alpha, futureExog)
	if err != nil {
		t.Fatalf("loaded Predict (json round-trip): %v", err)
	}
	assertSeriesEqual(t, "json forecast", wantFc, gotFc)
	assertSeriesEqual(t, "json lower", wantLo, gotLo)
	assertSeriesEqual(t, "json upper", wantHi, gotHi)

	// Path 2: Save/LoadARIMA round-trip.
	var buf bytes.Buffer
	if err := m.Save(&buf); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded2, err := LoadARIMA(&buf)
	if err != nil {
		t.Fatalf("LoadARIMA: %v", err)
	}
	gotFc2, gotLo2, gotHi2, err := loaded2.Predict(nPeriods, alpha, futureExog)
	if err != nil {
		t.Fatalf("loaded Predict (Save/Load): %v", err)
	}
	assertSeriesEqual(t, "save/load forecast", wantFc, gotFc2)
	assertSeriesEqual(t, "save/load lower", wantLo, gotLo2)
	assertSeriesEqual(t, "save/load upper", wantHi, gotHi2)
}

func assertSeriesEqual(t *testing.T, label string, want, got []float64) {
	t.Helper()
	if len(want) != len(got) {
		t.Errorf("%s: length mismatch want %d got %d", label, len(want), len(got))
		return
	}
	for i := range want {
		// Predict caches are regenerated from yTrain so floating-point sums
		// should match bit-exactly. Allow only NaN-vs-NaN equality.
		if math.IsNaN(want[i]) && math.IsNaN(got[i]) {
			continue
		}
		if want[i] != got[i] {
			t.Errorf("%s[%d]: want %g, got %g (diff %g)",
				label, i, want[i], got[i], got[i]-want[i])
		}
	}
}

func TestSerialize_RoundTripAirline011(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	m := NewARIMA(Order{P: 0, D: 1, Q: 1})
	m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
	m.MaxIter = 100
	if err := m.Fit(logAp, nil); err != nil {
		t.Fatal(err)
	}
	roundTripPredictParity(t, m, 12, nil, 0.05)
}

func TestSerialize_RoundTripAR1(t *testing.T) {
	y := simulateAR1(200, 0.6, 1.0, 42)
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.WithIntercept = true
	if err := m.Fit(y, nil); err != nil {
		t.Fatal(err)
	}
	roundTripPredictParity(t, m, 10, nil, 0)
}

func TestSerialize_RoundTripWithExog(t *testing.T) {
	// Simple AR(1) with one synthetic exog regressor.
	n := 150
	y := make([]float64, n)
	x := make([][]float64, n)
	rng := rand.New(rand.NewPCG(7, 8))
	for i := 0; i < n; i++ {
		x[i] = []float64{rng.Float64()}
		if i == 0 {
			y[i] = 1.5*x[i][0] + rng.NormFloat64()
		} else {
			y[i] = 0.5*y[i-1] + 1.5*x[i][0] + rng.NormFloat64()
		}
	}
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.WithIntercept = true
	if err := m.Fit(y, x); err != nil {
		t.Fatal(err)
	}
	futureX := make([][]float64, 5)
	for i := range futureX {
		futureX[i] = []float64{rng.Float64()}
	}
	roundTripPredictParity(t, m, 5, futureX, 0.05)
}

func TestSerialize_RoundTripBoxCox(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	lambda := 0.0 // log
	m := NewARIMA(Order{P: 0, D: 1, Q: 1})
	m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
	m.Lambda = &lambda
	m.MaxIter = 100
	if err := m.Fit(ap, nil); err != nil {
		t.Fatal(err)
	}
	roundTripPredictParity(t, m, 12, nil, 0.05)
}

func TestSerialize_RoundTripNonSimpleDifferencing(t *testing.T) {
	ap := datasets.LoadAirPassengers()
	logAp := make([]float64, len(ap))
	for i, v := range ap {
		logAp[i] = math.Log(v)
	}
	m := NewARIMA(Order{P: 0, D: 1, Q: 1})
	m.Seasonal = SeasonalOrder{P: 0, D: 1, Q: 1, M: 12}
	m.NonSimpleDifferencing = true
	m.MaxIter = 100
	if err := m.Fit(logAp, nil); err != nil {
		t.Fatal(err)
	}
	roundTripPredictParity(t, m, 12, nil, 0.05)
}

// Codex review flagged: in DiffuseStatsmodels mode, m.c and m.beta are
// rescaled AFTER wsCenteredCache is built — so a Load+Predict using the
// rescaled values may diverge from the original Predict using the cached
// (un-rescaled) wsCentered. This test confirms or refutes that.
func TestSerialize_RoundTripStatsmodelsWithIntercept(t *testing.T) {
	wi := datasets.LoadWineind()
	m := NewARIMA(Order{P: 1, D: 0, Q: 1})
	m.WithIntercept = true
	m.NonSimpleDifferencing = true
	m.DiffuseConvention = DiffuseStatsmodels
	m.MaxIter = 100
	if err := m.Fit(wi, nil); err != nil {
		t.Fatal(err)
	}
	roundTripPredictParity(t, m, 12, nil, 0.05)
}

func TestSerialize_UnfittedErrors(t *testing.T) {
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	if _, err := json.Marshal(m); err == nil {
		t.Error("expected error marshalling unfitted model")
	}
}

func TestSerialize_VersionMismatchRejected(t *testing.T) {
	body := []byte(`{"version":99,"order":{"P":1,"D":0,"Q":0}}`)
	m := &ARIMA{}
	if err := json.Unmarshal(body, m); err == nil {
		t.Error("expected error on unknown version")
	} else if !strings.Contains(err.Error(), "unsupported serialization version") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestSerialize_LengthMismatchRejected(t *testing.T) {
	// phi length 0 but Order.P = 2 → coherence check should fire.
	body := []byte(`{
		"version":1,
		"order":{"P":2,"D":0,"Q":0},
		"seasonal":{"P":0,"D":0,"Q":0,"M":0},
		"method":"css-ml",
		"diffuse_convention":"r",
		"phi":[],
		"theta":[],
		"resids":[],
		"y_train":[]
	}`)
	m := &ARIMA{}
	if err := json.Unmarshal(body, m); err == nil {
		t.Error("expected error on length mismatch")
	} else if !strings.Contains(err.Error(), "phi length") {
		t.Errorf("unexpected error: %v", err)
	}
}

// Tests for codex-flagged hardening: yTrain emptiness, nobs invariant, resids
// length, xTrain shape, negative orders.
func TestSerialize_RejectsMalformed(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string // substring of expected error
	}{
		{
			"empty yTrain",
			`{"version":1,"order":{"P":1,"D":0,"Q":0},"seasonal":{},"method":"ml","diffuse_convention":"r","phi":[0.5],"theta":[],"resids":[],"y_train":[]}`,
			"yTrain is empty",
		},
		{
			"nobs mismatch with yTrain",
			`{"version":1,"order":{"P":1,"D":1,"Q":0},"seasonal":{},"method":"ml","diffuse_convention":"r","phi":[0.5],"theta":[],"resids":[1,2,3],"y_train":[1,2,3,4,5],"nobs":99}`,
			"nobs",
		},
		{
			"resids length mismatch with nobs",
			`{"version":1,"order":{"P":1,"D":0,"Q":0},"seasonal":{},"method":"ml","diffuse_convention":"r","phi":[0.5],"theta":[],"resids":[1,2],"y_train":[1,2,3,4,5],"nobs":5}`,
			"resids length",
		},
		{
			"negative order field",
			`{"version":1,"order":{"P":-1,"D":0,"Q":0},"seasonal":{},"method":"ml","diffuse_convention":"r","phi":[],"theta":[],"resids":[],"y_train":[1]}`,
			"negative Order",
		},
		{
			"xTrain rows != yTrain length",
			`{"version":1,"order":{"P":1,"D":0,"Q":0},"seasonal":{},"method":"ml","diffuse_convention":"r","phi":[0.5],"theta":[],"resids":[1,2,3,4,5],"y_train":[1,2,3,4,5],"nobs":5,"n_exog":1,"beta":[0.1],"x_train":[[1.0],[2.0]]}`,
			"xTrain rows",
		},
		{
			"xTrain row width != NExog",
			`{"version":1,"order":{"P":1,"D":0,"Q":0},"seasonal":{},"method":"ml","diffuse_convention":"r","phi":[0.5],"theta":[],"resids":[1,2,3],"y_train":[1,2,3],"nobs":3,"n_exog":2,"beta":[0.1,0.2],"x_train":[[1,2],[3],[5,6]]}`,
			"xTrain row 1",
		},
		{
			"xTrain populated but NExog = 0",
			`{"version":1,"order":{"P":1,"D":0,"Q":0},"seasonal":{},"method":"ml","diffuse_convention":"r","phi":[0.5],"theta":[],"resids":[1,2,3],"y_train":[1,2,3],"nobs":3,"x_train":[[1],[2],[3]]}`,
			"xTrain non-empty but NExog is 0",
		},
		{
			"yTrain too short for diff head",
			`{"version":1,"order":{"P":0,"D":2,"Q":0},"seasonal":{},"method":"ml","diffuse_convention":"r","phi":[],"theta":[],"resids":[],"y_train":[1,2],"nobs":0}`,
			"insufficient for differencing",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := &ARIMA{}
			err := json.Unmarshal([]byte(tc.body), m)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not contain %q", err.Error(), tc.want)
			}
		})
	}
}

// C3: every futureExog row must have the right width.
func TestPredict_RejectsRaggedFutureExog(t *testing.T) {
	rng := rand.New(rand.NewPCG(99, 100))
	n := 100
	y := make([]float64, n)
	x := make([][]float64, n)
	for i := 0; i < n; i++ {
		x[i] = []float64{rng.Float64()}
		if i == 0 {
			y[i] = rng.NormFloat64()
		} else {
			y[i] = 0.5*y[i-1] + 1.5*x[i][0] + rng.NormFloat64()
		}
	}
	m := NewARIMA(Order{P: 1, D: 0, Q: 0})
	m.WithIntercept = true
	if err := m.Fit(y, x); err != nil {
		t.Fatal(err)
	}
	// Ragged: row 0 has 1 col, row 1 has 0 cols.
	bad := [][]float64{{0.5}, {}, {0.7}}
	if _, _, _, err := m.Predict(3, 0, bad); err == nil {
		t.Error("Predict accepted ragged futureExog")
	}
	// PredictBoot must also reject ragged input.
	if _, err := m.PredictBoot(3, 0.05, 50, 1, bad); err == nil {
		t.Error("PredictBoot accepted ragged futureExog")
	}
}
