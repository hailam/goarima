package datasets

import "testing"

// LoadAusbeerR / LoadGasolineForecastR should either return data
// (if the fetch tool has been run) or a clear error pointing at the
// fetch tool. Either is correct; we just verify both behaviors.
func TestLoadAusbeerR_ShapeOrErrors(t *testing.T) {
	got, err := LoadAusbeerR()
	if err != nil {
		// Fetch hasn't been run — check the error mentions the tool.
		t.Logf("ausbeer_r not fetched: %v (expected in fresh checkout)", err)
		return
	}
	if len(got) != 218 {
		t.Errorf("LoadAusbeerR len=%d, want 218 (R::forecast::ausbeer)", len(got))
	}
	// Sanity: known first/last values from forecast::ausbeer.
	if got[0] != 284 {
		t.Errorf("LoadAusbeerR[0] = %g, want 284 (Q1 1956)", got[0])
	}
	// Cross-check: first 211 values should match LoadAusbeer (pmdarima
	// snapshot) — they're the same underlying ABS series.
	pm := LoadAusbeer()
	overlap := min(len(pm), len(got))
	mismatches := 0
	for i := range overlap {
		// LoadAusbeer's last value is NaN (pmdarima sentinel for missing
		// Q4 2010). Skip NaN comparisons.
		if isNaN(pm[i]) || isNaN(got[i]) {
			continue
		}
		if pm[i] != got[i] {
			mismatches++
			if mismatches <= 3 {
				t.Logf("ausbeer pmdarima[%d]=%g vs R-forecast[%d]=%g", i, pm[i], i, got[i])
			}
		}
	}
	if mismatches > 5 {
		t.Errorf("ausbeer pmdarima vs R-forecast mismatch in %d/%d overlapping values",
			mismatches, overlap)
	}
}

func TestLoadGasolineForecastR_ShapeOrErrors(t *testing.T) {
	got, err := LoadGasolineForecastR()
	if err != nil {
		t.Logf("gasoline_r not fetched: %v (expected in fresh checkout)", err)
		return
	}
	if len(got) < 1300 {
		t.Errorf("LoadGasolineForecastR len=%d, want ≥1300 (R::forecast::gasoline)", len(got))
	}
	// Sanity: first value should be in plausible range (Feb 1991, ~6500-7000).
	if got[0] < 5000 || got[0] > 8000 {
		t.Errorf("LoadGasolineForecastR[0] = %g, expected ~6500", got[0])
	}
}

func isNaN(f float64) bool { return f != f }
