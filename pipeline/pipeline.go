// Package pipeline composes endogenous/exogenous transformers with a final
// ARIMA estimator, mirroring pmdarima.pipeline.Pipeline.
package pipeline

import (
	"errors"
	"fmt"

	"github.com/hailam/goarima/arima"
)

// EndogTransformer modifies the endogenous series y in place and is invertible.
type EndogTransformer interface {
	Fit(y []float64) error
	Transform(y []float64) ([]float64, error)
	InverseTransform(y []float64) ([]float64, error)
}

// ExogFeaturizer derives an exogenous matrix from y. It does not modify y.
type ExogFeaturizer interface {
	Fit(y []float64) error
	Transform(y []float64, x [][]float64, nPeriods int) ([][]float64, error)
	UpdateAndTransform(y []float64, x [][]float64) ([][]float64, error)
}

// Step represents one stage in a Pipeline. Exactly one of Endog/Exog must be set.
type Step struct {
	Name  string
	Endog EndogTransformer
	Exog  ExogFeaturizer
}

// Pipeline chains transformers with a final ARIMA estimator.
type Pipeline struct {
	Steps []Step
	Model *arima.ARIMA

	// Cached state from last Fit.
	endogChain []EndogTransformer
	exogChain  []ExogFeaturizer
	fitted     bool

	// Original-scale training data, kept so Update / Refit can rebuild the
	// transformer-chain inputs for the combined (history + new) series.
	// The model's own m.yTrain is on the post-transform scale, so we need
	// our own copy in original units.
	yTrain []float64
	xTrain [][]float64
}

// NewPipeline constructs a pipeline from steps and a final estimator.
//
// Each step's Name must be non-empty and unique.
func NewPipeline(steps []Step, model *arima.ARIMA) (*Pipeline, error) {
	if model == nil {
		return nil, errors.New("final estimator must be non-nil")
	}
	seen := map[string]bool{}
	for i, s := range steps {
		if s.Name == "" {
			return nil, fmt.Errorf("step %d: name required", i)
		}
		if seen[s.Name] {
			return nil, fmt.Errorf("duplicate step name %q", s.Name)
		}
		seen[s.Name] = true
		if (s.Endog == nil) == (s.Exog == nil) {
			return nil, fmt.Errorf("step %q: exactly one of Endog/Exog must be set", s.Name)
		}
	}
	return &Pipeline{Steps: steps, Model: model}, nil
}

// Fit applies each transformer in sequence and fits the final model.
//
// y is the endogenous series; X is the optional exogenous matrix.
func (p *Pipeline) Fit(y []float64, x [][]float64) error {
	yc := append([]float64{}, y...)
	xc := cloneMatrix(x)
	endogChain := []EndogTransformer{}
	exogChain := []ExogFeaturizer{}
	for _, s := range p.Steps {
		if s.Endog != nil {
			if err := s.Endog.Fit(yc); err != nil {
				return fmt.Errorf("fit step %q: %w", s.Name, err)
			}
			yt, err := s.Endog.Transform(yc)
			if err != nil {
				return fmt.Errorf("transform step %q: %w", s.Name, err)
			}
			yc = yt
			endogChain = append(endogChain, s.Endog)
		} else {
			if err := s.Exog.Fit(yc); err != nil {
				return fmt.Errorf("fit step %q: %w", s.Name, err)
			}
			xt, err := s.Exog.Transform(yc, xc, 0)
			if err != nil {
				return fmt.Errorf("transform step %q: %w", s.Name, err)
			}
			xc = xt
			exogChain = append(exogChain, s.Exog)
		}
	}
	var fitX [][]float64
	if len(xc) > 0 {
		fitX = xc
	}
	if err := p.Model.Fit(yc, fitX); err != nil {
		return fmt.Errorf("estimator fit: %w", err)
	}
	p.endogChain = endogChain
	p.exogChain = exogChain
	// Save the ORIGINAL-scale training data (pre-transform) so Update /
	// Refit can rebuild the transformer-chain inputs for the combined
	// (history + new) series.
	p.yTrain = append([]float64(nil), y...)
	p.xTrain = cloneMatrix(x)
	p.fitted = true
	return nil
}

// Update appends new observations and warm-starts the underlying model's
// MLE refresh. The transformer chain's fit-time state is preserved — only
// the model's parameters move. Mirrors pmdarima.pipeline.Pipeline.update.
//
// `newY` is on the original (untransformed) input scale; `newX` is the
// matching raw exog rows (or nil when the pipeline derives all exog from
// featurizers). The pipeline:
//
//  1. Concatenates newY/newX with the original-scale training data.
//  2. Applies each EndogTransformer.Transform (NOT Fit — the transformer
//     keeps its calibrated state) to produce the model-scale combined series.
//  3. Calls UpdateAndTransform on each ExogFeaturizer to extend its
//     internal feature index to cover the combined range.
//  4. Slices off the last len(newY) rows and passes them to m.Update for
//     the warm-start MLE refresh.
//
// Errors at any stage roll back without mutating pipeline state.
func (p *Pipeline) Update(newY []float64, newX [][]float64) error {
	if !p.fitted {
		return errors.New("pipeline not fitted")
	}
	yc, xc, err := p.transformCombined(newY, newX)
	if err != nil {
		return err
	}
	// Slice off the new tail to pass to m.Update — it appends to its own
	// m.yTrain internally and runs a short BFGS warm-start.
	tailY := yc[len(yc)-len(newY):]
	var tailX [][]float64
	if len(xc) > 0 {
		tailX = xc[len(xc)-len(newY):]
	}
	if err := p.Model.Update(tailY, tailX); err != nil {
		return fmt.Errorf("estimator update: %w", err)
	}
	// Commit: extend our original-scale training cache.
	p.yTrain = append(p.yTrain, newY...)
	p.xTrain = append(p.xTrain, cloneMatrix(newX)...)
	return nil
}

// Refit appends new observations and runs a full cold-start fit on the
// combined series — Hannan-Rissanen warmup, full BFGS, and Nelder-Mead
// polish. Same transformer-chain semantics as Update; only the underlying
// model's fit strategy differs.
//
// Like ARIMA.Refit, the pipeline's transformer steps and the underlying
// ARIMA orders are preserved — to re-search ARIMA orders, run AutoArima
// fresh on the combined data.
func (p *Pipeline) Refit(newY []float64, newX [][]float64) error {
	if !p.fitted {
		return errors.New("pipeline not fitted")
	}
	yc, xc, err := p.transformCombined(newY, newX)
	if err != nil {
		return err
	}
	tailY := yc[len(yc)-len(newY):]
	var tailX [][]float64
	if len(xc) > 0 {
		tailX = xc[len(xc)-len(newY):]
	}
	if err := p.Model.Refit(tailY, tailX); err != nil {
		return fmt.Errorf("estimator refit: %w", err)
	}
	p.yTrain = append(p.yTrain, newY...)
	p.xTrain = append(p.xTrain, cloneMatrix(newX)...)
	return nil
}

// transformCombined builds the (history + new) combined series, runs it
// through every endog transformer (Transform-only, no re-fit), and through
// every exog featurizer's UpdateAndTransform. Returns the post-chain y/x
// covering the FULL combined range.
func (p *Pipeline) transformCombined(newY []float64, newX [][]float64) ([]float64, [][]float64, error) {
	if len(newY) == 0 {
		return nil, nil, errors.New("newY must be non-empty")
	}
	if newX != nil && len(newX) != len(newY) {
		return nil, nil, fmt.Errorf("newX rows (%d) must match newY length (%d)", len(newX), len(newY))
	}
	yc := make([]float64, 0, len(p.yTrain)+len(newY))
	yc = append(yc, p.yTrain...)
	yc = append(yc, newY...)
	xc := cloneMatrix(p.xTrain)
	if newX != nil {
		xc = append(xc, cloneMatrix(newX)...)
	}
	// Apply each EndogTransformer (no re-fit — keep calibrated state).
	for _, t := range p.endogChain {
		yt, err := t.Transform(yc)
		if err != nil {
			return nil, nil, fmt.Errorf("update transform: %w", err)
		}
		yc = yt
	}
	// Apply each ExogFeaturizer's UpdateAndTransform — they internally
	// extend their feature index (e.g. dates) to cover the combined range
	// and return features for the FULL post-update period.
	for _, f := range p.exogChain {
		xt, err := f.UpdateAndTransform(yc, xc)
		if err != nil {
			return nil, nil, fmt.Errorf("update featurizer: %w", err)
		}
		xc = xt
	}
	return yc, xc, nil
}

// Predict produces a forecast on the original (untransformed) scale.
//
// If the pipeline contains an exog featurizer, future exog rows are derived
// from the featurizer (n_periods passed through). futureExog (optional) is a
// user-supplied exog matrix that gets concatenated with featurizer output.
func (p *Pipeline) Predict(nPeriods int, alpha float64, futureExog [][]float64) (forecast, lower, upper []float64, err error) {
	if !p.fitted {
		return nil, nil, nil, errors.New("pipeline not fitted")
	}
	var modelX [][]float64
	if len(p.exogChain) > 0 {
		// Reconstruct future exog from each featurizer in order. We pass the
		// optional user-supplied futureExog as the seed and let each
		// featurizer column-bind its own contribution on the right.
		xseed := futureExog
		for _, f := range p.exogChain {
			out, e := f.Transform(nil, xseed, nPeriods)
			if e != nil {
				return nil, nil, nil, fmt.Errorf("future exog from featurizer: %w", e)
			}
			xseed = out
		}
		modelX = xseed
	} else {
		// No featurizer in the pipeline: pass user-supplied futureExog
		// straight through. Pre-fix this branch left modelX nil and silently
		// dropped any futureExog the user provided, even when the underlying
		// model was fitted with raw exog.
		modelX = futureExog
	}
	fc, lo, hi, err := p.Model.Predict(nPeriods, alpha, modelX)
	if err != nil {
		return nil, nil, nil, err
	}
	// Apply endog inverse transforms in reverse order.
	for i := len(p.endogChain) - 1; i >= 0; i-- {
		var inv []float64
		inv, err = p.endogChain[i].InverseTransform(fc)
		if err != nil {
			return nil, nil, nil, err
		}
		fc = inv
		if lo != nil {
			lo, _ = p.endogChain[i].InverseTransform(lo)
			hi, _ = p.endogChain[i].InverseTransform(hi)
		}
	}
	return fc, lo, hi, nil
}

func cloneMatrix(x [][]float64) [][]float64 {
	if x == nil {
		return nil
	}
	out := make([][]float64, len(x))
	for i, row := range x {
		r := make([]float64, len(row))
		copy(r, row)
		out[i] = r
	}
	return out
}
