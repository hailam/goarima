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
	if len(xc) > 0 {
		// Until ARIMA accepts exogenous regressors, refuse to silently drop them.
		return errors.New("pipeline: exog matrix is not yet plumbed through to ARIMA")
	}
	if err := p.Model.Fit(yc); err != nil {
		return fmt.Errorf("estimator fit: %w", err)
	}
	p.endogChain = endogChain
	p.exogChain = exogChain
	p.fitted = true
	return nil
}

// Predict produces a forecast on the original (untransformed) scale.
func (p *Pipeline) Predict(nPeriods int, alpha float64) (forecast, lower, upper []float64, err error) {
	if !p.fitted {
		return nil, nil, nil, errors.New("pipeline not fitted")
	}
	fc, lo, hi, err := p.Model.Predict(nPeriods, alpha)
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
