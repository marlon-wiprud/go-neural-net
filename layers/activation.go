package layer

import (
	"go-neural-net/matrix"
	"go-neural-net/util"
	"math"
)

type ActivationFunc func(x float64) (y float64)

type activationLayer struct {
	activation      ActivationFunc
	activationPrime ActivationFunc
	input           *matrix.Matrix
	output          *matrix.Matrix
}

func NewActivationLayer(activation, activationPrime ActivationFunc) Layer {
	return &activationLayer{activation: activation, activationPrime: activationPrime}
}

func (al *activationLayer) ForwardPropogation(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	*al.input = *input
	*al.output = *input
	al.output.ForEach(math.Tanh)
	return al.output, nil
}

func (al *activationLayer) BackwardPropogation(output_error *matrix.Matrix, learningRate float64) (output *matrix.Matrix, err error) {
	output = al.input.Copy()
	output.ForEach(util.TanhPrime)
	return output.AddMatrix(output_error)
}
