package layer

import "go-neural-net/matrix"

type ActivationFunc func(x float64) (y float64)

type activationLayer struct {
	activation      ActivationFunc
	activationPrime ActivationFunc
	input           *matrix.Matrix
	output          *matrix.Matrix
}

// func NewActivationLayer(activation, activationPrime ActivationFunc) Layer {
// 	return &activationLayer{activation: activation, activationPrime: activationPrime}
// }

// func (al *activationLayer) ForwardPropogation(input *matrix.Matrix) (output *matrix.Matrix, err error) {
// 	*al.input = *input
// 	// TODO need a function to apply activation func to a matrix
// }
