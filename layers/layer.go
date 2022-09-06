package layer

import "go-neural-net/matrix"

type Layer interface {
	ForwardPropogation(input *matrix.Matrix) (output *matrix.Matrix, err error)
	BackwardPropogation(output_error *matrix.Matrix, learningRate float64) (output *matrix.Matrix, err error)
}
