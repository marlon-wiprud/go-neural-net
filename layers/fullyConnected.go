package layer

import "go-neural-net/matrix"

type fcLayer struct {
	weights *matrix.Matrix
	bias    *matrix.Matrix
	input   *matrix.Matrix
	output  *matrix.Matrix
}

func NewFCLayer(inputSize, outputSize int) Layer {
	w := matrix.NewMatrix(inputSize, outputSize)
	w.InitRandom()
	w.Sub(0.5)

	b := matrix.NewMatrix(1, outputSize)
	b.InitRandom()
	b.Sub(0.5)

	return &fcLayer{
		weights: w,
		bias:    b,
	}
}

func (fc *fcLayer) ForwardPropogation(input *matrix.Matrix) (*matrix.Matrix, error) {
	var err error

	*fc.input = *input // copy data, dont reassign reference
	fc.output, err = input.Dot(fc.weights).AddMatrix(fc.bias)
	if err != nil {
		return nil, err
	}

	return fc.output, nil
}

func (fc *fcLayer) BackwardPropogation(output_error *matrix.Matrix, learningRate float64) (*matrix.Matrix, error) {
	var err error

	inputError := output_error.Dot(fc.weights.Transpose())
	weightsError := fc.input.Transpose().Dot(output_error)

	fc.weights, err = fc.weights.SubMatrix(weightsError.Mul(learningRate))
	if err != nil {
		return nil, err
	}

	fc.bias, err = fc.bias.SubMatrix(output_error.Mul(learningRate))
	if err != nil {
		return nil, err
	}

	return inputError, nil
}
