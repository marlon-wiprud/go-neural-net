package network

import (
	layer "go-neural-net/layers"
	"go-neural-net/matrix"
)

type LossFunc func(x *matrix.Matrix, y *matrix.Matrix) (z *matrix.Matrix)

type nn struct {
	layers        []layer.Layer
	lossFunc      LossFunc
	lossFuncPrime LossFunc
}

func NewNeuralNet(loss, lossPrime LossFunc) *nn {
	return &nn{layers: []layer.Layer{}, lossFunc: loss, lossFuncPrime: lossPrime}
}

func (n *nn) Add(l layer.Layer) {
	n.layers = append(n.layers, l)
}

func (n *nn) Predict(inputData []*matrix.Matrix) (result []*matrix.Matrix, err error) {

	for _, input := range inputData {

		output := input.Copy()

		for _, l := range n.layers {
			output, err = l.ForwardPropogation(output)
			if err != nil {
				return nil, err
			}
		}

		result = append(result, output)
	}

	return result, nil
}

func (n *nn) Fit(xTrain, yTrain []*matrix.Matrix, epochs int, learningRate float64) (err error) {

	for epoch := 0; epoch < epochs; epoch++ {
		// var e float64

		for idx, input := range xTrain {
			output := input.Copy()

			for _, l := range n.layers {
				output, err = l.ForwardPropogation(output)
				if err != nil {
					return err
				}
			}

			// e += n.lossFunc(yTrain[idx], output)

			networkErr := n.lossFuncPrime(yTrain[idx], output)

			for i := len(n.layers) - 1; i >= 0; i-- {
				networkErr, err = n.layers[i].BackwardPropogation(networkErr, learningRate)
				if err != nil {
					return err
				}
			}
		}
	}

	return nil
}
