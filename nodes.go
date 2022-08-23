package main

import (
	"fmt"
)

type Node struct {
	weights    []float64
	bias       float64
	activation float64
}

func NewNode(inputShape int) *Node {
	var weights []float64

	for i := 0; i < inputShape; i++ {
		weights = append(weights, initWeight())
	}

	return &Node{
		bias:    initWeight(),
		weights: weights,
	}
}

// must map the layer inputs to the number of nodes
// eg the layer input may have length 2, but the number of nodes in the layer is 1
// | \
// |  - | - |
// | /
type Layer struct {
	nodes []*Node
}

func (layer *Layer) ForwardPass(inputs []float64) []float64 {
	var layerOutput []float64
	for _, node := range layer.nodes {

		// compute activation based on each node weight and input value
		activation := node.bias

		// fmt.Println("node len: ", len(inputs), len(node.weights))
		for idx, input := range inputs {
			activation += input * node.weights[idx]
		}

		// set the layers activation
		// TODO is this the proper name for this "layer activation" ?
		node.activation = sigmoid(activation)
		layerOutput = append(layerOutput, node.activation)
	}

	return layerOutput
}

func (layer *Layer) BackPropInit(inDelta []float64, learningRate float64, nextLayer *Layer) (outDelta []float64) {

	// loop through each node in the layer
	for idx, node := range layer.nodes {
		// create output delta values based on the nodes activation and input deltas
		// note that this assumes that in delta is the same length as nodes
		// TODO return error if not same lenth
		e := inDelta[idx] - node.activation
		d := e * dSigmoid(node.activation)

		// build input for the next layer
		outDelta = append(outDelta, d)

		// apply change in output weights
		node.bias += d * learningRate

		for nIdx, n := range nextLayer.nodes {
			// assumes that node weights are same length as next layers nodes ???
			node.weights[nIdx] += n.activation * d * learningRate
		}
	}

	return outDelta
}

func (layer *Layer) Backprop(inDelta []float64, learningRate float64, nextLayer *Layer) (outDelta []float64) {
	for _, node := range layer.nodes {
		var err float64

		// compute error
		for idx, id := range inDelta {
			err += id * node.weights[idx]
		}

		d := err * dSigmoid(node.activation)

		outDelta = append(outDelta, d)

		node.bias += d * learningRate

		for nIdx, n := range nextLayer.nodes {
			node.weights[nIdx] += n.activation * d * learningRate
		}
	}

	return outDelta
}

type NeuralNet struct {
	layers       []*Layer
	learningRate float64
}

func (nn *NeuralNet) ForwardPass(inputs []float64) {
	// loop through layers
	layerInput := inputs

	for _, layer := range nn.layers {
		layerInput = layer.ForwardPass(layerInput)
	}
}

func (nn *NeuralNet) getNextBackpropLayer(currentIdx int, inputs []float64) *Layer {
	if currentIdx > 0 {
		return nn.layers[currentIdx-1]
	}
	return nn.inputAsLayer(inputs)
}

func (nn *NeuralNet) inputAsLayer(inputs []float64) *Layer {
	layer := new(Layer)

	for _, input := range inputs {
		layer.nodes = append(layer.nodes, &Node{activation: input})
	}

	return layer
}

func (nn *NeuralNet) BackwardPass(trainingInput []float64, trainingOutput []float64) {
	numLayers := len(nn.layers) - 1

	inDelta := trainingOutput
	for i := numLayers; i >= numLayers-1; i-- {
		layer := nn.layers[i]
		fmt.Println("running back: ", len(inDelta), len(layer.nodes))
		if i == numLayers {
			inDelta = layer.BackPropInit(inDelta, nn.learningRate, nn.layers[i-1])
		} else {
			inDelta = layer.Backprop(inDelta, nn.learningRate, nn.getNextBackpropLayer(i, trainingInput))
		}
	}
	fmt.Println("final backprop: ", len(inDelta))
}

// func (nn *NeuralNet) Epoch()

func RunNetwork() {

	// input := [][]float64{
	// 	{0.0, 0.0},
	// 	{1.0, 0.0},
	// 	{0.0, 1.0},
	// 	{1.0, 1.0},
	// }

	// output := [][]float64{
	// 	{0.0},
	// 	{1.0},
	// 	{1.0},
	// 	{0.0},
	// }

	nn := &NeuralNet{
		learningRate: 0.1,
		layers: []*Layer{
			// hidden layer
			{
				nodes: []*Node{
					NewNode(2),
					NewNode(2),
				},
			},
			// output layer
			{
				nodes: []*Node{
					NewNode(2),
				},
			},
		},
	}

	nn.ForwardPass([]float64{0, 0})
	nn.BackwardPass([]float64{0, 0}, []float64{0})
}
