package main

import (
	"errors"
	"fmt"
	"log"
	"math"
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

func (layer *Layer) ForwardPass(inputs []float64) (layerOutput []float64, err error) {

	for _, node := range layer.nodes {

		// compute activation based on each node weight and input value
		activation := node.bias

		// make sure each node has the same amount of weights as the input array
		if len(inputs) != len(node.weights) {
			return layerOutput, errors.New("input length does not match node weights length")
		}

		// sum the activation by multiplying input by corresponding node weight
		for idx, input := range inputs {
			activation += input * node.weights[idx]
		}

		// set the layers activation
		node.activation = sigmoid(activation)

		// create the layer output as a convenience for the next layer
		layerOutput = append(layerOutput, node.activation)
	}

	return layerOutput, nil
}

func (layer *Layer) BackPropInit(inDelta []float64, learningRate float64, nextLayer *Layer) (outDelta []float64, err error) {

	if len(inDelta) != len(layer.nodes) {
		return outDelta, errors.New("inDelta length does not match layer nodes")
	}

	// loop through each node in the output layer
	for idx, node := range layer.nodes {

		e := inDelta[idx] - node.activation    // calculate error
		delta := e * dSigmoid(node.activation) // calculate delta
		outDelta = append(outDelta, delta)     // build input for next layer

		// apply change in output weights
		node.bias += delta * learningRate

		for nIdx, n := range nextLayer.nodes {
			// assumes that node weights are same length as next layers nodes ???
			node.weights[nIdx] += n.activation * delta * learningRate
		}
	}

	return outDelta, nil
}

func (layer *Layer) Backprop(inDelta []float64, learningRate float64, nextLayer, previousLayer *Layer) (outDelta []float64) {
	for _, node := range layer.nodes {

		// initialize error at 0
		var dErr float64

		// loop through previous layer nodes
		for idx, n := range previousLayer.nodes {
			// sum the error by multiplying the previous layer delta by the previous layer's node weights
			fmt.Println("check =>", len(inDelta), len(n.weights))
			// TODO this isn't even toughing every node weight!
			dErr += inDelta[idx] * n.weights[idx]
		}

		delta := dErr * dSigmoid(node.activation)
		outDelta = append(outDelta, delta)

		// apply weights
		node.bias += delta * learningRate

		for nIdx, n := range nextLayer.nodes {
			node.weights[nIdx] += n.activation * delta * learningRate
		}
	}

	return outDelta
}

type NeuralNet struct {
	layers       []*Layer
	learningRate float64
}

func (nn *NeuralNet) ForwardPass(inputs []float64) (err error) {

	// initialize layer input as the neural network input
	layerInput := inputs

	for _, layer := range nn.layers {
		// overwrite layer inputs to pass into the next iteration
		layerInput, err = layer.ForwardPass(layerInput)
		if err != nil {
			return err
		}
	}

	return nil
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

func (nn *NeuralNet) BackwardPass(trainingInput []float64, trainingOutput []float64) (err error) {
	numLayers := len(nn.layers) - 1

	inDelta := trainingOutput

	// loop backwards through layers
	for i := numLayers; i >= numLayers-1; i-- {
		// reference current layer
		layer := nn.layers[i]

		if i == numLayers {
			// pass in the expected output and the next layer
			inDelta, err = layer.BackPropInit(inDelta, nn.learningRate, nn.layers[i-1])
			if err != nil {
				return err
			}
		} else {
			// convert the training input to a layer
			inDelta = layer.Backprop(inDelta, nn.learningRate, nn.getNextBackpropLayer(i, trainingInput), nn.layers[i+1])
		}
	}

	return nil
}

func (nn *NeuralNet) Epoch(epochNumber int, trainingInput [][]float64, trainingOutput [][]float64) {

	log.Printf("running epoch %d...", epochNumber)
	trainingOrder := shuffledTrainingSetOrder(len(trainingInput))

	for i := 0; i < len(trainingOrder); i++ {
		input := trainingInput[i]
		output := trainingOutput[i]

		nn.ForwardPass(input)
		nn.BackwardPass(input, output)
	}
}

func (nn *NeuralNet) Train(n int, trainingInput [][]float64, trainingOutput [][]float64) {
	for i := 0; i < n; i++ {
		nn.Epoch(i, trainingInput, trainingOutput)
	}
}

func (nn *NeuralNet) OutputLayer() *Layer {
	return nn.layers[len(nn.layers)-1]
}

func (nn *NeuralNet) Output() (output []float64) {
	outputLayer := nn.OutputLayer()
	for _, n := range outputLayer.nodes {
		output = append(output, n.activation)
	}
	return output
}

func (nn *NeuralNet) Predict(input []float64) []float64 {
	nn.ForwardPass(input)
	nn.OutputLayer()
	return nn.Output()
}

func (nn *NeuralNet) PredictSingle(idx int, input [][]float64, output [][]float64) {
	prediction := nn.Predict(input[idx])
	log.Printf("input: %v expected %v actual %v %v", input[idx], output[idx], math.Round(prediction[0]), prediction)
}

func (nn *NeuralNet) Test(inputs [][]float64, outputs [][]float64) {
	nCorrect := 0
	nIncorrect := 0

	for idx, input := range inputs {
		prediction := nn.Predict(input)
		flooredPrediction := math.Round(prediction[0])
		correct := flooredPrediction == outputs[idx][0]

		if correct {
			nCorrect++
		} else {
			nIncorrect++
		}

		log.Printf("input: %v expected %v actual %v %v", input, outputs[idx], flooredPrediction, prediction)
	}

	log.Printf("accuracy: correct %d incorrect %d", nCorrect, nIncorrect)
}

func NewNeuralNet(learningRate float64, layers []int, inputShape int) *NeuralNet {
	var networklayers []*Layer

	for layerIdx, n := range layers {
		var nodes []*Node

		for i := 0; i < n; i++ {
			if layerIdx == 0 {
				nodes = append(nodes, NewNode(inputShape))
			} else {
				nodes = append(nodes, NewNode(len(networklayers[layerIdx-1].nodes)))
			}
		}

		networklayers = append(networklayers, &Layer{
			nodes: nodes,
		})
	}

	return &NeuralNet{
		learningRate: learningRate,
		layers:       networklayers,
	}
}

func RunNetwork() {

	input := [][]float64{
		{0.0, 0.0},
		{1.0, 0.0},
		{0.0, 1.0},
		{1.0, 1.0},
	}

	output := [][]float64{
		{0.0},
		{1.0},
		{1.0},
		{0.0},
	}

	// input := [][]float64{
	// 	{1, 1},
	// 	{1, 2},
	// 	{1, 3},
	// 	{1, 4},
	// }

	// output := [][]float64{
	// 	{2},
	// 	{3},
	// 	{4},
	// 	{5},
	// }

	layerConf := []int{2, 1}
	nn := NewNeuralNet(0.1, layerConf, len(input[0]))
	nn.Train(1, input, output)
	nn.Test(input, output)
}
