package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"gonum.org/v1/plot/plotter"
)

// https://towardsdatascience.com/simple-neural-network-implementation-in-c-663f51447547
// network has a single hidden layer with 2 nodes,
// and a single node for the output layer.
// this is minimal config for learning XOR

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dSigmoid(x float64) float64 {
	return x * (1 - x)
}

// TODO initialize all hidden weights
// https://gist.github.com/espiritusanti/b7485c68a06ef2c8c76d8c62c8c39d8f#file-main-cpp-L68
func initWeight() float64 {
	// return 1
	return rand.Float64()
}

const (
	NUM_INPUTS        = 2
	NUM_HIDDEN_NODES  = 2
	NUM_OUTPUTS       = 1
	NUM_TRAINING_SETS = 4
)

func shuffledTrainingSetOrder() []int {

	trainingSetOrder := []int{}

	for i := 0; i < NUM_TRAINING_SETS; i++ {
		trainingSetOrder = append(trainingSetOrder, i)
	}

	rand.Shuffle(len(trainingSetOrder), func(i, j int) {
		temp := trainingSetOrder[i]
		trainingSetOrder[i] = trainingSetOrder[j]
		trainingSetOrder[j] = temp
	})
	return trainingSetOrder
}

type Network struct {
	learningRate float64
	numEpochs    int
	// hidden layer
	hiddenLayer     [NUM_HIDDEN_NODES]float64
	hiddenWeights   [NUM_INPUTS][NUM_HIDDEN_NODES]float64
	hiddenLayerBias [NUM_HIDDEN_NODES]float64
	// output layer
	outputLayer     [NUM_OUTPUTS]float64
	outputWeights   [NUM_HIDDEN_NODES][NUM_OUTPUTS]float64
	outputLayerBias [NUM_OUTPUTS]float64

	trainingInput  [NUM_TRAINING_SETS][NUM_INPUTS]float64
	trainingOutput [NUM_TRAINING_SETS][NUM_OUTPUTS]float64
}

func NewNetwork(epochs int, learningRate float64) *Network {
	nn := new(Network)
	nn.learningRate = learningRate
	nn.numEpochs = epochs
	nn.trainingInput = [NUM_TRAINING_SETS][NUM_INPUTS]float64{
		{0.0, 0.0},
		{1.0, 0.0},
		{0.0, 1.0},
		{1.0, 1.0},
	}
	nn.trainingOutput = [NUM_TRAINING_SETS][NUM_OUTPUTS]float64{
		{0.0},
		{1.0},
		{1.0},
		{0.0},
	}

	nn.initWeights()

	return nn
}

// func (nn *Network) loopForward(forEachInput func(weight float64), forEachNode func() (float64)) {
// 	for i := 0; i < NUM_HIDDEN_NODES; i++ {
// 		for j := 0; j < NUM_INPUTS; j++ {

// 		}
// 	}
// }

func (nn *Network) initWeights() {
	for i := 0; i < NUM_INPUTS; i++ {
		for j := 0; j < NUM_HIDDEN_NODES; j++ {
			nn.hiddenWeights[i][j] = initWeight()
		}
	}

	for i := 0; i < NUM_HIDDEN_NODES; i++ {
		nn.hiddenLayerBias[i] = initWeight()
		for j := 0; j < NUM_OUTPUTS; j++ {
			nn.outputWeights[i][j] = initWeight()
		}
	}

	for i := 0; i < NUM_OUTPUTS; i++ {
		nn.outputLayerBias[i] = initWeight()
	}
}

func (nn *Network) forwardPass(inputs [NUM_INPUTS]float64) {
	// compute hidden layer activation
	for j := 0; j < NUM_HIDDEN_NODES; j++ {
		activation := nn.hiddenLayerBias[j]
		for k := 0; k < NUM_INPUTS; k++ {
			// activation is the sum of each input val multiplied by the hidden weight
			activation += inputs[k] * nn.hiddenWeights[k][j]
		}
		nn.hiddenLayer[j] = sigmoid(activation)
	}

	// compute output layer activation
	for j := 0; j < NUM_OUTPUTS; j++ {
		activation := nn.outputLayerBias[j]
		for k := 0; k < NUM_HIDDEN_NODES; k++ {
			activation += nn.hiddenLayer[k] * nn.outputWeights[k][j]
		}
		nn.outputLayer[j] = sigmoid(activation)
		// log.Printf("output_layer: %v", nn.outputLayer[j])
	}
}

func (nn *Network) backProp(inputs [NUM_INPUTS]float64, outputs [NUM_OUTPUTS]float64) (errs []float64) {
	// BACKPROP
	// compute change in output weights
	var deltaOutput [NUM_OUTPUTS]float64
	for j := 0; j < NUM_OUTPUTS; j++ {

		dErr := (outputs[j] - nn.outputLayer[j])
		errs = append(errs, dErr)
		// log.Printf("calc_err %f %f %f", outputs[j], nn.outputLayer[j], dErr)
		log.Printf("calc_err %f", dErr)
		deltaOutput[j] = dErr * dSigmoid(nn.outputLayer[j])
	}

	// compute change in hidden weights
	var deltaHidden [NUM_HIDDEN_NODES]float64
	for j := 0; j < NUM_HIDDEN_NODES; j++ {
		var dErr float64
		for k := 0; k < NUM_OUTPUTS; k++ {
			dErr += deltaOutput[k] * nn.outputWeights[j][k]
		}
		// log.Printf("hidden_weight_err: %f", dErr)
		deltaHidden[j] = dErr * dSigmoid(nn.hiddenLayer[j])

	}

	// apply change in output weights
	for j := 0; j < NUM_OUTPUTS; j++ {
		nn.outputLayerBias[j] += deltaOutput[j] * nn.learningRate
		for k := 0; k < NUM_HIDDEN_NODES; k++ {
			nn.outputWeights[k][j] += nn.hiddenLayer[k] * deltaOutput[j] * nn.learningRate
		}
	}

	// apply change in hidden weights
	for j := 0; j < NUM_HIDDEN_NODES; j++ {
		nn.hiddenLayerBias[j] += deltaHidden[j] * nn.learningRate
		for k := 0; k < NUM_INPUTS; k++ {
			nn.hiddenWeights[k][j] += inputs[k] * deltaHidden[j] * nn.learningRate
		}
	}

	return errs
}

func (nn *Network) epochs() {
	fmt.Println("epochs...", nn.numEpochs)

	// p := plot.New()
	// p.Title.Text = "Errors"

	var errors plotter.Values

	for i := 0; i < nn.numEpochs; i++ {
		errs := nn.epoch()
		errors = append(errors, errs...)
		log.Printf("completed epoch %d...", i)
		// log.Printf("log hidden weights %v", nn.hiddenLayer)
	}
	// hist, err := plotter.NewHist(errors, 10)
	// if err != nil {
	// 	panic(err)
	// }

	// p.Add(hist)

	// if err := p.Save(4*vg.Inch, 4*vg.Inch, "hist.png"); err != nil {
	// 	panic(err)
	// }
}

func (nn *Network) epoch() (outErrs []float64) {
	trainingSetOrder := shuffledTrainingSetOrder()

	for x := 0; x < NUM_TRAINING_SETS; x++ {
		i := trainingSetOrder[x]
		nn.forwardPass(nn.trainingInput[i])
		errs := nn.backProp(nn.trainingInput[i], nn.trainingOutput[i])
		outErrs = append(outErrs, errs...)
	}
	return outErrs
}

func (nn *Network) predict(input [NUM_INPUTS]float64) [NUM_OUTPUTS]float64 {
	// {0.0, 0.0}, -> 0.0
	// {1.0, 0.0}, -> 1.0
	// {0.0, 1.0}, -> 1.0
	// {1.0, 1.0}, -> 0.0

	nn.forwardPass(input)
	return nn.outputLayer
}

func main() {
	nn := NewNetwork(10000, 0.1)
	nn.initWeights()
	nn.epochs()
	outputLayer := nn.predict([NUM_INPUTS]float64{1, 1})
	log.Printf("predicted: %v", outputLayer)
}
