package main

import (
	"log"
	"testing"
)

func TestInitWeight(t *testing.T) {

	nn := NewNetwork(10, 0.1)
	nn.initWeights()
	log.Printf("hidden weights: %v", nn.hiddenWeights)
	log.Printf("hidden layer bias: %v", nn.hiddenLayerBias)
	log.Printf("output weights: %v", nn.outputWeights)
	log.Printf("ouput bias: %v", nn.outputLayerBias)
}
