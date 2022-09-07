package util

import (
	"go-neural-net/matrix"
	"math"
)

func TanhPrime(x float64) float64 {
	return 1 - math.Pow(math.Tanh(x), 2)
}

func MSE(x *matrix.Matrix, y *matrix.Matrix) (z *matrix.Matrix) {
	//TODO https://www.omnicalculator.com/statistics/mse
	return x
}

func MSEPrime(x *matrix.Matrix, y *matrix.Matrix) (z *matrix.Matrix) {
	//TODO https://www.omnicalculator.com/statistics/mse
	return x
}
