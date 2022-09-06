package util

import "math"

func TanhPrime(x float64) float64 {
	return 1 - math.Pow(math.Tanh(x), 2)
}
