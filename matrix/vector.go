package matrix

import "fmt"

type Vector []float64

func (x Vector) Dot(y Vector) (z float64, err error) {
	err = x.ForEach(y, func(xVal, yVal float64) {
		z += xVal * yVal
	})
	return z, err
}

func (x Vector) Add(y Vector) (z []float64, err error) {
	err = x.ForEach(y, func(xVal, yVal float64) {
		z = append(z, xVal+yVal)
	})
	return z, err
}

func (x Vector) Sub(y Vector) (z []float64, err error) {
	err = x.ForEach(y, func(xVal, yVal float64) {
		z = append(z, xVal-yVal)
	})
	return z, err
}

func (x Vector) ForEach(y Vector, cb func(xVal, yVal float64)) (err error) {
	if err := x.MatchLen(y); err != nil {
		return err
	}

	for idx, val := range x {
		cb(val, y[idx])
	}

	return nil
}

func (x Vector) MatchLen(y Vector) error {
	if len(x) != len(y) {
		return fmt.Errorf("vectors %d %d not the same length", len(x), len(y))
	}

	return nil
}
