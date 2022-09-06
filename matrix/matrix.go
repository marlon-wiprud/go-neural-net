package matrix

import (
	"fmt"
)

type Matrix struct {
	values  []float64
	Rows    int
	Columns int
}

func NewMatrix(rows, columns int) *Matrix {
	return &Matrix{
		values:  []float64{},
		Rows:    rows,
		Columns: columns,
	}
}

func (m *Matrix) ForEach(cb func(val float64) float64) {
	if len(m.values) > 0 {
		for i, v := range m.values {
			m.values[i] = cb(v)
		}
	} else {
		for i := 0; i < m.Rows*m.Columns; i++ {
			m.values = append(m.values, cb(0))
		}
	}

}

func (m *Matrix) GetRow(row int) []float64 {
	x := row * m.Columns       // x coordinate
	y := (row + 1) * m.Columns // y coordinate
	return m.values[x:y]
}

func (m *Matrix) GetValue(row, column int) float64 {
	r := m.GetRow(row)
	return r[column]
}

func (m *Matrix) GetColumn(column int) (col []float64) {
	for i := 0; i < m.Rows; i++ {
		r := m.GetRow(i)
		col = append(col, r[column])
	}

	return col
}

func (m *Matrix) Mul(x float64) {
	m.ForEach(func(val float64) float64 {
		return val * x
	})
}

func (m *Matrix) Div(x float64) {
	m.ForEach(func(val float64) float64 {
		return val / x
	})
}

func (m *Matrix) Add(x float64) {
	m.ForEach(func(val float64) float64 {
		return val + x
	})
}

func (m *Matrix) Sub(x float64) {
	m.ForEach(func(val float64) float64 {
		return val - x
	})
}

func (m *Matrix) Transpose() *Matrix {
	var values []float64

	for i := 0; i < m.Columns; i++ {
		col := m.GetColumn(i)
		values = append(values, col...)
	}

	return &Matrix{
		Rows:    m.Columns,
		Columns: m.Rows,
		values:  values,
	}
}

func (m *Matrix) ForEachColumn(cb func([]float64)) {
	for i := 0; i < m.Columns; i++ {
		col := m.GetColumn(i)
		cb(col)
	}
}

func (m *Matrix) ForEachRow(cb func([]float64)) {
	for i := 0; i < m.Rows; i++ {
		cb(m.GetRow(i))
	}
}

func (m *Matrix) Dot(x *Matrix) *Matrix {
	var values []float64

	m.ForEachRow(func(row []float64) {
		x.ForEachColumn(func(col []float64) {
			d, _ := Vector(row).Dot(col)
			values = append(values, d)
		})
	})

	return &Matrix{
		Rows:    m.Rows,
		Columns: x.Columns,
		values:  values,
	}
}

type Vector []float64

func (x Vector) Dot(y Vector) (z float64, err error) {
	if len(x) != len(y) {
		return z, fmt.Errorf("vectors %d %d not the same length", len(x), len(y))
	}

	for idx, val := range x {
		z += val * y[idx]
	}

	return z, nil
}
