package matrix

import (
	"fmt"
	"math/rand"
	"time"
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

func (m *Matrix) Copy() *Matrix {
	z := new(Matrix)
	*z = *m
	return z
}

func (m *Matrix) GetColumn(column int) (col []float64) {
	for i := 0; i < m.Rows; i++ {
		r := m.GetRow(i)
		col = append(col, r[column])
	}

	return col
}

func (m *Matrix) Mul(x float64) *Matrix {
	z := m.Copy()

	z.ForEach(func(val float64) float64 {
		return val * x
	})

	return z
}

func (m *Matrix) Div(x float64) *Matrix {
	z := m.Copy()

	z.ForEach(func(val float64) float64 {
		return val / x
	})
	return z
}

func (m *Matrix) Add(x float64) *Matrix {
	z := m.Copy()

	z.ForEach(func(val float64) float64 {
		return val + x
	})
	return z
}

func (m *Matrix) Sub(x float64) *Matrix {
	z := m.Copy()

	z.ForEach(func(val float64) float64 {
		return val - x
	})

	return z
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

func (m *Matrix) ForEachRow(cb func(idx int, row []float64)) {
	for i := 0; i < m.Rows; i++ {
		cb(i, m.GetRow(i))
	}
}

func (m *Matrix) Dot(x *Matrix) *Matrix {
	var values []float64

	m.ForEachRow(func(idx int, row []float64) {
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

func (m *Matrix) InitRandom() {
	rand.Seed(time.Now().UnixNano())

	m.ForEach(func(val float64) float64 {
		return rand.Float64()
	})
}

func (m *Matrix) AddMatrix(x *Matrix) (*Matrix, error) {

	if x.Rows == 1 {
		return m.AddVector(x.values), nil
	}

	if err := m.MatchDim(x); err != nil {
		return nil, err
	}

	z := NewMatrix(m.Rows, m.Columns)

	m.ForEachRow(func(idx int, xRow []float64) {
		yRow := x.GetRow(idx)
		zRow, _ := Vector(xRow).Add(yRow)
		z.values = append(z.values, zRow...)
	})

	return z, nil
}

func (m *Matrix) AddVector(x Vector) *Matrix {
	z := NewMatrix(m.Rows, m.Columns)

	m.ForEachRow(func(idx int, row []float64) {
		zRow, _ := Vector(row).Add(x)
		z.values = append(z.values, zRow...)
	})

	return z
}

func (m *Matrix) SubMatrix(x *Matrix) (*Matrix, error) {
	if err := m.MatchDim(x); err != nil {
		return nil, err
	}

	z := NewMatrix(m.Rows, m.Columns)

	m.ForEachRow(func(idx int, xRow []float64) {
		yRow := x.GetRow(idx)
		zRow, _ := Vector(xRow).Sub(yRow)
		z.values = append(z.values, zRow...)
	})

	return z, nil
}

func (m *Matrix) MatchDim(x *Matrix) error {
	if m.Columns != x.Columns || m.Rows != x.Rows {
		return fmt.Errorf("shape (%d, %d) does not match (%d, %d)", m.Columns, m.Rows, x.Columns, x.Rows)
	}
	return nil
}
