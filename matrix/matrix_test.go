package matrix

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func initMatrix() *Matrix {
	m := NewMatrix(3, 3)

	var idx float64

	m.ForEach(func(val float64) float64 {
		idx += 1
		return idx
	})

	return m
}

func TestGetRow(t *testing.T) {

	m := initMatrix()

	row := m.GetRow(0)

	assert.Equal(t, float64(1), row[0])
	assert.Equal(t, float64(2), row[1])
	assert.Equal(t, float64(3), row[2])
}

func TestGetValue(t *testing.T) {
	m := initMatrix()
	val := m.GetValue(2, 2)
	assert.Equal(t, float64(9), val)
}

func TestGetColumn(t *testing.T) {
	m := initMatrix()
	col := m.GetColumn(0)

	assert.Equal(t, float64(1), col[0])
	assert.Equal(t, float64(4), col[1])
	assert.Equal(t, float64(7), col[2])
}

func TestAdd(t *testing.T) {
	m := initMatrix()
	m.Add(1)

	var idx int

	m.ForEach(func(val float64) float64 {
		idx += 1
		assert.Equal(t, float64(idx+1), val)
		return val
	})
}

func TestSub(t *testing.T) {
	m := initMatrix()
	m.Sub(1)

	var idx int

	m.ForEach(func(val float64) float64 {
		idx += 1
		assert.Equal(t, float64(idx-1), val)
		return val
	})
}

func TestMul(t *testing.T) {
	m := initMatrix()
	m.Mul(3)

	var idx int

	m.ForEach(func(val float64) float64 {
		idx += 1
		assert.Equal(t, float64(idx*3), val)
		return val
	})
}

func TestDiv(t *testing.T) {
	m := initMatrix()
	m.Div(2)

	var idx int

	m.ForEach(func(val float64) float64 {
		idx += 1
		assert.Equal(t, float64(idx)/2, val)
		return val
	})
}

func TestTranspose(t *testing.T) {
	m := initMatrix()
	mt := m.Transpose()

	row := mt.GetRow(0)
	assert.Equal(t, float64(1), row[0])
	assert.Equal(t, float64(4), row[1])
	assert.Equal(t, float64(7), row[2])
}

func TestVectorDot(t *testing.T) {
	v := Vector([]float64{1, 2, 3})
	y := []float64{4, 5, 6}

	z, err := v.Dot(y)
	assert.NoError(t, err)
	assert.Equal(t, float64(32), z)

	v = Vector([]float64{3, 4})
	y = []float64{5, 7}

	z, err = v.Dot(y)
	assert.NoError(t, err)
	assert.Equal(t, float64(43), z)

}

func TestMatrixDot(t *testing.T) {
	x := NewMatrix(2, 2)
	y := NewMatrix(2, 2)

	var n float64

	x.ForEach(func(val float64) float64 {
		n += 1
		return n
	})

	y.ForEach(func(val float64) float64 {
		n += 1
		return n
	})

	z := x.Dot(y)

	assert.Equal(t, float64(19), z.GetValue(0, 0))
	assert.Equal(t, float64(22), z.GetValue(0, 1))
	assert.Equal(t, float64(43), z.GetValue(1, 0))
	assert.Equal(t, float64(50), z.GetValue(1, 1))
}
