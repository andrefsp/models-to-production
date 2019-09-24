package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLoadAndPredict(t *testing.T) {
	model := NewLinearRegression(exportPath)
	assert.Nil(t, model.Load())
	assert.NotNil(t, model.tfModel)

	prediction, err := model.Predict(10)
	assert.Nil(t, err)
	assert.NotEqual(t, prediction, 0)

}
