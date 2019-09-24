package main

import (
	"errors"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

type linearRegression struct {
	exportPath string
	tfModel    *tensorflow.SavedModel
}

func (l *linearRegression) Load() error {
	model, err := tensorflow.LoadSavedModel(
		l.exportPath, []string{tfTagServing}, nil,
	)
	if err != nil {
		return err
	}

	l.tfModel = model

	return nil
}

func (l *linearRegression) Predict(value int32) (int32, error) {
	return 0, errors.New("does not predict")
}

func NewLinearRegression(path string) *linearRegression {
	return &linearRegression{
		exportPath: path,
	}
}
