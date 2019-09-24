package main

import (
	"log"

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

func (l *linearRegression) Predict(value float32) (float32, error) {
	inputTensor, err := tensorflow.NewTensor([]float32{value})
	if err != nil {
		return 0, err
	}

	resultTensor, err := l.tfModel.Session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			l.tfModel.Graph.Operation("X").Output(0): inputTensor,
		},
		[]tensorflow.Output{
			l.tfModel.Graph.Operation("model").Output(0),
		},
		nil,
	)

	if err != nil {
		log.Printf("%s", err)
		return 0, err
	}

	return resultTensor[0].Value().([]float32)[0], nil
}

func NewLinearRegression(path string) *linearRegression {
	return &linearRegression{
		exportPath: path,
	}
}
