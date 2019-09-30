package main

import (
	"encoding/json"
	"log"
	"os"
	"path"
	"strings"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

type linearRegression struct {
	exportPath string
	tfModel    *tensorflow.SavedModel
	inputName  string
	outputName string
}

func (l *linearRegression) Load() error {
	// load model
	model, err := tensorflow.LoadSavedModel(
		l.exportPath, []string{tfTagServing}, nil,
	)
	if err != nil {
		return err
	}

	// load IO config
	ioFile, err := os.Open(path.Join(l.exportPath, "io_config.json"))
	if err != nil {
		return err
	}

	ioConfig := struct {
		Input  string `json:"input_name"`
		Output string `json:"output_name"`
	}{}
	err = json.NewDecoder(ioFile).Decode(&ioConfig)
	if err != nil {
		return err
	}

	// set io configuration
	l.inputName = strings.Split(ioConfig.Input, ":")[0]
	l.outputName = strings.Split(ioConfig.Output, ":")[0]

	// assign tensorflow model
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
			l.tfModel.Graph.Operation(l.inputName).Output(0): inputTensor,
		},
		[]tensorflow.Output{
			l.tfModel.Graph.Operation(l.outputName).Output(0),
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
