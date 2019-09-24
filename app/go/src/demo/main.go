package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
)

var exportPath = os.Getenv("MODEL_EXPORT_PATH")

const tfTagServing = "serve"

func main() {
	// load model from export path

	model := NewLinearRegression(exportPath)
	if err := model.Load(); err != nil {
		panic(err)
	}

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		x := r.URL.Query().Get("X")

		value, err := strconv.ParseFloat(x, 64)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		prediction, err := model.Predict(float32(value))
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}

		fmt.Fprintf(w, "f(%s) = %f", x, prediction)
	})
	log.Fatal(http.ListenAndServe(":8080", nil))
}
