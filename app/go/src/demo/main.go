package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
)

var exportPath = os.Getenv("EXPORT_PATH")

const tfTagServing = "serve"

func main() {
	// load model from export path

	model := NewLinearRegression(exportPath)
	if err := model.Load(); err != nil {
		panic(err)
	}

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// get input argument
		x := r.URL.Query().Get("X")
		if x == "" {
			x = r.URL.Query().Get("x")
		}

		log.Printf("Resolving model(x=%s) \n", x)

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

	log.Printf("Serving model on port :8080")

	log.Fatal(http.ListenAndServe(":8080", nil))
}
