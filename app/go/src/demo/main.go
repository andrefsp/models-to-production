package main

import (
	"fmt"
	"os"
)

var exportPath = os.Getenv("MODEL_EXPORT_PATH")

const tfTagServing = "serve"

func main() {
	fmt.Println("bla!")
}
