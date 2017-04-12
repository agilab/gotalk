package gotalk

import (
	"log"
	"testing"
)

func TestSort(t *testing.T) {
	array := []float32{1.0, 3.0, 2.0, 1.5, 7.2, 1.1}
	log.Printf("%+v", topNSort(array, 3))
	log.Printf("%+v", topNSort(array, 4))
	log.Printf("%+v", topNSort(array, 1))
	log.Printf("%+v", topNSort(array, 7))
}
