package main

import (
	"flag"
	"io/ioutil"
	"log"
	"os"
	"runtime/pprof"
	"time"

	"github.com/agilab/gotalk"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	modelFile  = flag.String("model", "", "模型文件")
	imageFile  = flag.String("image", "", "图片文件")
	vocabFile  = flag.String("vocab", "", "词典文件")
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
)

func main() {
	flag.Parse()

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	vocab := gotalk.Vocabulary{}
	err := vocab.LoadFromFile(*vocabFile)
	if err != nil {
		log.Fatalf("无法载入词典文件：%s", err)
	}

	// 载入模型文件
	model, err := ioutil.ReadFile(*modelFile)
	if err != nil {
		log.Fatalf("模型文件读取错误：%s", err)
	}

	// 从模型文件构建 graph
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatalf("Graph import 错误：%s", err)
	}

	// 创建 tf session
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatalf("TF session 创建错误：", err)
	}
	defer session.Close()

	// 创建图像 tensor
	image, err := makeTensorFromImage(*imageFile)
	if err != nil {
		log.Fatalf("无法创建图像 tensor：", err)
	}

	start := time.Now()
	gotalk.GenerateCaption(session, graph, &vocab, image)
	elapsed := time.Since(start)
	log.Printf("花费时间 %s", elapsed)
}

func makeTensorFromImage(filename string) (*tf.Tensor, error) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	tensor, err := tf.NewTensor(string(bytes))
	if err != nil {
		return nil, err
	}
	return tensor, err
}
