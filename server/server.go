package main

import (
	"encoding/json"
	"flag"
	"io/ioutil"
	"log"
	"net/http"

	"github.com/agilab/gotalk"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	modelFile = flag.String("model", "", "模型文件")
	vocabFile = flag.String("vocab", "", "词典文件")
	port      = flag.String("port", "", "服务端口")

	session *tf.Session
	graph   *tf.Graph
	vocab   *gotalk.Vocabulary
)

func main() {
	flag.Parse()

	// 载入词典
	vocab = &gotalk.Vocabulary{}
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
	graph = tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatalf("Graph import 错误：%s", err)
	}

	// 创建 tf session
	session, err = tf.NewSession(graph, nil)
	if err != nil {
		log.Fatalf("TF session 创建错误：", err)
	}
	defer session.Close()

	http.HandleFunc("/im2txt", process)

	log.Fatal(http.ListenAndServe(":"+*port, nil))

}

func process(w http.ResponseWriter, r *http.Request) {
	url := r.URL.Query()["url"]
	if len(url) != 1 {
		return
	}
	log.Printf("%s", url[0])

	// 得到 url 图像的字节串
	response, err := http.Get(url[0])
	if err != nil {
		return
	}
	image, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return
	}
	tensor, err := tf.NewTensor(string(image))
	if err != nil {
		return
	}

	// 生成标题
	captions, err := gotalk.GenerateCaption(session, graph, vocab, tensor)
	if err != nil {
		return
	}
	log.Printf("%+v", captions)

	json.NewEncoder(w).Encode(captions)
}
