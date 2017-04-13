package gotalk

import (
	"flag"
	"math"
	"strings"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	sentenceSize = flag.Int("sentence_size", 20, "最大句子长度")
	beamSize     = flag.Int("beam_size", 3, "beam search 搜索宽度")
)

// 返回带 probability 的标题
type Captions struct {
	Results []CaptionResult `json:"results"`
}
type CaptionResult struct {
	Probability float32 `json:"probability"`
	Sentence    string  `json:"sentence"`
}

// 可以多线程调用
func GenerateCaption(session *tf.Session, graph *tf.Graph, vocab *Vocabulary, image *tf.Tensor) (Captions, error) {
	caps := Captions{}

	// 从图像得到 LSTM 初始状态
	initialState, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("image_feed").Output(0): image,
		},
		[]tf.Output{
			graph.Operation("lstm/initial_state").Output(0),
		},
		nil)
	if err != nil {
		return caps, err
	}

	// 初始化第一个 beam
	beam := Beam{
		logProb:   float64(0.0),
		sentence:  []int64{vocab.StartId},
		stateFeed: initialState[0].Value().([][]float32)[0],
	}
	tnb := TopNBeams{}
	tnb.Init(*beamSize)
	tnb.Push(beam)

	for iSentence := 0; iSentence < *sentenceSize; iSentence++ {
		// 从 beams 构造 LSTM 输入
		stateSeq := [][]float32{}
		inputSeq := []int64{}
		tnbOpen := TopNBeams{}
		tnbOpen.Init(*beamSize)
		for iBeam := 0; iBeam < tnb.size; iBeam++ {
			// 只添加尚未完成搜索的 beam，为了降低搜索空间大小
			if !tnb.beams[iBeam].isClosed {
				stateSeq = append(stateSeq, tnb.beams[iBeam].stateFeed)
				lenSentence := len(tnb.beams[iBeam].sentence)
				inputSeq = append(inputSeq, tnb.beams[iBeam].sentence[lenSentence-1])
				tnbOpen.Push(tnb.beams[iBeam])
			}

		}

		// 如果所有 beam 都已经完成，结束
		if tnbOpen.size == 0 {
			break
		}

		// 创建 TF graph 输入 tensor
		stateFeed, _ := tf.NewTensor(stateSeq)
		inputFeed, _ := tf.NewTensor(inputSeq)

		// 执行 LSTM 单元运算，注意这里的 batch size = tnbOpen.size
		output, err := session.Run(
			map[tf.Output]*tf.Tensor{
				graph.Operation("input_feed").Output(0):      inputFeed,
				graph.Operation("lstm/state_feed").Output(0): stateFeed,
			},
			[]tf.Output{
				graph.Operation("softmax").Output(0),
				graph.Operation("lstm/state").Output(0),
			},
			nil)
		if err != nil {
			return caps, err
		}
		softmax := output[0].Value().([][]float32)   // softmax 保存的是词 one-hot encoding 的概率值
		lstmState := output[1].Value().([][]float32) // 下个 LSTM 计算的状态输入

		// newTnb 中将添加下一轮 LSTM 计算的所有 beam
		newTnb := TopNBeams{}
		newTnb.Init(*beamSize)

		// 先添加已经关闭的 beam
		for iBatch := 0; iBatch < tnb.size; iBatch++ {
			if tnb.beams[iBatch].isClosed {
				newTnb.Push(tnb.beams[iBatch])
				continue
			}
		}

		// 然后添加所有新得到的 beam
		for iBatch := 0; iBatch < tnbOpen.size; iBatch++ {
			// 得到 top n 的概率
			sortedProb := topNSort(softmax[iBatch], *beamSize)

			// 添加新 beam
			for iWord := 0; iWord < len(sortedProb) && iWord < *beamSize; iWord++ {
				id := int64(sortedProb[iWord])
				value := float64(softmax[iBatch][id])

				se := make([]int64, len(tnbOpen.beams[iBatch].sentence))
				copy(se, tnbOpen.beams[iBatch].sentence)
				se = append(se, id)
				beam := Beam{
					logProb:   tnbOpen.beams[iBatch].logProb + math.Log(value),
					sentence:  se,
					stateFeed: lstmState[iBatch],
				}

				// 检查该 bean 是否已经结束
				if id == vocab.EndId || id == 3 {
					beam.isClosed = true
				}
				newTnb.Push(beam)
			}
		}
		tnb = newTnb
	}

	// 从最终结果生成返回的 Json 结构体
	for iBatch := 0; iBatch < tnb.size; iBatch++ {
		result := CaptionResult{}
		result.Probability = float32(math.Exp(tnb.beams[iBatch].logProb))

		joinedSentence := ""
		sentence := tnb.beams[iBatch].sentence
		for iWord := 0; iWord < len(sentence); iWord++ {
			word := vocab.GetWord(sentence[iWord])
			id := sentence[iWord]

			// 去除特殊字符
			if id != vocab.StartId && id != vocab.EndId && id != vocab.UnkId && word != "." {
				joinedSentence = joinedSentence + " " + word
			}
		}
		result.Sentence = strings.TrimSpace(joinedSentence)
		caps.Results = append(caps.Results, result)
	}
	return caps, nil
}
