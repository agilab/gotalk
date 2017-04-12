package gotalk

import (
	"flag"
	"math"
	"strings"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	sentenceSize = flag.Int("sentence_size", 20, "最大句子长度")
)

type Captions struct {
	Results []CaptionResult `json:"results"`
}
type CaptionResult struct {
	Probability float32 `json:"probability"`
	Sentence    string  `json:"sentence"`
}

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
			if !tnb.beams[iBeam].isClosed {
				stateSeq = append(stateSeq, tnb.beams[iBeam].stateFeed)
				lenSentence := len(tnb.beams[iBeam].sentence)
				inputSeq = append(inputSeq, tnb.beams[iBeam].sentence[lenSentence-1])
				tnbOpen.Push(tnb.beams[iBeam])
			}

		}
		if tnbOpen.size == 0 {
			break
		}
		stateFeed, _ := tf.NewTensor(stateSeq)
		inputFeed, _ := tf.NewTensor(inputSeq)

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
		softmax := output[0].Value().([][]float32)
		lstmState := output[1].Value().([][]float32)

		newTnb := TopNBeams{}
		newTnb.Init(*beamSize)

		// 先添加已经关闭的 beam
		for iBatch := 0; iBatch < tnb.size; iBatch++ {
			if tnb.beams[iBatch].isClosed {
				newTnb.Push(tnb.beams[iBatch])
				continue
			}
		}

		for iBatch := 0; iBatch < tnbOpen.size; iBatch++ {
			sortedProb := topNSort(softmax[iBatch], *beamSize)

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
				if id == vocab.EndId || id == 3 {
					beam.isClosed = true
				}
				newTnb.Push(beam)
			}
		}
		tnb = newTnb
	}

	for iBatch := 0; iBatch < tnb.size; iBatch++ {
		result := CaptionResult{}
		result.Probability = float32(math.Exp(tnb.beams[iBatch].logProb))

		joinedSentence := ""
		sentence := tnb.beams[iBatch].sentence
		for iWord := 0; iWord < len(sentence); iWord++ {
			word := vocab.GetWord(sentence[iWord])
			id := sentence[iWord]
			if id != vocab.StartId && id != vocab.EndId && id != vocab.UnkId && word != "." {
				joinedSentence = joinedSentence + " " + word
			}
		}
		result.Sentence = strings.TrimSpace(joinedSentence)
		caps.Results = append(caps.Results, result)
	}
	return caps, nil
}
