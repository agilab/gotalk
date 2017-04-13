package gotalk

import (
	"flag"
)

type Beam struct {
	sentence  []int64
	logProb   float64
	stateFeed []float32
	isClosed  bool
}

type TopNBeams struct {
	beams []Beam
	size  int
}

// 使用前必须初始化
func (b *TopNBeams) Init(maxBeamSize int) {
	b.beams = make([]Beam, maxBeamSize)
	b.size = 0
}

// 如果 logProb 大于其中任何一个元素，按照顺序将 beam 添加到 TopNBeams 中
func (b *TopNBeams) Push(beam Beam) {
	// 找到可以插入的点
	iInsert := 0
	for ; iInsert < b.size && beam.logProb <= b.beams[iInsert].logProb; iInsert++ {
	}

	// 无处插入
	if iInsert == b.maxBeamSize {
		return
	}

	// 添加到最后
	if iInsert == b.size {
		b.beams[b.size] = beam
		b.size++
		return
	}

	// 添加到中间，并去掉最后一个
	if b.size == b.maxBeamSize {
		for j := b.size - 1; j > iInsert; j-- {
			b.beams[j] = b.beams[j-1]
		}
		b.beams[iInsert] = beam
		return
	}

	// 添加到中间
	for j := b.size; j > iInsert; j-- {
		b.beams[j] = b.beams[j-1]
	}
	b.beams[iInsert] = beam
	b.size++
}
