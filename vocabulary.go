package gotalk

import (
	"bufio"
	"os"
	"strings"
)

const (
	startWord = "<S>"
	endWord   = "</S>"
	unkWord   = "<UNK>"
)

type Vocabulary struct {
	idToWord map[int64]string
	wordToId map[string]int64

	StartId int64
	EndId   int64
	UnkId   int64
}

func (v *Vocabulary) LoadFromFile(filename string) error {
	v.idToWord = make(map[int64]string)
	v.wordToId = make(map[string]int64)
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	id := int64(0)
	for scanner.Scan() {
		fields := strings.Split(scanner.Text(), " ")
		word := fields[0]
		v.idToWord[id] = word
		v.wordToId[word] = id
		id++
	}
	if err := scanner.Err(); err != nil {
		return err
	}

	if _, found := v.wordToId[unkWord]; !found {
		v.wordToId[unkWord] = id
		v.idToWord[id] = unkWord
	}

	v.StartId = v.wordToId[startWord]
	v.EndId = v.wordToId[endWord]
	v.UnkId = v.wordToId[unkWord]

	return nil
}

func (v *Vocabulary) GetId(word string) int64 {
	if id, found := v.wordToId[word]; found {
		return id
	}
	return v.UnkId
}

func (v *Vocabulary) GetWord(id int64) string {
	if word, found := v.idToWord[id]; found {
		return word
	}
	return unkWord
}
