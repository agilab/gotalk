package gotalk

// 得到 array 中 top n 的元素的 index
func topNSort(array []float32, n int) []int {
	result := make([]int, 0, n)
	for i, number := range array {
		j := 0
		for ; j < len(result) && number <= array[result[j]]; j++ {
		}

		if j == len(result) {
			if j < n {
				result = append(result, i)
			}
			continue
		}

		if len(result) == n {
			for k := n - 1; k > j; k-- {
				result[k] = result[k-1]
			}
		} else {
			result = append(result, 0)
			for k := len(result) - 1; k > j; k-- {
				result[k] = result[k-1]
			}
		}
		result[j] = i
	}

	return result
}
