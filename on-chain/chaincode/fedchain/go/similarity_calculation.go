package main

import (
	"encoding/json"
	"strconv"
	"fmt"
	"math"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
	"sort"
)

type argsort struct {
	s    []float32 // Points to orignal array but does NOT alter it.
	inds []int     // Indexes to be returned.
}

func (a argsort) Len() int {
	return len(a.s)
}

func (a argsort) Less(i, j int) bool {
	return a.s[a.inds[i]] < a.s[a.inds[j]]
}

func (a argsort) Swap(i, j int) {
	a.inds[i], a.inds[j] = a.inds[j], a.inds[i]
}

func (s *SmartContract) UpdateGlobalSimilarityMatrix(ctx contractapi.TransactionContextInterface) ([][]float32, error){
	var sparseMatrix [][]float32
	globalSimilarityMatrix, err := s.GetGlobalSimilarityMatrix(ctx)

	for i := 0; i < clientNum; i++{
		localModelMetaInfo, _ := s.GetLocalModelMetaInfo(ctx, "org"+strconv.Itoa(i))
		sparseMatrix = append(sparseMatrix, localModelMetaInfo.SparseVector)
	}

	for i := 0; i < clientNum; i++{
		for j := i+1; j < clientNum; j++{
			similarity := CalculateSimilarity(sparseMatrix[i], sparseMatrix[j])
			globalSimilarityMatrix.SimilarityMatrix[i][j] = similarity
			globalSimilarityMatrix.SimilarityMatrix[j][i] = similarity 
		}
	}

	globalSimilarityMatrixBytes, _ := json.Marshal(globalSimilarityMatrix)
	err = ctx.GetStub().PutState("similarityMatrix", globalSimilarityMatrixBytes)

	return globalSimilarityMatrix.SimilarityMatrix, err
}

func CalculateSimilarity(v1 []float32, v2 []float32) float32 {
	//similarity
	// if len(v1) == 0 || len(v2) == 0{
	// 	return 0.0
	// }

	// result := 0.0
	// for i := 0; i<len(v1); i++{
	// 	result += math.Pow(float64(v1[i] - v2[i]), 2)
	// }

	// result = math.Sqrt(result)

	// if result == 0 {
	// 	result = 1.0
	// } else{
	// 	result = 1.0 / result
	// }

	//distance
	if len(v1) == 0 || len(v2) == 0{
		return 1000
	}

	result := 0.0
	for i := 0; i<len(v1); i++{
		result += math.Pow(float64(v1[i] - v2[i]), 2)
	}

	result = math.Sqrt(result)
	return float32(result)
}