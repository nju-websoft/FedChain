// import(
//     "log"
//     "encoding/json"
//     "encoding/pem"
//     "crypto/x509"
//     "fmt"
//     "bytes"
//     "math"
//     "sort"
//     "strings"
//     "github.com/hyperledger/fabric-contract-api-go/contractapi"
// )

package main

import (
	"encoding/json"
	"strconv"
	"fmt"
	"math"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
	"time"
	"sort"
)

var clientNum int
var clusterNum int

// SmartContract provides functions for managing a car
type SmartContract struct {
	contractapi.Contract
}

// model detail information
// k: curHashId v:ModelBlock
type ModelBlock struct {
	ModelType string `json:"modelType"` 
	PrevHashId string `json:"prevHashId"` 
	ModelUrl string `json:"modelUrl"`
	Timestamp string `json:"timestamp"`
	Organization string `json:"organization"`
}

// global model meta information 
// k: 'global' v: GlobalMetaInfo
type GlobalModelMetaInfo struct {
	CurHashId string `json:"curHashId"`
	Round int `json:"round"`
	UploadCount int `json:"uploadCount"`
	TriggerAvgNum int `json:"triggerAvgNum"`

	LocalUpdates map[string]ModelBlock `json:"localUpdates"`
	OrgIds []int `json:"orgIds"`
	Clusters [][]int `json:"clusters"`
	ClusterModelUrls map[string]string `json:"clusterModelUrls"`
}

type ClusterModelMetaInfo struct {
	Cluster string `json:"cluster"`
	Round int `json:"round"`
	ModelUrl string `json:"modelUrl"`
}

// meta infomation of local model in each organization 
// k: 'orgX' v: LocalModelMetaInfo
type LocalModelMetaInfo struct {
	CurHashId string `json:"curHashId"`
	Round int `json:"round"`
	LocalModelBlock ModelBlock `json:"localModelBlock"`
	
	SparseVector []float32 `json:"sparseVector"`
	//SimilarityVector []float32 `json:"similarityVector"`
}

// // meta infomation of local model in each cluster
// // k: 'clusterX' v: ClusterModelMetaInfo
// type ClusterModelMetaInfo struct {
// 	CurHashId string `json:"curHashId"`
// }

//##################################################################

type GlobalSimilarityMatrix struct{
	SimilarityMatrix [][]float32 `json:"similarityMatrix"`
}

type CollaborativeInfo struct{
	CollaborativeClients []int `json:"collaborativeClients"`
	ModelUrls []string `json:"modelUrls"`
}

//#############################Init######################################

func (s *SmartContract) InitLedger(ctx contractapi.TransactionContextInterface) error {

	clientNum = 10
	clusterNum = 1

	var initCluster [][]int
	var clientIds []int
	for i := 0; i < clientNum; i++{
		clientIds = append(clientIds, i)
	}
	initCluster = append(initCluster, clientIds)

	var globalModelMetaInfo GlobalModelMetaInfo = GlobalModelMetaInfo{CurHashId:"0", Round:0, UploadCount:0, TriggerAvgNum: 1, LocalUpdates:make(map[string]ModelBlock), OrgIds:[]int{}, Clusters:initCluster, ClusterModelUrls:make(map[string]string)}
	globalModelMetaInfoBytes, err := json.Marshal(globalModelMetaInfo)
	err = ctx.GetStub().PutState("global",globalModelMetaInfoBytes)

	timeStr := time.Now().Format("2022-10-01 15:49:05")
	var modelBlock ModelBlock = ModelBlock{ModelType:"global", PrevHashId:"", ModelUrl:"./models/server/0.pth", Timestamp:timeStr, Organization:"Public"}
	modelBlockBytes, err := json.Marshal(modelBlock)
	err = ctx.GetStub().PutState("0", modelBlockBytes)

	var globalSimilarityMatrix GlobalSimilarityMatrix = GlobalSimilarityMatrix{SimilarityMatrix: [][]float32{}}
	
	for i := 0; i < clientNum; i++{
		temp := make([]float32, clientNum)
		for j := range temp{
			temp[j] = 1000
		}
		temp[i] = 0

		globalSimilarityMatrix.SimilarityMatrix = append(globalSimilarityMatrix.SimilarityMatrix, temp)

		var localModelMetaInfo LocalModelMetaInfo = LocalModelMetaInfo{CurHashId:"0", Round:0, LocalModelBlock:modelBlock, SparseVector:[]float32{}}
		localModelMetaInfoBytes, _ := json.Marshal(localModelMetaInfo)
		err = ctx.GetStub().PutState("org"+strconv.Itoa(i), localModelMetaInfoBytes)
	}

	for i := 0; i < clusterNum; i++{
		var clusterModelMetaInfo ClusterModelMetaInfo = ClusterModelMetaInfo{Cluster:"cluster"+strconv.Itoa(i), Round:0, ModelUrl:"./models/server/cluster"+strconv.Itoa(i)+".pth"}
		clusterModelMetaInfoBytes, _ := json.Marshal(clusterModelMetaInfo)
		err = ctx.GetStub().PutState("cluster"+strconv.Itoa(i), clusterModelMetaInfoBytes)
	}

	globalSimilarityMatrixBytes, _ := json.Marshal(globalSimilarityMatrix)
	err = ctx.GetStub().PutState("similarityMatrix", globalSimilarityMatrixBytes)

	var accessControlTable AccessControlTable = AccessControlTable{AccessTable:make(map[string]SubjectAttribute), ExpiredTime:30, Policy:""}
	accessControlTableBytes, _ := json.Marshal(accessControlTable)
	err = ctx.GetStub().PutState("000", accessControlTableBytes)
	err = ctx.GetStub().PutState("001", accessControlTableBytes)
	err = ctx.GetStub().PutState("002", accessControlTableBytes)
	return err
}

//#########################################Read##################


func (s *SmartContract) GetGlobalModelMetaInfo(ctx contractapi.TransactionContextInterface) (*GlobalModelMetaInfo,error) {
	globalModelMetaInfoBytes, err := ctx.GetStub().GetState("global")

	var globalModelMetaInfo GlobalModelMetaInfo
	_ = json.Unmarshal(globalModelMetaInfoBytes, &globalModelMetaInfo)
	return &globalModelMetaInfo, err
}

func (s *SmartContract) GetClusterModelMetaInfo(ctx contractapi.TransactionContextInterface, cluster string) (*ClusterModelMetaInfo,error) {
	clusterModelMetaInfoBytes, err := ctx.GetStub().GetState(cluster)

	var clusterModelMetaInfo ClusterModelMetaInfo
	_ = json.Unmarshal(clusterModelMetaInfoBytes, &clusterModelMetaInfo)
	return &clusterModelMetaInfo, err
}

func (s *SmartContract) GetLocalModelMetaInfo(ctx contractapi.TransactionContextInterface,org string) (*LocalModelMetaInfo, error) {
	localModelMetaInfoBytes, err := ctx.GetStub().GetState(org)

	var localModelMetaInfo LocalModelMetaInfo
	_ = json.Unmarshal(localModelMetaInfoBytes, &localModelMetaInfo)
	return &localModelMetaInfo,err
}

func (s *SmartContract) GetModelBlock(ctx contractapi.TransactionContextInterface,curHashId string) (*ModelBlock, error) {
	modelBlockBytes, err := ctx.GetStub().GetState(curHashId)

	var modelBlock ModelBlock
	_ = json.Unmarshal(modelBlockBytes, &modelBlock)
	return &modelBlock,err
}

func (s *SmartContract) GetGlobalSimilarityMatrix(ctx contractapi.TransactionContextInterface) (*GlobalSimilarityMatrix, error) {
	globalSimilarityMatrixBytes, err := ctx.GetStub().GetState("similarityMatrix")

	var globalSimilarityMatrix GlobalSimilarityMatrix
	_ = json.Unmarshal(globalSimilarityMatrixBytes, &globalSimilarityMatrix)
	return &globalSimilarityMatrix, err
}

func (s *SmartContract) GetCollaborativeClients(ctx contractapi.TransactionContextInterface, org int, lossList []float32, threshold float32) (*CollaborativeInfo, error){
	collaborativeInfo := CollaborativeInfo{CollaborativeClients: []int{}, ModelUrls: []string{}}

	globalSimilarityMatrix, err := s.GetGlobalSimilarityMatrix(ctx)
	similarityVector := globalSimilarityMatrix.SimilarityMatrix[org]

	//get maxindex
	// maxIndex := org
	// var maxVal float32 = -1
	// for i := 0; i<clientNum; i++{
	// 	if i != org && maxVal < similarityVector[i] {
	// 		maxVal = similarityVector[i]
	// 		maxIndex = i
	// 	}
	// } 
	// collaborativeInfo.CollaborativeClients = append(collaborativeInfo.CollaborativeClients, maxIndex)

	// localModelMetaInfo, _ := s.GetLocalModelMetaInfo(ctx, "org"+strconv.Itoa(maxIndex))
	// collaborativeInfo.ModelUrls = append(collaborativeInfo.ModelUrls, localModelMetaInfo.LocalModelBlock.ModelUrl)

	//inverse knapsack-problem
	var ratios []float32
	for i := 0; i<clientNum; i++{
		ratios = append(ratios, similarityVector[i] / lossList[i])
	}
    indices := Argsort(ratios)
	
	maxNum := 2
	for i := 0; i<clientNum; i++{
		if indices[i] != org && threshold > 0 && len(collaborativeInfo.CollaborativeClients) < maxNum{
			collaborativeInfo.CollaborativeClients = append(collaborativeInfo.CollaborativeClients, indices[i])
			localModelMetaInfo, _ := s.GetLocalModelMetaInfo(ctx, "org"+strconv.Itoa(indices[i]))
			collaborativeInfo.ModelUrls = append(collaborativeInfo.ModelUrls, localModelMetaInfo.LocalModelBlock.ModelUrl)
			threshold -= lossList[indices[i]]
		}
	}

	return &collaborativeInfo, err
}

//#########################################Write##################

func (s *SmartContract) UpdateLocalModel(ctx contractapi.TransactionContextInterface, org string, curHashId string, modelUrl string, round int, timestamp string, sparseVector []float32) error{
	localModelMetaInfo, err := s.GetLocalModelMetaInfo(ctx, org)

	var modelBlock ModelBlock = ModelBlock{ModelType:"local", PrevHashId:localModelMetaInfo.CurHashId, ModelUrl:modelUrl, Timestamp:timestamp, Organization:org}

	localModelMetaInfo.CurHashId = curHashId
	localModelMetaInfo.Round = round
	localModelMetaInfo.LocalModelBlock = modelBlock
	localModelMetaInfo.SparseVector = sparseVector

	localModelMetaInfoBytes, err := json.Marshal(localModelMetaInfo)
	err = ctx.GetStub().PutState(org, localModelMetaInfoBytes)
	return err
}

func (s *SmartContract) UpdateClusterModel(ctx contractapi.TransactionContextInterface, cluster string, modelUrl string, round int) error{
	clusterModelMetaInfo, err := s.GetClusterModelMetaInfo(ctx, cluster)
	clusterModelMetaInfo.Round = round
	clusterModelMetaInfo.ModelUrl = modelUrl

	clusterModelMetaInfoBytes, err := json.Marshal(clusterModelMetaInfo)
	err = ctx.GetStub().PutState(cluster, clusterModelMetaInfoBytes)
	return err
}

func (s *SmartContract) ScanLocalUpdates(ctx contractapi.TransactionContextInterface) (map[string]LocalModelMetaInfo, error){
	globalModelMetaInfo, err := s.GetGlobalModelMetaInfo(ctx)
	localUpdates := make(map[string]LocalModelMetaInfo)
	
	for i := 0; i < clientNum; i++{
		org := "org"+strconv.Itoa(i)
		localModelMetaInfo, _ := s.GetLocalModelMetaInfo(ctx, org)
		//if localModelMetaInfo.Round !=0 && localModelMetaInfo.Round >= globalModelMetaInfo.Round{//factchain
		if localModelMetaInfo.Round !=0 && localModelMetaInfo.Round > globalModelMetaInfo.Round{//fedavg
			localUpdates[org] = *localModelMetaInfo
		} 
	}

	return localUpdates, err
	// scanInfo := struct{
	// 	globalInfo GlobalModelMetaInfo
	// 	localUpdates map[string]ModelBlock
	// }{
	// 	globalInfo: globalModelMetaInfo,
	// 	localUpdates: localUpdates,
	// }

	// scanInfoBytes, _ := json.Marshal(scanInfo)
	// return scanInfoBytes, err
}

func (s *SmartContract) ScanClusterUpdates(ctx contractapi.TransactionContextInterface) (map[string]ClusterModelMetaInfo, error){
	globalModelMetaInfo, err := s.GetGlobalModelMetaInfo(ctx)
	clusterUpdates := make(map[string]ClusterModelMetaInfo)
	
	for i := 0; i < clusterNum; i++{
		cluster := "cluster"+strconv.Itoa(i)
		clusterModelMetaInfo, _ := s.GetClusterModelMetaInfo(ctx, cluster)
		if clusterModelMetaInfo.Round > globalModelMetaInfo.Round{
			clusterUpdates[cluster] = *clusterModelMetaInfo
		} 
	}

	return clusterUpdates, err
}

func (s *SmartContract) UpdateGlobalModel(ctx contractapi.TransactionContextInterface, curHashId string, modelUrl string, clusters [][]int, timestamp string) error{
	globalModelMetaInfo, err := s.GetGlobalModelMetaInfo(ctx)

	// var modelBlock ModelBlock = ModelBlock{ModelType:"global", PrevHashId:globalModelMetaInfo.CurHashId, ModelUrl:modelUrl, Timestamp:timestamp, Organization:"Public"}
	// modelBlockBytes, err := json.Marshal(modelBlock)
	// err = ctx.GetStub().PutState(curHashId, modelBlockBytes)

	globalModelMetaInfo.CurHashId = curHashId
	globalModelMetaInfo.Round += 1
	globalModelMetaInfo.Clusters = clusters

	globalModelMetaInfoBytes, err := json.Marshal(globalModelMetaInfo)
	err = ctx.GetStub().PutState("global",globalModelMetaInfoBytes)
	return err
}

func (s *SmartContract) UpdateGlobalClusterInfo(ctx contractapi.TransactionContextInterface, clusters [][]int, localUpdates map[string]ModelBlock) error{
	globalModelMetaInfo, err := s.GetGlobalModelMetaInfo(ctx)
	
	globalModelMetaInfo.LocalUpdates = localUpdates
	globalModelMetaInfo.Clusters = clusters

	globalModelMetaInfoBytes, err := json.Marshal(globalModelMetaInfo)
	err = ctx.GetStub().PutState("global",globalModelMetaInfoBytes)
	return err
}

// func (s *SmartContract) UpdateGlobalClusterModelUrl(ctx contractapi.TransactionContextInterface, cluster string, modelUrl string) error{
// 	globalModelMetaInfo, err := s.GetGlobalModelMetaInfo(ctx)
	
// 	globalModelMetaInfo.ClusterModelUrls[cluster] = modelUrl

// 	globalModelMetaInfoBytes, err := json.Marshal(globalModelMetaInfo)
// 	err = ctx.GetStub().PutState("global",globalModelMetaInfoBytes)
// 	return err
// }

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


func main() {

	chaincode, err := contractapi.NewChaincode(new(SmartContract))

	if err != nil {
		fmt.Printf("Error create fabcar chaincode: %s", err.Error())
		return
	}

	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error starting fabcar chaincode: %s", err.Error())
	}
}


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

// ArgsortNew allocates and returns an array of indexes into the source float
// array.
func Argsort(src []float32) []int {
	inds := make([]int, len(src))
	for i := range src {
		inds[i] = i
	}
	a := argsort{s: src, inds: inds}
	sort.Sort(a)
	return inds
}
