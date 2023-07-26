package main

import (
	"encoding/json"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

type SubjectAttribute struct {
	Org string `json:"org"`
	GrantTimestamp int `json:"grantTimestamp"`
}

//k: resourceId v: {sid, subject}
//expiredTime: 30s
type AccessControlTable struct {
	AccessTable map[string]SubjectAttribute `json:"accessTable"`
	ExpiredTime int `json:"expiredTime"`
	Policy string `json:"policy"`
}

type RequestEncInfo struct {
	RequesterID string `json:"requesterID"`
	ModelID string `json:"modelID"`
	EncryptionCode string `json:"code"`
}

type GrantInfo struct {
	RequesterID string `json:"requesterID"`
	ModelID string `json:"modelID"`
	Granted string `json:"granted"`
}

func (s *SmartContract) RecordResource(ctx contractapi.TransactionContextInterface, resourceId string) error {
	var accessControlTable AccessControlTable = AccessControlTable{AccessTable:make(map[string]SubjectAttribute), ExpiredTime:30, Policy:""}
	accessControlTableBytes, err := json.Marshal(accessControlTable)
	err = ctx.GetStub().PutState(resourceId, accessControlTableBytes)
	return err
}

//####################################Single Request Access Control##################

func (s *SmartContract) CheckAccess(ctx contractapi.TransactionContextInterface, requestee string, sid string, requestTime int) (string, error) {
	localModelMetaInfo, err := s.GetLocalModelMetaInfo(ctx, requestee)

	//accessControlTableBytes, err := ctx.GetStub().GetState(localModelMetaInfo.CurHashId)
	accessControlTableBytes, err := ctx.GetStub().GetState("000")	
	var accessControlTable AccessControlTable
	_ = json.Unmarshal(accessControlTableBytes, &accessControlTable) 

	if value, ok := accessControlTable.AccessTable[sid]; ok{
		if requestTime > value.GrantTimestamp + accessControlTable.ExpiredTime {
			return "grant time expired", err
		}

		return localModelMetaInfo.CurHashId, err
	}
	return "no grant access", err
}

func (s *SmartContract) GrantAccess(ctx contractapi.TransactionContextInterface, requestee string, sid string, org string, requestTime int) (string, error) {
	localModelMetaInfo, err := s.GetLocalModelMetaInfo(ctx, requestee)
	
	//accessControlTableBytes, err := ctx.GetStub().GetState(localModelMetaInfo.CurHashId)	
	accessControlTableBytes, err := ctx.GetStub().GetState("000")	
	var accessControlTable AccessControlTable
	_ = json.Unmarshal(accessControlTableBytes, &accessControlTable) 

	if value, ok := accessControlTable.AccessTable[sid]; ok{
		if requestTime > value.GrantTimestamp + accessControlTable.ExpiredTime {
			value.GrantTimestamp = requestTime
			accessControlTable.AccessTable[sid] = value
		}
	}else{
		value = SubjectAttribute{Org:org, GrantTimestamp:requestTime}
		accessControlTable.AccessTable[sid] = value
	}

	accessControlTableBytes, err = json.Marshal(accessControlTable)
	//err = ctx.GetStub().PutState(localModelMetaInfo.CurHashId, accessControlTableBytes)
	err = ctx.GetStub().PutState("000", accessControlTableBytes)
	return localModelMetaInfo.LocalModelBlock.ModelUrl, err
}

//####################################Group Requests Access Control##################

func (s *SmartContract) CheckGroupAccess(ctx contractapi.TransactionContextInterface, requestee string, sids []string, requestTimes []int) ([]string, error) {
	localModelMetaInfo, err := s.GetLocalModelMetaInfo(ctx, requestee)
	
	//accessControlTableBytes, err := ctx.GetStub().GetState(localModelMetaInfo.CurHashId)		
	accessControlTableBytes, err := ctx.GetStub().GetState("000")
	var accessControlTable AccessControlTable
	_ = json.Unmarshal(accessControlTableBytes, &accessControlTable) 

	result := []string{}
	for i := 0; i < len(sids); i++{
		if value, ok := accessControlTable.AccessTable[sids[i]]; ok{
			if requestTimes[i] > value.GrantTimestamp + accessControlTable.ExpiredTime {
				result = append(result, "grant time expired")
			}else{
				result = append(result, localModelMetaInfo.LocalModelBlock.ModelUrl)
			}
		}else{
			result = append(result, "no grant access")
		}
	}

	return result, err
}

func (s *SmartContract) GrantGroupAccess(ctx contractapi.TransactionContextInterface, requestee string, sids []string, orgs []string, requestTimes []int) ([]string, error) {
	localModelMetaInfo, err := s.GetLocalModelMetaInfo(ctx, requestee)
	
	//accessControlTableBytes, err := ctx.GetStub().GetState(localModelMetaInfo.CurHashId)	
	accessControlTableBytes, err := ctx.GetStub().GetState(requestee)			
	var accessControlTable AccessControlTable
	_ = json.Unmarshal(accessControlTableBytes, &accessControlTable) 

	result := []string{}
	for i := 0; i < len(sids); i++{
		if value, ok := accessControlTable.AccessTable[sids[i]]; ok{
			if requestTimes[i] > value.GrantTimestamp + accessControlTable.ExpiredTime {
				value.GrantTimestamp = requestTimes[i]
				accessControlTable.AccessTable[sids[i]] = value
			}
		}else{
			value = SubjectAttribute{Org:orgs[i], GrantTimestamp:requestTimes[i]}
			accessControlTable.AccessTable[sids[i]] = value
		}

		result = append(result, localModelMetaInfo.LocalModelBlock.ModelUrl)
	}

	accessControlTableBytes, err = json.Marshal(accessControlTable)
	//err = ctx.GetStub().PutState(localModelMetaInfo.CurHashId, accessControlTableBytes)
	err = ctx.GetStub().PutState(requestee, accessControlTableBytes)
	return result, err
}

func (s *SmartContract) GetRequestEncInfo(ctx contractapi.TransactionContextInterface, requesterID string, modelId string) (*RequestEncInfo,error) {
	RequestEncInfoBytes, err := ctx.GetStub().GetState(requesterID+","+modelId)

	var requestEncInfo RequestEncInfo
	_ = json.Unmarshal(RequestEncInfoBytes, &requestEncInfo)
	return &requestEncInfo, err
}

func (s *SmartContract) UpdateGrantInfo(ctx contractapi.TransactionContextInterface, requesterID string, modelId string, granted string) error {
	var grantInfo GrantInfo = GrantInfo{RequesterID:requesterID, ModelID:modelId, Granted:granted}
	grantInfoBytes, err := json.Marshal(grantInfo)
	err = ctx.GetStub().PutState(requesterID+","+modelId, grantInfoBytes)
	return err
}