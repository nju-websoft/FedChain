CONFIG_ROOT=/opt/gopath/src/github.com/hyperledger/fabric/peer
ORG1_MSPCONFIGPATH=${CONFIG_ROOT}/crypto/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
ORG1_TLS_ROOTCERT_FILE=${CONFIG_ROOT}/crypto/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
ORG2_MSPCONFIGPATH=${CONFIG_ROOT}/crypto/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
ORG2_TLS_ROOTCERT_FILE=${CONFIG_ROOT}/crypto/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
ORDERER_TLS_ROOTCERT_FILE=${CONFIG_ROOT}/crypto/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem

CHANNEL_NAME=mychannel
CHAINCODE_NAME=federated
CHAINCODE_PATH=github.com/chaincode/${CHAINCODE_NAME}/go/

if [ "$1" == "querycommitted" ]; then
	docker exec \
   	    cli \
   	    peer lifecycle chaincode querycommitted --channelID ${CHANNEL_NAME} --name ${CHAINCODE_NAME}

elif [ "$1" == "getOrg" ]; then
   	     peer chaincode query \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
   	     -c '{"Args":["GetOrg"]}'

elif [ "$1" = "authorize" ]; then
        member=$2
        balance=$3
   	     peer chaincode invoke \
	     -o localhost:7050 \
             --ordererTLSHostnameOverride orderer.example.com \
   	     --tls --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
             --peerAddresses localhost:7051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
             --peerAddresses localhost:9051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
   	     -c '{"Args":["AuthorizeMember","'${member}'","'${balance}'"]}'


elif [ "$1" = "deauthorize" ]; then
        member=$2
   	     peer chaincode invoke \
	     -o localhost:7050 \
             --ordererTLSHostnameOverride orderer.example.com \
   	     --tls --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
             --peerAddresses localhost:7051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
             --peerAddresses localhost:9051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
   	     -c '{"Args":["DeauthorizeMember","'${member}'"]}'

elif [ "$1" = "getAccount" ]; then
	member=$2
   	     peer chaincode query \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
   	     -c '{"Args":["GetAccount","'${member}'"]}'

elif [ "$1" = "getOrgAccount" ]; then
	member=$2
   	     peer chaincode query \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
   	     -c '{"Args":["GetOrgAccounts","'${member}'"]}'

elif [ "$1" = "query" ]; then
	subjectT=$2
	predicateT=$3
   	     peer chaincode invoke \
	     -o localhost:7050 \
             --ordererTLSHostnameOverride orderer.example.com \
   	     --tls --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
             --peerAddresses localhost:7051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
             --peerAddresses localhost:9051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
   	     -c '{"Args":["QueryTriple","'${subjectT}'","'${predicateT}'"]}'

elif [ "$1" = "queryAll" ]; then
	member=$2
   	     peer chaincode query \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
   	     -c '{"Args":["GetAllAnswers"]}'

elif [ "$1" = "get" ]; then
	subjectT=$2
	predicateT=$3
   	     peer chaincode query \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
   	     -c '{"Args":["GetTriple","'${subjectT}'","'${predicateT}'"]}'

elif [ "$1" = "subjectQuery" ]; then
	subjectT=$2
   	     peer chaincode query \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
   	     -c '{"Args":["GetSubjectTriple","'${subjectT}'"]}'

elif [ "$1" = "commit" ]; then
        member=$2
	subjectT=$3
	predicateT=$4
	objectT=$5
	scoreT=$6
        timestamp=$7
   	     peer chaincode invoke \
	     -o localhost:7050 \
             --ordererTLSHostnameOverride orderer.example.com \
   	     --tls --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
             --peerAddresses localhost:7051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
             --peerAddresses localhost:9051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
   	     -c '{"Args":["CommitTriple","'${member}'","'${subjectT}'","'${predicateT}'","'${objectT}'","'${scoreT}'","'${timestamp}'"]}'

elif [ "$1" = "answer" ]; then
        member=$2
	subjectT=$3
	predicateT=$4
	reward=$5
        timeCounts=$6
        mulValue=$7
   	     peer chaincode invoke \
	     -o localhost:7050 \
             --ordererTLSHostnameOverride orderer.example.com \
   	     --tls --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
             --peerAddresses localhost:7051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
             --peerAddresses localhost:9051 --tlsRootCertFiles ${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
   	     -c '{"Args":["RequestTriple","'${member}'","'${subjectT}'","'${predicateT}'","'${reward}'","'${timeCounts}'","'${mulValue}'"]}'


fi
