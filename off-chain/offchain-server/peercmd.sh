CONFIG_ROOT=/opt/gopath/src/github.com/hyperledger/fabric/peer
ORG1_MSPCONFIGPATH=${CONFIG_ROOT}/crypto/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
ORG1_TLS_ROOTCERT_FILE=${CONFIG_ROOT}/crypto/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
ORG2_MSPCONFIGPATH=${CONFIG_ROOT}/crypto/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp
ORG2_TLS_ROOTCERT_FILE=${CONFIG_ROOT}/crypto/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
ORDERER_TLS_ROOTCERT_FILE=${CONFIG_ROOT}/crypto/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem

CHANNEL_NAME=mychannel
CHAINCODE_NAME=federated
CHAINCODE_PATH=github.com/chaincode/${CHAINCODE_NAME}/go/

if [ "$1" == "getOrg" ]; then
   	     peer chaincode query \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
   	     -c '{"Args":["GetOrg"]}'

elif [ "$1" == "testMongo" ]; then
   	     peer chaincode query \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
   	     -c '{"Args":["TestMongo"]}'

elif [ "$1" == "getKey" ]; then
 	     peer chaincode invoke \
	     -o localhost:7050 \
             --ordererTLSHostnameOverride orderer.example.com \
   	     --tls --cafile ../test-network/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
             --peerAddresses localhost:7051 --tlsRootCertFiles ../test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
             --peerAddresses localhost:9051 --tlsRootCertFiles ../test-network/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
   	     -c '{"Args":["GetIncreaseKey"]}'


elif [ "$1" == "getOrgInfo" ]; then
            org=$2
   	     peer chaincode query \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
   	     -c '{"Args":["GetOrgInfo","'${org}'"]}'

elif [ "$1" = "getGlobal" ]; then
   	     peer chaincode query \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
   	     -c '{"Args":["GetGlobalModel"]}'

elif [ "$1" = "getModelBlock" ]; then
	key=$2
   	     peer chaincode query \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
   	     -c '{"Args":["GetModelBlock","'${key}'"]}'

elif [ "$1" = "uploadLocal" ]; then
        org=$2
        key=$3
   	     peer chaincode invoke \
	     -o localhost:7050 \
             --ordererTLSHostnameOverride orderer.example.com \
   	     --tls --cafile ../test-network/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
             --peerAddresses localhost:7051 --tlsRootCertFiles ../test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
             --peerAddresses localhost:9051 --tlsRootCertFiles ../test-network/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
   	     -c '{"Args":["UploadLocalModel","'${org}'","'${key}'"]}'

elif [ "$1" == "avg" ]; then
 	     peer chaincode invoke \
	     -o localhost:7050 \
             --ordererTLSHostnameOverride orderer.example.com \
   	     --tls --cafile ../test-network/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
   	     -C ${CHANNEL_NAME} \
   	     -n ${CHAINCODE_NAME} \
             --peerAddresses localhost:7051 --tlsRootCertFiles ../test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
             --peerAddresses localhost:9051 --tlsRootCertFiles ../test-network/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \
   	     -c '{"Args":["FederatedAvg"]}'

fi
