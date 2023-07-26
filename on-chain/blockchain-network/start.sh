CHAINCODE_NAME=$1
ADD=${2:-"NONE"} #ADD or NONE

docker volume prune

source network.sh up createChannel -ca
source network.sh deployCC -ccn ${CHAINCODE_NAME} -ccp ../chaincode/${CHAINCODE_NAME}/go -ccl go

if [ $ADD == "add" ];then

cd addOrg3
source addOrg3.sh up
cd ../
source path.sh 3 11051
source lifecycle.sh ${CHAINCODE_NAME}

cd addOrg4
source addOrg4.sh up
cd ../
source path.sh 4 13051
source lifecycle.sh ${CHAINCODE_NAME}

cd addOrg5
source addOrg5.sh up
cd ../
source path.sh 5 15051
source lifecycle.sh ${CHAINCODE_NAME}

cd addOrg6
source addOrg6.sh up
cd ../
source path.sh 6 17051
source lifecycle.sh ${CHAINCODE_NAME}

elif [ $ADD == "test" ];then

cd addOrg3
source addOrg3.sh up
cd ../
source path.sh 3 11051
source lifecycle.sh ${CHAINCODE_NAME}

cd addOrg4
source addOrg4.sh up
cd ../
source path.sh 4 13051
source lifecycle.sh ${CHAINCODE_NAME}

fi
#tar -zvcf organizations.tar.gz organizations
