# FedChain
This repository is for our submission. 

## Start

#### 1.  We use Hyperledger Fabric v2.2 as the default blockchain.


##### Prerequisites:
1. [HyperLedger Fabric v2.2.x LTS](https://www.hyperledger.org/projects/fabric "HyperLedger Fabric Homepage")
2. Download this repository, and merge BEAS/fabric-samples with the HyperLedger fabric-samples directory.

##### Network Setup:
```
$  cd fabric-samples/blockchain-network
$  ./network.sh fedchain
```
#### 2.  Each client has a lightweight server based on node.js listening and processing requests off the chain.

1. Create a working dictionary in fabric-samples
```
$  cd fabric-samples/offchain-server
```
2. Install server dependancies:
```
$  npm install
```
3. Run web server for each client:
```
$  node client.js
```

#### 3.   We start multiple client processes for local training.

1. Start 10 clients
```
$ ./start_clients.sh 10
```
