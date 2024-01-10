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
3. Install Caliper dependancies:
```
$  npm install --only=prod @hyperledger/caliper-cli
$  npx caliper bind --caliper-bind-sut fabric:2.1
```

#### 3.   We start multiple client processes for local training.

1. Start 10 clients
```
$ ./start_clients.sh 10
```
2. End
```
$ ./end.sh
```

#### 4.   To facilitate the performance comparison of various federated learning methods, we implemented an off-chain version to simplify operation.

1. Modify the hyperparameters in ./config/conf.json
2. Run the script such as:
```
$  python main.py -c ./config/conf.json
```

#### 5.   We plot the convergence comparison with plot.py based on the results in ./offchain/offchain-train/results