'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');
const fs = require('fs')

class MyWorkload extends WorkloadModuleBase {
    constructor() {
        super();
    }
    
    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);

        // if(workerIndex==0){
        //     const datasetPath=__dirname+roundArguments.datasetPath;
        //     try {
        //         // read contents of the file
        //         const data2 = fs.readFileSync(datasetPath+'subjects.txt', 'UTF-8');
        //         subjects  = data2.split(/\r?\n/);
        //         console.log(subjects);
        //     } catch (err) {
        //         console.error(err);
        //     }
        // }

        // else{
        //     console.log("index:"+workerIndex);
        // }
    }
    
    async submitTransaction() {
       
        try{
            //console.log("########################"+this.roundArguments.contractId);
            const answerRequest = {
                contractId: this.roundArguments.contractId,
                contractFunction: 'getGlobalModelMetaInfo',
                invokerIdentity: 'Admin@org1.example.com',
                contractArguments: [],
                readOnly: false
            };
            await this.sutAdapter.sendRequests(answerRequest);

        }catch(err){
	        console.log(err);
        }
    }
    
    async cleanupWorkloadModule() {

    }
}

function createWorkloadModule() {
    return new MyWorkload();
}

var subjects=[];
var index=0;

module.exports.createWorkloadModule = createWorkloadModule;
