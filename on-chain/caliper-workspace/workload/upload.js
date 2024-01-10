'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');
const fs = require('fs')

class MyWorkload extends WorkloadModuleBase {
    constructor() {
        super();
    }
    
    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);

    }
    
    async submitTransaction() {
        index=(index+1)%10;
        var org = 'org'+index;
        var curHashId = gethashcode();
        var url = "./" + org + "/" + curHashId;
        var localRound = round;
        var timestamp =  (new Date()).valueOf();
        //var vector = "[0.1, 0.2, 0.3]";
        var array = Array.from({ length: 100 }, () => Math.random());
        var vector = JSON.stringify(array);

        round = round + 1;

        try{
            const request = {
                contractId: this.roundArguments.contractId,
                contractFunction: 'UpdateLocalModel',
                invokerIdentity: 'Admin@org1.example.com',
                contractArguments: [org, curHashId, url, localRound, timestamp, vector],
                readOnly: false
            };
            await this.sutAdapter.sendRequests(request);
        }catch(err){
            console.log(err);
        }

        //var randomId = Math.floor(Math.random()*10)%accounts.length;
        /*var user=accounts[index];
        index=(index+1)%accounts.length;
        const myArgs = {
            contractId: this.roundArguments.contractId,
            contractFunction: 'CommitTriple',
            invokerIdentity: 'Admin@org1.example.com',
            contractArguments: [user,'whh','test1',user,'0.003','20210706'],
            readOnly: false
        };*/

        /*const myArgs = {
            contractId: this.roundArguments.contractId,
            contractFunction: 'GetTriple',
            invokerIdentity: 'Admin@org1.example.com',
            contractArguments: ['whh','test1'],
            readOnly: true
        };*/
       
        // try{
        //     var book=books[index];
        //     index=(index+1)%books.length;
        //     var datas=book.split(/\t/);
        //     const request = {
        //         contractId: this.roundArguments.contractId,
        //         contractFunction: 'UploadLocalModel',
        //         invokerIdentity: 'Admin@org1.example.com',
        //         contractArguments: [datas[0],datas[1],'author',datas[3],datas[4],'20210707'],
        //         readOnly: false
        //     };
        //     await this.sutAdapter.sendRequests(request);
        // }catch(err){
	    //     console.log(err);
        // }
    }
    
    async cleanupWorkloadModule() {

    }
}

function createWorkloadModule() {
    return new MyWorkload();
}

var str = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];
function generateMixed(n) {
     var res = "";
     for(var i = 0; i < n ; i ++) {
         var id = Math.ceil(Math.random()*35);
         res += str[id];
     }
     return res;
}

function hashCode(str) {
    var h = 0;
    var len = str.length;
    var t = 2147483648;
    for (var i = 0; i < len; i++) {
        h = 31 * h + str.charCodeAt(i);
        if (h > 2147483647) h %= t;
    }
    /*var t = -2147483648 * 2;
     while (h > 2147483647) {
     h += t
     }*/
    return h;
}

function randomWord(randomFlag, min, max) {
    var str = "",
        range = min,
        arr = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];

    if (randomFlag) {
        range = Math.round(Math.random() * (max - min)) + min;
    }
    for (var i = 0; i < range; i++) {
        var pos = Math.round(Math.random() * (arr.length - 1));
        str += arr[pos];
    }
    return str;
}

function gethashcode() {
    var timestamp = (new Date()).valueOf();
    var myRandom=randomWord(false,6);
    var hashcode=hashCode(myRandom+timestamp.toString());
    return hashcode;
}

var accounts=[];
var subjects=[];
var books=[];
var index=0;
var round=0;

module.exports.createWorkloadModule = createWorkloadModule;
