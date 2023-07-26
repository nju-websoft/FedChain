import argparse, json
import datetime
import os
import logging
import time

import numpy as np
import torch, random

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED, FIRST_COMPLETED

from server import *
from client import *
import dataset
import models
import plot

# random.seed(666)

if __name__ == '__main__':

    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if torch.cuda.is_available():
        print(torch.version.cuda)
        print(torch.__version__)
        device0 = "cuda:0"  # gpu:1
        device1 = "cuda:1"  # gpu:3
        device2 = "cuda:2"  # gpu:0
        device3 = "cuda:3"  # gpu:2
    else:
        device0 = device1 = device2 = device3 = "cpu"

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)
    print(conf)

    #executor = ThreadPoolExecutor(max_workers=conf['k'])

    session = requests.Session()
    org_server = "http://114.212.82.242:8080/"
    upload_local_api = "testupload"
    headers = {'content-type': 'application/json'}

    best_acc = 0.0
    save_name = conf["dataset"] + "_result_" + ("iid" if conf["iid"] else ("niid" + str(conf["niid_beta"]))) + "_scalesfl"
    save_dir = os.path.join("./results", save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_conf = json.dumps(conf)
    f2 = open(os.path.join(save_dir, "conf.json"), 'w')
    f2.write(save_conf)
    f2.close()

    epochs = []
    accuracy = []
    validloss = []

    dataset = dataset.GetDataSet(dataset_name=conf["dataset"], is_iid=conf["iid"], beta=conf["niid_beta"], client_num=conf["client_num"])
    server = Server(conf, eval_dataset=None, device=torch.device(device0))
    acc, loss = server.model_eval_graph(dataset)

    epochs.append(0)
    validloss.append(loss)
    accuracy.append(acc)

    clients = []
    print("Create {} clients".format(conf["client_num"]))


    client_datalen = []
    total_len = len(dataset.cora_dataset.idx_train)
    for i in range(conf["client_num"]):
        clients.append(Client(
            conf=conf,
            #train_dataset=torch.utils.data.ConcatDataset([subset, global_subset]),
            train_dataset=None,
            test_dataset=None,
            id=i,
            global_model=server.global_model,
            #device=torch.device("cuda:" + str((i+1) % 4))
            device=torch.device(device1)
        ))

        client_datalen.append(len(dataset.client_idcs[i]))
    print(client_datalen)

    print("Start Training...")
    client_indices=list(range(conf["client_num"]))
    best_acc = 0
    bad_count = 0
    start = time.time()
    for e in range(1, conf["global_epochs"]):
        k = random.randint(conf["min_k"], conf["max_k"])
        candidates = random.sample(clients, k)
        for c in candidates:
            c.local_train_graph(server.global_model, dataset=dataset)

        # acc, loss = server.scalesfl_cluster_model_weight_aggregate(clients, client_datalen, total_len)
        #acc, loss = server.scalesfl_cluster_model_weight_aggregate(clients, client_datalen, total_len, dataset)
        acc, loss = server.model_weight_aggregate([c.local_model for c in candidates], client_datalen=client_datalen, total_len=total_len, ids=[c.client_id for c in candidates], dataset=dataset)

        end = time.time()
        print("Epoch %d, acc: %f, loss: %f, time consume: %f\n" % (e, acc, loss, end-start))

        # if e > 50 and acc > best_acc:
        #     best_acc = acc
        #     torch.save(server.global_model.state_dict(), os.path.join(save_dir, 'global_model.pth'))

        epochs.append(e)
        validloss.append(loss)
        accuracy.append(acc)

        if acc > best_acc:
            best_acc = acc
            bad_count = 0
        else:
            bad_count += 1

        # if bad_count >= 40:
        #     plot.plot(epochs, accuracy, label1="accuracy", dir=save_dir, name=save_name)
        #     plot.save_array(epochs, accuracy, validloss, dir=save_dir, name=save_name)
        #     print("Finish training")
        #     break

        if e % 10 == 0 and e > 0:
            plot.plot(epochs, accuracy, label1="accuracy", dir=save_dir, name=save_name)
            plot.save_array(epochs, accuracy, validloss, dir=save_dir, name=save_name)

    print("Finish the federated learning, best acc: {}".format(best_acc))

    session.close()