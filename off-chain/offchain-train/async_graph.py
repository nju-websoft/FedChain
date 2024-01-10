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

np.random.seed(666)
random.seed(666)

def softmax(x):

    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

if __name__ == '__main__':

    if torch.cuda.is_available():
        print(torch.version.cuda)
        print(torch.__version__)
        device0 = "cuda:0"
        device1 = "cuda:1"
        device2 = "cuda:2"
        device3 = "cuda:3"
    else:
        device0 = device1 = device2 = device3 = "cpu"

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)
    print(conf)

    best_acc = 0.0
    save_name = conf["dataset"] + "_result_" + ("iid" if conf["iid"] else ("niid"+str(conf["niid_beta"]))) + "_" + "fedasync"
    save_dir = os.path.join("./results", save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_conf = json.dumps(conf)
    f2 = open(os.path.join(save_dir, "conf.json"), 'w')
    f2.write(save_conf)
    f2.close()

    dataset = dataset.GetDataSet(dataset_name=conf["dataset"], is_iid=conf["iid"], beta=conf["niid_beta"], client_num=conf["client_num"])
    server = Server(conf, eval_dataset=None, device=torch.device(device0))
    acc, loss = server.model_eval_graph(dataset=dataset)

    epochs = []
    accuracy = []
    validloss = []
    epochs.append(0)
    validloss.append(loss)
    accuracy.append(acc)

    clients = []
    print("Create {} clients".format(conf["client_num"]))

    best_acc = 0
    bad_count = 0
    for i in range(conf["client_num"]):
        clients.append(Client(
            conf=conf,
            #train_dataset=torch.utils.data.ConcatDataset([subset, global_subset]),
            train_dataset=None,
            test_dataset=None,
            id=i,
            global_model=server.global_model,
            #device=torch.device("cuda:" + str((i+1) % 4))
            device=torch.device(device0)
        ))

    # sort_indices = list(np.argsort(subset_size_list)[:3])
    # print(sort_indices)

    print("Start Training...")
    client_indices=list(range(conf["client_num"]))

    start = time.time()
    total_time_cost = 0
    for e in range(conf["global_epochs"]):

        k = random.randint(conf["min_k"], conf["max_k"])
        candidates = random.sample(clients, k)
        for c in candidates:
            c.local_train_graph(server.global_model, dataset, c.round)

        select = random.sample(candidates, 1)[0]

        # candidate = random.sample(sort_indices, 1)[0]
        # select = clients[candidate]
        # select.local_train(server.global_model, select.round)

        print("select {} for agg".format(select.client_id))
        server.async_model_weight_aggregate(copy.deepcopy(select.local_model), select.round)
        select.round = server.round
        acc, loss = server.model_eval_graph(dataset=dataset)

        epochs.append(e)
        validloss.append(loss)
        accuracy.append(acc)

        if acc > best_acc:
            best_acc = acc
            bad_count = 0
        else:
            bad_count += 1

        if bad_count >= 80:
            plot.plot(epochs, accuracy, label1="accuracy", dir=save_dir, name=save_name)
            plot.save_array(epochs, accuracy, validloss, dir=save_dir, name=save_name)
            print("Finish training")
            break

        if e % 10 == 0 and e > 0:
            plot.plot(epochs, accuracy, label1="accuracy", dir=save_dir, name=save_name)
            plot.save_array(epochs, accuracy, validloss, dir=save_dir, name=save_name)

        end = time.time()
        total_time_cost += (end - start)
        print("Round {}, acc:{}, loss:{}, time consume:{}\n".format(server.round, acc, loss, total_time_cost))
        start = end


    print("Finish the federated learning, time consume: {}".format(total_time_cost))
