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
    save_name = conf["dataset"] + "_result_" + ("iid" if conf["iid"] else ("niid" + str(conf["niid_beta"]))) + "_fedat"
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
    server = Server(conf, dataset.test_dataset, device=torch.device(device0))
    acc, loss = server.model_eval()

    epochs.append(0)
    validloss.append(loss)
    accuracy.append(acc)

    clients = []
    print("Create {} clients".format(conf["client_num"]))


    shard_size = dataset.train_data_size // conf["client_num"]
    shard_id = np.random.permutation(dataset.train_data_size // shard_size)

    client_datalen = []
    total_len = len(dataset.train_dataset)
    for i in range(conf["client_num"]):
        if conf["iid"] == False:
            subset = torch.utils.data.Subset(dataset.train_dataset, dataset.client_idcs[i])

        else:
            shards = list(range(shard_id[i] * shard_size, shard_id[i] * shard_size + shard_size))
            subset = torch.utils.data.Subset(dataset.train_dataset, shards)

        clients.append(Client(
            conf=conf,
            #train_dataset=torch.utils.data.ConcatDataset([subset, global_subset]),
            train_dataset=subset,
            test_dataset=dataset.test_dataset,
            id=i,
            global_model=server.global_model,
            #device=torch.device("cuda:" + str((i+1) % 4))
            device=torch.device(device0)
        ))

        client_datalen.append(len(subset))


    clusters = [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
    update_count = [1, 1, 1]
    #probability = [0.7, 0.25, 0.05]
    probability = [0.3, 0.3, 0.4]

    tier_models = [copy.deepcopy(server.global_model) for i in range(len(clusters))]

    server.clients_cluster_map = {}
    for id in range(conf["client_num"]):
        server.clients_cluster_map[id] = id % 3

    print("Start Training...")
    client_indices = list(range(conf["client_num"]))
    best_acc = 0
    bad_count = 0
    start = time.time()
    for e in range(1, conf["global_epochs"]):
        chosen_cluster = random.choices(range(len(clusters)), probability)[0]
        nums = random.randint(conf["min_k"], conf["max_k"])
        chosen_clients = random.sample(clusters[chosen_cluster], nums)
        candidates = [clients[id] for id in chosen_clients]

        for c in candidates:
            c.local_train(server.global_model)

        fed_state_dict = collections.OrderedDict()
        weight = 1.0 / nums

        for key, param in server.global_model.state_dict().items():
            sum = torch.zeros_like(param)
            for c in candidates:
                sum.add_(c.local_model.state_dict()[key].clone().to(server.device) * weight)
            fed_state_dict[key] = sum

        tier_models[chosen_cluster].load_state_dict(fed_state_dict)

        update_count[chosen_cluster] += 1
        sum = 0
        for cnt in update_count:
            sum += cnt

        print("cluster: {}, update_count: {}".format(chosen_clients, update_count))

        weight = update_count[len(update_count)-1-chosen_cluster] / sum
        weight = 1
        acc, loss = server.async_model_weight_aggregate(tier_models[chosen_cluster], 0, weight=weight)

        # weights = [0.0, 0.0, 0.0]
        # for i in range(len(weights)):
        #     weights[i] = update_count[len(update_count)-1-i] / sum
        # tier_models[chosen_cluster].load_state_dict(fed_state_dict)
        # acc, loss = server.model_weight_aggregate(tier_models, ids=chosen_clients, w=weights)

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

        if bad_count >= 50:
            plot.plot(epochs, accuracy, label1="accuracy", dir=save_dir, name=save_name)
            plot.save_array(epochs, accuracy, validloss, dir=save_dir, name=save_name)
            print("Finish training")
            break

        if e % 10 == 0 and e > 0:
            plot.plot(epochs, accuracy, label1="accuracy", dir=save_dir, name=save_name)
            plot.save_array(epochs, accuracy, validloss, dir=save_dir, name=save_name)

    print("Finish the federated learning, best acc: {}".format(best_acc))

    session.close()