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

# random.seed(333)

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
    save_name = conf["dataset"] + "_result_" + ("iid" if conf["iid"] else ("niid" + str(conf["niid_beta"]))) + "_" + (
        "exchange" if conf["exchange"] else "base") + ("_cluster" if conf["cluster"] else "") + ("_prox" if conf["prox"] else "")
    suffix = ("" if conf["ablation"]=="full" else ("_without_" + conf["ablation"]))
    save_dir = os.path.join("./results", save_name + suffix)
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
    acc, loss = server.model_eval_graph(dataset=dataset)

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
            # train_dataset=torch.utils.data.ConcatDataset([subset, global_subset]),
            train_dataset=None,
            test_dataset=None,
            id=i,
            global_model=server.global_model,
            # device=torch.device("cuda:" + str((i+1) % 4))
            device=torch.device(device0)
        ))

        client_datalen.append(len(dataset.client_idcs[i]))
    print(client_datalen)

    print("Start Training...")
    client_indices=list(range(conf["client_num"]))

    if conf["exchange"]:
        extractors = []
        for id in client_indices:
            extractors.append(clients[id].extractor)
    best_acc = 0
    bad_count = 0
    start = time.time()
    for e in range(1, conf["global_epochs"]):

        k = random.randint(conf["min_k"], conf["max_k"])
        candidates = random.sample(clients, k)

        if conf["exchange"]:
            for c in candidates:
                similarity_vector = c.extractor.sparsity_similarity_vector(c.extractor.mean_list, extractors)
                c.extractor.similarity_vector = similarity_vector

                p = random.random()
                if p > conf["exchange_probability"]:
                    if conf["cluster"]:
                        c.local_train_graph(server.cluster_models[server.clients_cluster_map[c.client_id]], dataset)
                    else:
                        c.local_train_graph(server.global_model, dataset)
                        #c.moon_train_graph(server.global_model, dataset=dataset)

                else:
                    similarity_vector = c.extractor.sparsity_similarity_vector(c.extractor.mean_list, extractors)
                    c.extractor.similarity_vector = similarity_vector
                    #exchange_indices = np.argsort(-similarity_vector)[1:(1+random.randint(1, conf["max_collaborative_count"]))]

                    ratios = similarity_vector / c.loss_list
                    threshold = conf["threshold"]
                    selected_indices = np.argsort(ratios)
                    collaborative_indices = []
                    for id in selected_indices:
                        if id != c.client_id and threshold > 0 and len(collaborative_indices) < conf["max_collaborative_count"]:
                            collaborative_indices.append(id)
                            threshold -= c.loss_list[id]

                    #collaborative_indices = random.sample(client_indices, 1)

                    print("exchange map: {} \n vector:{} \n loss list:{}".format(collaborative_indices, similarity_vector, c.loss_list))

                    acc1, _ = c.eval_model_graph(dataset=dataset)
                    print("Before : client {} valid acc {}".format(c.client_id, acc1))

                    #acc2, _ = c.fuse_model_by_teachers_graph([[clients[id].client_id, clients[id].local_model] for id in collaborative_indices], dataset)
                    acc2, _ = c.fuse_model_by_teachers_graph([clients[id] for id in collaborative_indices], dataset)
                    print("After : client {} valid acc {}".format(c.client_id, acc2))


            if conf["cluster"]:
                acc, loss = server.cluster_model_weight_aggregate([(c.client_id, c.local_model, c.extractor.similarity_vector) for c in candidates], dataset=dataset)

            else:
                #server.model_weight_aggregate([c.local_model for c in candidates])
                acc, loss = server.model_weight_aggregate([copy.deepcopy(c.local_model) for c in candidates],
                                                          client_datalen, total_len, [c.client_id for c in candidates],
                                                          dataset=dataset)

                #acc, loss = server.model_eval_graph(dataset=dataset)

        else:
            for c in candidates:
                if conf["cluster"]:
                    c.local_train(server.cluster_models[server.clients_cluster_map[c.client_id]])
                else:
                    c.local_train_graph(server.global_model, dataset)

            if conf["cluster"]:
                acc, loss = server.cluster_model_weight_aggregate(
                    [(c.client_id, c.local_model, c.extractor.similarity_vector) for c in candidates])
            else:
                acc, loss = server.model_weight_aggregate([copy.deepcopy(c.local_model) for c in candidates],
                                                          client_datalen, total_len, [c.client_id for c in candidates],
                                                          dataset=dataset)

        # acc, loss = server.model_eval()
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

        # if bad_count >= 80:
        #     plot.plot(epochs, accuracy, label1="accuracy", dir=save_dir, name=save_name)
        #     plot.save_array(epochs, accuracy, validloss, dir=save_dir, name=save_name)
        #     print("Finish training")
        #     break

        if e % 10 == 0 and e > 0:
            plot.plot(epochs, accuracy, label1="accuracy", dir=save_dir, name=save_name)
            plot.save_array(epochs, accuracy, validloss, dir=save_dir, name=save_name)

    print("Finish the federated learning, best acc: {}".format(best_acc))

    session.close()