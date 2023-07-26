import argparse, json
import datetime
import os
import logging

import numpy as np
import torch, random
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

from server import *
from client import *
import dataset
from torch import nn
import models
import plot

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        global mean_list
        mean_temp = np.zeros((x.size(0), 1))
        # for name, module in self.submodule.features._modules.items():
        #
        #     if name in self.extracted_layers:
        #         x = module(x)
        #         temp = x.cpu().detach().numpy()
        #         density = np.count_nonzero(temp, (2, 3))
        #         sparsity = (np.size(temp, 2) * np.size(temp, 3) - density) / (np.size(temp, 2) * np.size(temp, 3))
        #         mean_temp = np.concatenate((mean_temp, sparsity), axis=1)
        #
        #     else:
        #         x = module(x)
        x = self.submodule(x)
        print(x)
        # mean_temp = np.delete(mean_temp, 0, 1)
        # mean_list = np.concatenate((mean_list, mean_temp), axis=0)


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

    print(torch.cuda.is_available())
    print(torch.version.cuda)

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)
    print(conf)

    dataset = dataset.GetDataSet(dataset_name=conf["dataset"], is_iid=conf["iid"], beta=conf["niid_beta"])

    server = Server(conf, dataset.test_dataset, device=torch.device("cuda:0"))
    #server = Server(conf, eval_dataset=None, device=torch.device(device1))

    #数据集全集训练：上限
    # client = Client(
    #         conf=conf,
    #         train_dataset=dataset.train_dataset,
    #         test_dataset=dataset.test_dataset,
    #         id=0,
    #         global_model=None,
    #         device=torch.device("cuda:"+str(1)))
    #
    # # client = Client(
    # #         conf=conf,
    # #         #train_dataset=torch.utils.data.ConcatDataset([subset, global_subset]),
    # #         train_dataset=None,
    # #         test_dataset=None,
    # #         id=0,
    # #         #device=torch.device("cuda:" + str((i+1) % 4))
    # #         device=torch.device(device1)
    # #     )
    # print("Create 1 client for training in complete dataset.")
    #
    #
    # client.local_train(client.local_model)

    #client.local_train_graph(client.local_model, dataset)

    #客户端单独训练：下限
    # save_name = conf["dataset"] + "_result_" + ("iid" if conf["iid"] else ("niid"+str(conf["niid_beta"]))) + "_" + "independent"
    # save_dir = os.path.join("./results", save_name)
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    #
    # save_conf = json.dumps(conf)
    # f2 = open(os.path.join(save_dir, "conf.json"), 'w')
    # f2.write(save_conf)
    # f2.close()
    #
    # acc, loss = server.model_eval()
    # epochs = []
    # accuracy = []
    # validloss = []
    # epochs.append(0)
    # validloss.append(loss)
    # accuracy.append(acc)

    clients = []
    client_datalen = []
    print("Create {} clients".format(conf["client_num"]))
    shard_size = dataset.train_data_size // conf["client_num"]
    shard_id = np.random.permutation(dataset.train_data_size // shard_size)
    for i in range(conf["client_num"]):
        if conf["iid"] == False:
            subset = torch.utils.data.Subset(dataset.train_dataset, dataset.client_idcs[i])

        else:
            shards = list(range(shard_id[i] * shard_size, shard_id[i] * shard_size + shard_size))
            subset = torch.utils.data.Subset(dataset.train_dataset, shards)

        clients.append(Client(
            conf=conf,
            # train_dataset=torch.utils.data.ConcatDataset([subset, global_subset]),
            train_dataset=subset,
            test_dataset=dataset.test_dataset,
            id=i,
            global_model=server.global_model,
            # device=torch.device("cuda:" + str((i+1) % 4))
            device=torch.device(device1)
        ))
        client_datalen.append(len(subset))

    id = np.argmax(client_datalen)
    clients[id].local_train(clients[id].local_model)

    # clients[].local_train(client.local_model)
    # for e in range(1, conf["global_epochs"]):
    #     k = random.randint(conf["min_k"], conf["max_k"])
    #     candidates = random.sample(clients, k)
    #
    #     avg_sum = 0.0
    #     for c in candidates:
    #         acc, loss = c.local_train_graph(c.local_model, dataset=dataset)
    #         avg_sum += acc
    #     avg_sum /= k
    #     print("Epoch {}: Independent training average acc: {}\n".format(e, avg_sum))
