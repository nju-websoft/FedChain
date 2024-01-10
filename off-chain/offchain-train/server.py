import collections
import os
import models, torch
import requests
import schedule
import time
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
import copy
from attack.defenses import *

np.random.seed(33)

class Server(object):

    def __init__(self, conf, eval_dataset=None, device=None):

        self.conf = conf

        self.eval_dataset = eval_dataset
        if eval_dataset is not None:
            self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=False)

        #model = models.get_model(conf["model_name"], load_from_local=True)
        # model = models.CNNMnist()
        if self.conf["dataset"] == "cifar10":
            model = models.VGGCifar(num_classes=10)

        elif self.conf["dataset"] == "cifar100":
            model = models.VGGCifar(num_classes=100)

        elif self.conf["dataset"] == "mnist" or self.conf["dataset"] == "fashion_mnist":
            model = models.VGGMNIST()

        elif self.conf["dataset"] == "imagenet":
            model = models.VGGCifar(num_classes=200)

        elif self.conf["dataset"] == "agnews":
            model = models.BertClassifer.from_pretrained('./models/bert-base-uncased/', num_labels=4)

        elif self.conf["dataset"] == "shakespeare":
            model = models.CharLSTM()

        elif self.conf["dataset"] == "sent140":
            model = models.RNN()
            # model = models.CharLSTM()

        elif self.conf["dataset"] == "cora":
            #model = models.GCN(nfeat=1433, nhid=32, nclass=7, dropout=0.1)
            model = models.GCN(input_dim=1433, hidden_dim=16, num_classes=7, p=0.5)

        elif self.conf["dataset"] == "citeseer":
            #model = models.GCN(nfeat=3703, nhid=32, nclass=6, dropout=0.1)
            model = models.GCN(input_dim=3703, hidden_dim=16, num_classes=6, p=0.5)

        self.device = device
        self.global_model = model.to(self.device)
        #self.global_model = model

        self.round = 0

        self.similarity_matrix = np.identity(self.conf["client_num"])
        #print(self.similarity_matrix)
        self.cluster_models = {}
        for cluster_id in range(self.conf["cluster_centers"]):
            self.cluster_models[cluster_id] = copy.deepcopy(self.global_model)

        self.clusters = [np.array(list(range(self.conf["client_num"])))]
        self.clients_cluster_map = {}
        for id in range(self.conf["client_num"]):
            self.clients_cluster_map[id] = id % self.conf["cluster_centers"]


    def model_update_aggregate(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():
            #update_per_layer = weight_accumulator[name] * self.conf["lambda"]
            update_per_layer = weight_accumulator[name]
            if self.conf['dp']:
                sigma = self.conf['sigma']
                #为梯度添加高斯噪声
                if torch.cuda.is_available():
                    noise = torch.cuda.FloatTensor(update_per_layer.shape).normal_(0, sigma)
                else:
                    noise = torch.FloatTensor(update_per_layer.shape).normal_(0, sigma)
                #print(update_per_layer)
                update_per_layer.add_(noise)

            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def model_weight_aggregate(self, models, client_datalen=None, total_len=None, ids=None, dataset=None, w=None):

         fed_state_dict = collections.OrderedDict()

         # for key, param in self.global_model.state_dict().items():
         #    sum = torch.zeros_like(param)
         #    for model in models:
         #         sum.add_(model.state_dict()[key].clone().to(self.device))
         #    sum = torch.div(sum, len(models))
         #    fed_state_dict[key] = sum

         if w is None:
             client_datalen = np.array(client_datalen)
             client_datalen = client_datalen / total_len
             ids_len = client_datalen[ids]
             ids_len /= np.sum(ids_len)
             weights = ids_len
         else:
             weights = w

         if self.conf["defense"] == "krum":
             #selected_models = krum(models, ids, num_malicious=2)

             selected_models = multi_krum(models, ids, num_malicious=int(self.conf["client_num"] * self.conf["malicious_ratio"]), multi=3)
             for key, param in self.global_model.state_dict().items():
                sum = torch.zeros_like(param)
                for i in range(len(selected_models)):
                    model = selected_models[i]
                    sum.add_(model.state_dict()[key].clone().to(self.device) * (1.0 / len(selected_models)))

                fed_state_dict[key] = sum
             self.global_model.load_state_dict(fed_state_dict)

         elif self.conf["defense"] == "trimmed":
             fed_state_dict = trimmed_mean(models)
             self.global_model.load_state_dict(fed_state_dict)

         elif self.conf["defense"] == "median":
             fed_state_dict = median(models)
             self.global_model.load_state_dict(fed_state_dict)

         elif self.conf["defense"] == "bulyan":
             fed_state_dict = bulyan(models, ids, num_malicious=int(self.conf["client_num"] * self.conf["malicious_ratio"]), num_to_agg=3)
             self.global_model.load_state_dict(fed_state_dict)

         else:
             for key, param in self.global_model.state_dict().items():
                sum = torch.zeros_like(param)
                for i in range(len(ids)):
                    model =models[i]
                    #sum.add_(model.state_dict()[key].clone().to(self.device) * ids_len[i])
                    sum.add_(model.state_dict()[key].clone().to(self.device) * weights[i])

                fed_state_dict[key] = sum
             self.global_model.load_state_dict(fed_state_dict)

         if dataset is None:
             acc, loss = self.model_eval()
         else:
             acc, loss = self.model_eval_graph(dataset)

         return acc, loss

    def cluster_aggregate(self, clients, client_datalen, total_len, dataset=None):
        cluster_datalen = [0, 0, 0]
        for i in range(len(clients)):
            index = i % 3
            cluster_datalen[index] += client_datalen[i]

        clients_id = [c.client_id for c in clients]
        fed_state_dict = collections.OrderedDict()
        dict_list = []

        for index, cluster in enumerate(self.clusters):
            if len(cluster) == 0:
                continue

            cluster_state_dict = collections.OrderedDict()

            for key, param in self.global_model.state_dict().items():
                sum = torch.zeros_like(param)
                for id in cluster:
                    if id in clients_id:
                        sum.add_(clients[id].local_model.state_dict()[key].clone().to(self.device) * (
                                    client_datalen[id] * 1.0 / cluster_datalen[index]))

                # sum = torch.div(sum, len(cluster))

                cluster_state_dict[key] = sum

            dict_list.append(cluster_state_dict)

        for key, param in self.global_model.state_dict().items():
            sum = torch.zeros_like(param)
            for i in range(len(dict_list)):
                dict = dict_list[i]
                sum.add_(dict[key] * (cluster_datalen[i] * 1.0 / total_len))
            # fed_state_dict[key] = torch.div(sum, len(dict_list))
            fed_state_dict[key] = sum

        self.global_model.load_state_dict(fed_state_dict)
        if dataset is None:
            acc, loss = self.model_eval()
        else:
            acc, loss = self.model_eval_graph(dataset)
        print("global model acc is {}".format(acc))
        return acc, loss

    def cluster_model_weight_aggregate(self, updates, dataset=None):

        models = {}
        ids = []
        for update in updates:
            id, model, similarity_vector = update
            ids.append(id)
            models[id] = model
            if similarity_vector is not None:
                self.similarity_matrix[id] = similarity_vector
                for i in range(self.similarity_matrix.shape[0]):
                    self.similarity_matrix[i][id] = similarity_vector[i]

        cluster_labels = AgglomerativeClustering(n_clusters=self.conf["cluster_centers"],
                                            affinity='precomputed', linkage='complete').fit_predict(self.similarity_matrix)

        clusters = []
        for i in range(self.conf["cluster_centers"]):
            cluster = np.where(cluster_labels == i)[0]
            clusters.append(cluster)
            for id in cluster:
                self.clients_cluster_map[id] = i
        print("{}".format(clusters))

        best_acc = 0
        best_loss = 0
        fed_state_dict = collections.OrderedDict()
        dict_list = []
        for index, cluster in enumerate(clusters):
            cluster_state_dict = collections.OrderedDict()
            update_ids = np.intersect1d(ids, cluster)
            if len(update_ids) != 0:
                for key, param in self.global_model.state_dict().items():
                    sum = torch.zeros_like(param)
                    for update_id in update_ids:
                        sum.add_(models[update_id].state_dict()[key].clone().to(self.device))
                    sum = torch.div(sum, len(update_ids))
                    cluster_state_dict[key] = sum

                dict_list.append(cluster_state_dict)
                self.cluster_models[index].load_state_dict(cluster_state_dict)

            if dataset is None:
                acc, loss = self.model_eval(model_for_test=self.cluster_models[index])
            else:
                acc, loss = self.model_eval_graph(dataset)

            if acc > best_acc:
                best_acc = acc
                best_loss = loss

        for key, param in self.global_model.state_dict().items():
            sum = torch.zeros_like(param)
            for dict in dict_list:
                sum.add_(dict[key])
            fed_state_dict[key] = torch.div(sum, len(dict_list))

        for index, cluster in enumerate(clusters):
            if len(cluster) == 0:
                self.cluster_models[index].load_state_dict(fed_state_dict)
        self.global_model.load_state_dict(fed_state_dict)
        best_acc, best_loss = self.model_eval()

        print("global model acc is {}".format(best_acc))
        return best_acc, best_loss

        # clusters = []
        # for i in range(self.conf["cluster_centers"]):
        #     clusters.append(np.intersect1d(np.where(cluster_labels == i), ids))
        #
        # for index, cluster in enumerate(clusters):
        #     for id in cluster:
        #         self.clients_cluster_map[id] = index
        # print("{} \n {}".format(cluster_labels, clusters))
        #
        # fed_state_dict = collections.OrderedDict()
        # dict_list = []
        # for index, cluster in enumerate(clusters):
        #     if len(cluster) == 0:
        #         continue
        #
        #     cluster_state_dict = collections.OrderedDict()
        #     for key, param in self.global_model.state_dict().items():
        #         sum = torch.zeros_like(param)
        #         for id in cluster:
        #             sum.add_(models[id].state_dict()[key].clone().to(self.device))
        #             self.clients_cluster_map[id] = index
        #         sum = torch.div(sum, len(cluster))
        #
        #         cluster_state_dict[key] = sum
        #
        #     dict_list.append(cluster_state_dict)
        #     self.cluster_models[index].load_state_dict(cluster_state_dict)
        #
        #     self.global_model.load_state_dict(cluster_state_dict)
        #     acc, _ = self.model_eval()
        #     print("cluster model acc is {}".format(acc))
        #
        # for key, param in self.global_model.state_dict().items():
        #     sum = torch.zeros_like(param)
        #     for dict in dict_list:
        #         sum.add_(dict[key])
        #     fed_state_dict[key] = torch.div(sum, len(dict_list))
        #
        # for index, cluster in enumerate(clusters):
        #     if len(cluster) == 0:
        #         self.cluster_models[index].load_state_dict(fed_state_dict)
        #
        # self.global_model.load_state_dict(fed_state_dict)
        # acc, loss = self.model_eval()
        # print("global model acc is {}".format(acc))
        # return acc, loss

        # print("{}".format(self.clusters))
        #
        # fed_state_dict = collections.OrderedDict()
        # dict_list = []
        # global_acc = 0.0
        # global_loss = 0.0
        # global_index = 0
        # for index, cluster in enumerate(self.clusters):
        #     cluster_ids = list(set(ids).intersection(set(cluster)))
        #     if len(cluster_ids) == 0:
        #         continue
        #     print(cluster_ids)
        #
        #     cluster_state_dict = collections.OrderedDict()
        #     for key, param in self.global_model.state_dict().items():
        #         sum = torch.zeros_like(param)
        #         for id in cluster_ids:
        #             sum.add_(models[id].state_dict()[key].clone().to(self.device))
        #         sum = torch.div(sum, len(cluster_ids))
        #         cluster_state_dict[key] = sum
        #
        #     dict_list.append(cluster_state_dict)
        #     self.cluster_models[index].load_state_dict(cluster_state_dict)
        #
        #     self.global_model.load_state_dict(cluster_state_dict)
        #     acc, loss = self.model_eval()
        #     print("cluster model acc is {}".format(acc))
        #     if acc > global_acc:
        #         global_acc = acc
        #         global_loss = loss
        #         global_index = index
        #
        # if self.round % 1 == 0:
        #     print("Changing the clusters of clients")
        #     cluster_labels = SpectralClustering(n_clusters=self.conf["cluster_centers"], affinity='precomputed').fit_predict(self.similarity_matrix)
        #     self.clusters = []
        #     for i in range(self.conf["cluster_centers"]):
        #         self.clusters.append(np.intersect1d(np.where(cluster_labels == i), list(range(self.conf["client_num"]))))
        #     for index, cluster in enumerate(self.clusters):
        #         for id in cluster:
        #             self.clients_cluster_map[id] = index
        #
        #     for cluster_id in range(self.conf["cluster_centers"]):
        #         self.cluster_models[cluster_id].load_state_dict(self.cluster_models[global_index].state_dict())
        # self.round += 1
        # return global_acc, global_loss


    def async_model_weight_aggregate(self, model, local_model_round, dataset=None, weight=0):
         self.round += 1

         print("Server start {}-th round avg with the old {}-th local model".format(self.round, local_model_round))
         # if local_model_round < self.round - 2:
         #     alpha = float(1/(self.round-2-local_model_round)**2)
         # else:
         #     alpha = 1.0
         if weight != 0:
            alpha = weight
         else:
            alpha = float((self.round - local_model_round + 1)**(-0.9))
         print("the local model weight is {}".format(alpha))

         fed_state_dict = collections.OrderedDict()

         for key, param in self.global_model.state_dict().items():
            sum = self.global_model.state_dict()[key].clone().to(self.device)
            sum.add_(torch.mul(model.state_dict()[key].clone().to(self.device), alpha))
            sum = torch.div(sum, (1+alpha))
            fed_state_dict[key] = sum

         self.global_model.load_state_dict(fed_state_dict)

         if dataset is None:
             acc, loss = self.model_eval()
         else:
             acc, loss = self.model_eval_graph(dataset)

         return acc, loss

    def model_eval(self, model_for_test=None):
        if self.eval_dataset is None:
            return None, None

        if model_for_test is None:
            test_model = self.global_model.eval()
        else:
            test_model = model_for_test.eval()
            test_model.to(self.device)

        total_loss = 0.0
        correct = 0
        dataset_size = 0

        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]
            data = data.to(self.device)
            target = target.to(self.device, dtype=torch.long)
            output = test_model(data)

            # else:
            #     ids, masks, target = batch
            #     dataset_size += target.size()[0]
            #     ids = ids.to(self.device)
            #     masks = masks.to(self.device)
            #     target = target.to(self.device, dtype=torch.long)
            #     output, _ = test_model(ids, token_type_ids=None, attention_mask=masks, labels=target)

            total_loss += torch.nn.functional.cross_entropy(output, target,
                                                            reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        print("finish testing model, acc:{}, total_loss:{}".format(acc, total_l))
        return acc, total_l

    def model_eval_graph(self, dataset, model_for_test=None):
        def accuracy(output, labels):
            preds = output.max(1)[1].type_as(labels)
            correct = preds.eq(labels).double()
            correct = correct.sum()
            return correct / len(labels)

        if model_for_test is None:
            test_model = self.global_model.eval()
        else:
            test_model = model_for_test.eval()
            test_model.to(self.device)

        dataset.cora_dataset.labels = dataset.cora_dataset.labels.to(self.device, dtype=torch.long)
        dataset.cora_dataset.features = dataset.cora_dataset.features.to(self.device)
        dataset.cora_dataset.adj = dataset.cora_dataset.adj.to(self.device)
        output = test_model(dataset.cora_dataset.features, dataset.cora_dataset.adj)
        loss_val = torch.nn.functional.cross_entropy(output[dataset.cora_dataset.idx_val],
                               dataset.cora_dataset.labels[dataset.cora_dataset.idx_val])
        acc_val = accuracy(output[dataset.cora_dataset.idx_val], dataset.cora_dataset.labels[dataset.cora_dataset.idx_val])
        print("Valid loss {}. Valid accuracy {}".format(loss_val, acc_val))
        return acc_val, loss_val

        # dataset.cora_dataset.features = dataset.cora_dataset.features.to(self.device)
        # dataset.cora_dataset.y_val = dataset.cora_dataset.y_val.to(self.device)
        # dataset.cora_dataset.idx_val = dataset.cora_dataset.idx_val.to(self.device)
        #
        # output = test_model(dataset.cora_dataset.features, dataset.cora_dataset.adj)
        # # loss_val = F.nll_loss(output[dataset.idx_test], dataset.labels[dataset.idx_test])
        # loss_val = models.get_loss(output, dataset.cora_dataset.y_val, dataset.cora_dataset.idx_val)
        # acc_val = models.get_accuracy(output, dataset.cora_dataset.y_val, dataset.cora_dataset.idx_val)
        # print("Valid loss {}. Valid accuracy {}".format(loss_val, acc_val))
        # return acc_val, loss_val

    def save_model(self, dir='./models/server/', client_id=-1, model_name="model"):
        #path = os.path.join(dir, 'model.pth')
        path = os.path.join(dir, model_name + '.pth')
        print("Save global model to {}".format(path))
        #torch.save(self.global_model.state_dict(), path)
        torch.save(self.global_model, path)


    def scalesfl_cluster_model_weight_aggregate(self, clients, client_datalen, total_len, dataset=None):
        cluster_datalen = [0.0, 0.0, 0.0]
        for i in range(len(clients)):
            index = clients[i].client_id%3
            cluster_datalen[index] += client_datalen[i]

        clients_id = [c.client_id for c in clients]
        fed_state_dict = collections.OrderedDict()
        dict_list = {}

        for index, cluster in enumerate(self.clusters):

            ids = list(set(clients_id).intersection(set(cluster)))
            if len(ids) == 0:
                continue
            data_len = np.array(client_datalen)[ids]
            data_len = data_len * 1.0 / np.sum(data_len)

            cluster_state_dict = collections.OrderedDict()
            for key, param in self.global_model.state_dict().items():
                sum = torch.zeros_like(param)
                for i in range(len(ids)):
                    id = ids[i]
                    weight = data_len[i]
                    for client in clients:
                        if client.client_id == id:
                            sum.add_(client.local_model.state_dict()[key].clone().to(self.device) * weight)
                cluster_state_dict[key] = sum
            dict_list[index] = cluster_state_dict


        for key, param in self.global_model.state_dict().items():
            sum = torch.zeros_like(param)

            for index, _ in enumerate(self.clusters):
                if index in dict_list:
                    dict = dict_list[index]
                    sum.add_(dict[key]*(cluster_datalen[index]/np.sum(cluster_datalen)))
            fed_state_dict[key] = sum

        self.global_model.load_state_dict(fed_state_dict)
        if dataset is None:
            acc, loss = self.model_eval()
        else:
            acc, loss = self.model_eval_graph(dataset)
        print("global model acc is {}".format(acc))
        return acc, loss

    def casfl_cluster_model_weight_aggregate(self, clients, client_datalen, total_len, dataset=None):

        models = {c.client_id:c.local_model for c in clients}
        ids = np.array([c.client_id for c in clients])

        avg_acc = 0
        avg_loss = 0
        cnt = 0
        for index, cluster in enumerate(self.clusters):
            cluster_state_dict = collections.OrderedDict()
            update_ids = np.intersect1d(ids, cluster)
            if len(update_ids) != 0:
                for key, param in self.global_model.state_dict().items():
                    sum = torch.zeros_like(param)
                    for update_id in update_ids:
                        sum.add_(models[update_id].state_dict()[key].clone().to(self.device))
                    sum = torch.div(sum, len(update_ids))
                    cluster_state_dict[key] = sum
                self.cluster_models[index].load_state_dict(cluster_state_dict)

            if dataset is None:
                acc, loss = self.model_eval(model_for_test=self.cluster_models[index])
            else:
                acc, loss = self.model_eval_graph(dataset, model_for_test=self.cluster_models[index])

            avg_acc += acc
            avg_loss += loss
            cnt += 1

        avg_acc /= cnt
        avg_loss /= cnt
        return avg_acc, avg_loss

        # cluster_datalen = [0.0] * self.conf["cluster_centers"]
        # for i in range(len(clients)):
        #     index = clients[i].client_id%3
        #     cluster_datalen[index] += client_datalen[i]
        #
        # clients_id = [c.client_id for c in clients]
        # fed_state_dict = collections.OrderedDict()
        # dict_list = {}
        #
        # for index, cluster in enumerate(self.clusters):
        #
        #     ids = list(set(clients_id).intersection(set(cluster)))
        #     if len(ids) == 0:
        #         continue
        #     data_len = np.array(client_datalen)[ids]
        #     data_len = data_len * 1.0 / np.sum(data_len)
        #
        #     cluster_state_dict = collections.OrderedDict()
        #     for key, param in self.global_model.state_dict().items():
        #         sum = torch.zeros_like(param)
        #         for i in range(len(ids)):
        #             id = ids[i]
        #             weight = data_len[i]
        #             for client in clients:
        #                 if client.client_id == id:
        #                     sum.add_(client.local_model.state_dict()[key].clone().to(self.device) * weight)
        #         cluster_state_dict[key] = sum
        #     dict_list[index] = cluster_state_dict
        #
        # acc_avg = 0.0
        # loss_avg = 0.0
        # cnt = 0
        # for cluster_id in range(self.conf["cluster_centers"]):
        #     if cluster_id in dict_list:
        #         cnt += 1
        #
        #         self.cluster_models[cluster_id].load_state_dict(cluster_state_dict)
        #
        #         if dataset is None:
        #             acc, loss = self.model_eval(model_for_test=self.cluster_models[cluster_id])
        #         else:
        #             acc, loss = self.model_eval(model_for_test=self.cluster_models[cluster_id], dataset=dataset)
        #
        #         acc_avg += acc
        #         loss_avg += loss
        #
        # acc_avg /= cnt
        # loss_avg /= cnt
        # return acc_avg, loss_avg

    def best_cluster_model_weight_aggregate(self, clients, client_datalen, total_len, dataset=None):
        cluster_datalen = [0.0, 0.0, 0.0]
        for i in range(len(clients)):
            index = i%3
            cluster_datalen[index] += client_datalen[i]

        clients_id = [c.client_id for c in clients]
        fed_state_dict = collections.OrderedDict()
        dict_list = []

        for index, cluster in enumerate(self.clusters):
            if len(cluster) == 0:
                continue

            cluster_state_dict = collections.OrderedDict()

            for key, param in self.global_model.state_dict().items():
                sum = torch.zeros_like(param)
                for id in cluster:
                    if id in clients_id:
                       sum.add_(clients[id].local_model.state_dict()[key].clone().to(self.device)*(client_datalen[id] * 1.0 / cluster_datalen[index]))
                cluster_state_dict[key] = sum

            dict_list.append(cluster_state_dict)

        for key, param in self.global_model.state_dict().items():
            sum = torch.zeros_like(param)
            for i in range(len(dict_list)):
                dict = dict_list[i]
                #sum.add_(dict[key]*(cluster_datalen[i]*1.0/total_len))
                sum.add_(dict[key] * (cluster_datalen[i]/ np.sum(cluster_datalen)))
            fed_state_dict[key] = sum

        self.global_model.load_state_dict(fed_state_dict)

        if dataset is None:
            acc, loss = self.model_eval()
        else:
            acc, loss = self.model_eval_graph(dataset)
        print("global model acc is {}".format(acc))
        return acc, loss