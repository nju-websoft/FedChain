import os.path
import collections
import sys

import torch, copy
import numpy as np
import dataset
from server import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import models

import requests
import schedule
import time
import json
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from sparsity_extract import *
from attack.lie_attack import *

np.random.seed(33)

class Client(object):

    def __init__(self, conf, train_dataset, test_dataset=None, id=-1, global_model=None, device=None, is_malicious=False, backdoor_dataset=None):

        self.conf = conf

        #model = models.get_model(conf["model_name"], load_from_local=True)
        if self.conf["dataset"] == "cifar10":
            model = models.VGGCifar(num_classes=10)
        if self.conf["dataset"] == "cifar100":
            model = models.VGGCifar(num_classes=100)
        if self.conf["dataset"] == "mnist" or self.conf["dataset"] == "fashion_mnist":
            model = models.VGGMNIST()
        if self.conf["dataset"] == "imagenet":
            model = models.VGGCifar(num_classes=200)
        if self.conf["dataset"] == "agnews":
            model = models.BertClassifer.from_pretrained('./models/bert-base-uncased/', num_labels=4)
        if self.conf["dataset"] == "shakespeare":
            model = models.CharLSTM()
        if self.conf["dataset"] == "sent140":
            model = models.RNN()
        if self.conf["dataset"] == "cora":
            #model = models.GCN(nfeat=1433, nhid=32, nclass=7, dropout=0.1)
            model = models.GCN(input_dim=1433, hidden_dim=16, num_classes=7, p=0.5)
        if self.conf["dataset"] == "citeseer":
            #model = models.GCN(nfeat=3703, nhid=32, nclass=6, dropout=0.1)
            model = models.GCN(input_dim=3703, hidden_dim=16, num_classes=6, p=0.5)

        #if torch.cuda.is_available():
            #self.local_model = model.cuda()
        self.device = device

        if global_model:
            print("Initialize parameters from global model")
            model.load_state_dict(global_model.state_dict())

        self.local_model = model.to(self.device)

        self.client_id = id

        self.round = 0

        self.train_dataset = train_dataset
        if train_dataset is not None:
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],
                                                        shuffle=True)

        self.test_dataset = test_dataset
        if test_dataset is not None:
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.conf["batch_size"],
                                                        shuffle=False)

        self.extractor = None
        self.time_list = []
        if self.conf["exchange"]:
            if self.conf["dataset"] == "shakespeare":
                 self.extractor = NLPFeatureExtractor(model=self.local_model, dataset_name=self.conf["dataset"])
            elif self.conf["dataset"] == "cora" or self.conf["dataset"] == "citeseer":
                 self.extractor = GCNFeatureExtractor(model=self.local_model, dataset_name=self.conf["dataset"])
            else:
                 self.extractor = FeatureExtractor(model=self.local_model, dataset_name=self.conf["dataset"])

            self.loss_list = [1000 for _ in range(self.conf["client_num"])]

        # map = {}
        # for batch_id, batch in enumerate(self.train_loader):
        #     data, target = batch
        #     for t in target:
        #         if t.item() in map:
        #             map[t.item()] += 1
        #         else:
        #             map[t.item()] = 1
        # print(map)

        self.prev_local_model = self.local_model

        self.is_malicious = is_malicious
        self.backdoor_dataset = backdoor_dataset
        if backdoor_dataset is not None:
            self.backdoor_train_loader = torch.utils.data.DataLoader(self.backdoor_dataset, batch_size=conf["batch_size"],
                                                        shuffle=True)

    def local_train(self, model, model_round=None):

        if model_round is not None:
            self.round = model_round + 1
            print("Client {} pull the {}-th round model for training".format(self.client_id, model_round))

        # for name, param in model.state_dict().items():
        #     self.local_model.state_dict()[name].copy_(param.clone())
        self.local_model.load_state_dict(copy.deepcopy(model.state_dict()))

        if not self.extractor is None:
            self.extractor.update_model(self.local_model)
            self.extractor.mean_list = np.zeros((1, self.extractor.num_channels))

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])

        #graph
        #optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'])

        criterion = torch.nn.CrossEntropyLoss()

        epoch = random.randint(self.conf["min_local_epochs"], self.conf["max_local_epochs"])

        self.local_model.to(self.device)

        #self.extractor.mean_list = np.zeros((1, self.extractor.num_channels))

        start = time.time()


        for e in range(epoch):
            self.local_model.train()
            total_loss = 0.0

            for batch_id, batch in enumerate(tqdm(self.train_loader, file=sys.stdout)):
            #for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                data = data.to(self.device)
                target = target.to(self.device, dtype=torch.long)
                optimizer.zero_grad()
                output = self.local_model(data)

                loss = criterion(output, target)

                if self.conf["prox"] == True:
                    proximal_term = 0.0
                    for w, w_t in zip(self.local_model.parameters(), model.parameters()):
                        proximal_term += (w - w_t.clone().to(self.device)).norm(2)
                    #print("{},{}".format(loss, proximal_term))
                    loss = loss + (0.1 / 2) * proximal_term

                loss.backward(retain_graph=True)
                total_loss += loss.item()

                if self.conf['dp'] == True:
                    torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0, norm_type=2)

                optimizer.step()

                if e == epoch - 1 and not self.extractor is None:
                    with torch.no_grad():
                        self.extractor(data)

            end = time.time()
            acc, total_l = self.eval_model()
            print("Epoch {} done. Train loss {}. Valid accuracy {}, Consume: {}".format(e, total_loss, acc, end-start))

        print("client {} finish local train".format(self.client_id))

        if not self.extractor is None:

           self.extractor.old_all_vectors = self.extractor.mean_list.copy()
           temp = self.extractor.mean_list.mean(axis=0)
           self.extractor.mean_list = temp.reshape((1, self.extractor.num_channels))

           self.time_list.append(end-start)
           #print(self.extractor.old_all_vectors.shape)
        return acc, total_l


    def fuse_model_by_teachers(self, models, model_round=None):
        if model_round is not None:
            self.round = model_round + 1
            print("Client {} pull the {}-th round model for collaborative training".format(self.client_id, model_round))

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['distillation_lr'], momentum=self.conf['momentum'])

        #optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['distillation_lr']) # shakespeare数据集专用

        criterion1 = torch.nn.CrossEntropyLoss()
        criterion2 = torch.nn.KLDivLoss(reduction="batchmean")

        for i in range(len(models)):
            #models[i][1] = models[i][1].eval()
            models[i][1] = models[i][1].train()
            models[i][1] = models[i][1].to(self.device)
            self.loss_list[models[i][0]] = 0.0

        count = 0
        epochs = self.conf["exchange_local_epochs"]
        for e in range(epochs):
            self.local_model.train()
            total_loss = 0.0

            for batch_id, batch in enumerate(tqdm(self.train_loader, file=sys.stdout)):
            #for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                data = data.to(self.device)
                target = target.to(self.device, dtype=torch.long)
                optimizer.zero_grad()
                student_output = self.local_model(data)

                if self.conf["dataset"] == "agnews":
                    T = 1
                    alpha = 0.2
                    beta = 0.8

                elif self.conf["dataset"] == "imagenet":
                    T = 2
                    alpha = 0.5
                    beta = 0.5

                elif self.conf["dataset"] == "fashion_mnist":
                    T = 1
                    alpha = 0.7
                    beta = 0.3

                elif self.conf["dataset"] == "cifar10":
                    T = 1
                    alpha = 0.55
                    beta = 0.45

                elif self.conf["dataset"] == "cifar100":
                    T = 1
                    alpha = 0.4
                    beta = 0.6

                elif self.conf["dataset"] == "shakespeare":
                    T = 1
                    alpha = 0.3
                    beta = 0.7

                teacher_output = torch.zeros_like(student_output)
                weight = 1 / len(models)
                with torch.no_grad():
                    for _, model in models:
                        teacher_output += weight * model(data)
                loss1 = criterion1(student_output, target)
                loss2 = criterion2(F.log_softmax(student_output / T, dim=1),
                                   F.softmax(teacher_output / T, dim=1)) * T * T

                # loss1 = criterion1(student_output, target)
                # loss2 = torch.tensor(0, dtype=torch.float, device=self.device)
                # for id, model in models:
                #     kl_loss = criterion2(F.log_softmax(student_output/T, dim=1), F.softmax(model(data)/T, dim=1)) * T * T
                #     loss2 += kl_loss
                #
                #     if e == epochs - 1:
                #         self.loss_list[id] += kl_loss.item()

                if e == 1 and count < 2:
                    print("{}, {}, {}".format(T, alpha, beta))
                    print("loss1:{}, alpha*loss1:{}; loss2:{}, beta*loss2:{}".format(loss1, alpha * loss1, loss2, beta*loss2))
                    count += 1

                loss = alpha * loss1 + beta * loss2
                loss.backward(retain_graph=True)
                total_loss += loss.item()

                if self.conf['dp'] == True:
                    torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0, norm_type=2)

                optimizer.step()

            acc, total_l = self.eval_model()
            print("Epoch {} done. Train loss {}. Valid accuracy {}".format(e, total_loss, acc))
        #torch.cuda.empty_cache()
        return acc, total_l

    def eval_model(self):
        if self.test_dataset is None:
            return None, None

        self.local_model.eval()
        # self.extractor.update_model(self.local_model)
        # self.extractor.mean_list = np.zeros((1, self.extractor.num_channels))

        total_loss = 0.0
        correct = 0
        dataset_size = 0

        count = 0

        for batch_id, batch in enumerate(tqdm(self.test_loader, file=sys.stdout)):
        #for batch_id, batch in enumerate(self.test_loader):
            data, target = batch
            dataset_size += data.size()[0]
            data = data.to(self.device)
            target = target.to(self.device, dtype=torch.long)
            output = self.local_model(data)
            # self.extractor(data)

            total_loss += torch.nn.functional.cross_entropy(output, target,
                                                            reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability

            if count < 0:
                print(pred)
                count += 1

            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        #print("Valid dataset acc: {}, total_l: {}".format(acc, total_l))

        # temp = self.extractor.mean_list.mean(axis=0)
        # self.extractor.mean_list = temp.reshape((1, self.extractor.num_channels))
        return acc, total_l

    def eval_model_graph(self, dataset):
        def accuracy(output, labels):
            preds = output.max(1)[1].type_as(labels)
            correct = preds.eq(labels).double()
            correct = correct.sum()
            return correct / len(labels)

        self.local_model.eval()

        # dataset.cora_dataset.features = dataset.cora_dataset.features.to(self.device)
        # # dataset.adj = dataset.adj.to(self.device)
        # #dataset.labels = dataset.labels.to(self.device, dtype=torch.long)
        # dataset.cora_dataset.y_val = dataset.cora_dataset.y_val.to(self.device)
        # dataset.cora_dataset.idx_val = dataset.cora_dataset.idx_val.to(self.device)
        #
        # output = self.local_model(dataset.cora_dataset.features, dataset.cora_dataset.adj)
        # # loss_val = F.nll_loss(output[dataset.idx_test], dataset.labels[dataset.idx_test])
        # # acc_val = dataset.accuracy(output[dataset.idx_test], dataset.labels[dataset.idx_test])
        # loss_val = models.get_loss(output, dataset.cora_dataset.y_val, dataset.cora_dataset.idx_val)
        # acc_val = models.get_accuracy(output, dataset.cora_dataset.y_val, dataset.cora_dataset.idx_val)
        # print("client {}: Valid loss {}. Valid accuracy {}".format(self.client_id, loss_val, acc_val))


        dataset.cora_dataset.labels = dataset.cora_dataset.labels.to(self.device, dtype=torch.long)
        dataset.cora_dataset.features = dataset.cora_dataset.features.to(self.device)
        dataset.cora_dataset.adj = dataset.cora_dataset.adj.to(self.device)
        output = self.local_model(dataset.cora_dataset.features, dataset.cora_dataset.adj)
        loss_val = torch.nn.functional.cross_entropy(output[dataset.cora_dataset.idx_val],
                               dataset.cora_dataset.labels[dataset.cora_dataset.idx_val])
        acc_val = accuracy(output[dataset.cora_dataset.idx_val], dataset.cora_dataset.labels[dataset.cora_dataset.idx_val])
        print("client {}: Valid loss {}. Valid accuracy {}".format(self.client_id, loss_val, acc_val))
        return acc_val, loss_val

    def save_model(self, dir='./models/clients/', client_id=-1, model_name="model"):
        if client_id!=-1:
            path = os.path.join(dir, 'client' + str(client_id), model_name + '.pth')
        else:
            path = os.path.join(dir, 'client'+str(self.client_id), model_name + '.pth')
        print("Save model to {}".format(path))
        torch.save(self.local_model, path)

    def load_model(self, path):
        print("Load model from {}".format(path))
        #self.local_model.load_state_dict(torch.load(path))
        #self.local_model = torch.load(path, map_location=self.device)
        self.local_model = torch.load(path, map_location=self.device)

    def local_train_graph(self, model, dataset, model_round=None):

        if model_round is not None:
            self.round = model_round + 1
            print("Client {} pull the {}-th round model for training".format(self.client_id, model_round))

        # for name, param in model.state_dict().items():
        #     self.local_model.state_dict()[name].copy_(param.clone())
        self.local_model.load_state_dict(model.state_dict())

        if not self.extractor is None:
            self.extractor.update_model(self.local_model)

        cora_dataset = dataset.cora_dataset
        # client_idcs = dataset.client_idcs
        # if len(client_idcs[self.client_id]) == 0:
        #     return
        #optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf["lr"])

        criterion = torch.nn.CrossEntropyLoss()

        epoch = random.randint(self.conf["min_local_epochs"], self.conf["max_local_epochs"])

        self.local_model.to(self.device)

        dataset.cora_dataset.labels = dataset.cora_dataset.labels.to(self.device, dtype=torch.long)
        dataset.cora_dataset.features = dataset.cora_dataset.features.to(self.device)
        dataset.cora_dataset.adj = dataset.cora_dataset.adj.to(self.device)
        for e in range(epoch):
            self.local_model.train()
            optimizer.zero_grad()
            output = self.local_model(dataset.cora_dataset.features, dataset.cora_dataset.adj)
            #loss_train = criterion(output[dataset.cora_dataset.idx_train], dataset.cora_dataset.labels[dataset.cora_dataset.idx_train])

            mask = dataset.cora_dataset.idx_train[dataset.client_idcs[self.client_id]]
            loss_train = criterion(output[mask], dataset.cora_dataset.labels[mask])
            loss_train.backward()
            optimizer.step()

            acc_val, loss_val = self.eval_model_graph(dataset=dataset)
        return acc_val, loss_val

        # # self.extractor.mean_list = np.zeros((1, self.extractor.num_channels))
        #
        # dataset.cora_dataset.features = dataset.cora_dataset.features.to(self.device)
        # dataset.cora_dataset.y_train = dataset.cora_dataset.y_train.to(self.device)
        # dataset.cora_dataset.idx_train = dataset.cora_dataset.idx_train.to(self.device)
        #
        # mask = torch.from_numpy(np.array(dataset.client_idcs[self.client_id]))
        # mask = mask.to(self.device)
        #
        # epoch = random.randint(self.conf["min_local_epochs"], self.conf["max_local_epochs"])
        #
        # for e in range(epoch):
        #     self.local_model.train()
        #
        #     optimizer.zero_grad()
        #     output = self.local_model(dataset.cora_dataset.features, dataset.cora_dataset.adj)
        #     #loss_train = torch.nn.functional.cross_entropy(output, cora_dataset.labels, reduction="sum")
        #     #loss_train = F.nll_loss(output[client_idcs[self.client_id]], cora_dataset.labels[client_idcs[self.client_id]])
        #
        #     #loss_train = models.get_loss(output, dataset.cora_dataset.y_train, dataset.cora_dataset.idx_train)
        #     loss_train = models.get_loss(output, dataset.cora_dataset.y_train, mask)
        #     loss_train.backward()
        #     optimizer.step()
        #
        #     # if e == epoch - 1 and not self.extractor is None:
        #     #     with torch.no_grad():
        #     #         self.extractor(cora_dataset.features, cora_dataset.adj)
        #
        # acc_val, loss_val = self.eval_model_graph(dataset=dataset)
        #
        # return acc_val, loss_val

    def fuse_model_by_teachers_graph(self, clients, dataset, model_round=None):
        if model_round is not None:
            self.round = model_round + 1
            print("Client {} pull the {}-th round model for collaborative training".format(self.client_id, model_round))

        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf["lr"])

        #optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['distillation_lr']) # shakespeare数据集专用

        criterion1 = torch.nn.CrossEntropyLoss()
        criterion2 = torch.nn.KLDivLoss(reduction="batchmean")

        for i in range(len(clients)):
            #models[i][1] = models[i][1].eval()
            #clients[i].local_model.eval()
            clients[i].local_model.to(self.device)
            self.loss_list[i] = 0.0

        self.local_model.to(self.device)

        dataset.cora_dataset.labels = dataset.cora_dataset.labels.to(self.device, dtype=torch.long)
        dataset.cora_dataset.features = dataset.cora_dataset.features.to(self.device)
        dataset.cora_dataset.adj = dataset.cora_dataset.adj.to(self.device)

        T = 1
        alpha = 0.7
        beta = 0.3

        count = 0
        epochs = self.conf["exchange_local_epochs"]
        for e in range(epochs):
            self.local_model.train()
            optimizer.zero_grad()

            student_output = self.local_model(dataset.cora_dataset.features, dataset.cora_dataset.adj)
            mask = dataset.cora_dataset.idx_train[dataset.client_idcs[self.client_id]]
            loss1 = criterion1(student_output[mask], dataset.cora_dataset.labels[mask])

            teacher_output = clients[0].local_model(dataset.cora_dataset.features, dataset.cora_dataset.adj)
            mask2 = dataset.cora_dataset.idx_train[dataset.client_idcs[clients[0].client_id]]
            loss2 = criterion2(F.log_softmax(student_output[mask], dim=1),
                               F.softmax(teacher_output[mask], dim=1))

            if e == 1 and count < 2:
                print("{}, {}, {}".format(T, alpha, beta))
                print("loss1:{}, alpha*loss1:{}; loss2:{}, beta*loss2:{}".format(loss1, alpha * loss1, loss2,
                                                                                 beta * loss2))
                count += 1

            loss = alpha * loss1 + beta * loss2
            loss.backward()
            optimizer.step()

            acc_val, loss_val = self.eval_model_graph(dataset=dataset)
        return acc_val, loss_val


    def moon_train(self, model, model_round=None):
        if model_round is not None:
           self.round = model_round + 1
           print("Client {} pull the {}-th round model for training".format(self.client_id, model_round))

        self.local_model.load_state_dict(copy.deepcopy(model.state_dict()))

        #optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])

        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'])

        criterion = torch.nn.CrossEntropyLoss()
        cos = torch.nn.CosineSimilarity(dim=-1)

        epoch = random.randint(self.conf["min_local_epochs"], self.conf["max_local_epochs"])

        self.local_model.to(self.device)

        model.train()
        self.prev_local_model.train()

        start = time.time()
        for e in range(epoch):
            self.local_model.train()
            total_loss = 0.0

            for batch_id, batch in enumerate(tqdm(self.train_loader, file=sys.stdout)):
            #for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                data = data.to(self.device)
                target = target.to(self.device, dtype=torch.long)
                optimizer.zero_grad()

                output = self.local_model(data)
                global_output = model(data)
                prev_output = self.prev_local_model(data)

                posi = cos(output, global_output)
                logits = posi.reshape(-1, 1)
                nega = cos(output, prev_output)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                loss1 = criterion(output, target)

                labels = torch.zeros(data.size(0)).cuda().long()
                loss2 = criterion(logits, labels)
                loss = loss1 + loss2

                loss.backward(retain_graph=True)
                total_loss += loss.item()

                if self.conf['dp'] == True:
                    torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0, norm_type=2)

                optimizer.step()

                if e == epoch - 1 and not self.extractor is None:
                    with torch.no_grad():
                        self.extractor(data)

            end = time.time()
            acc, total_l = self.eval_model()
            print("Epoch {} done. Train loss {}. Valid accuracy {}, Consume: {}".format(e, total_loss, acc, end-start))

        print("client {} finish local train".format(self.client_id))


        self.prev_local_model = self.local_model
        return acc, total_l

    def moon_train_graph(self, model, dataset, model_round=None):

        if model_round is not None:
            self.round = model_round + 1
            print("Client {} pull the {}-th round model for training".format(self.client_id, model_round))

        # for name, param in model.state_dict().items():
        #     self.local_model.state_dict()[name].copy_(param.clone())
        self.local_model.load_state_dict(model.state_dict())


        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf["lr"])

        criterion = torch.nn.CrossEntropyLoss()
        cos = torch.nn.CosineSimilarity(dim=-1)

        epoch = random.randint(self.conf["min_local_epochs"], self.conf["max_local_epochs"])

        self.local_model.to(self.device)
        model.train()
        self.prev_local_model.train()

        dataset.cora_dataset.labels = dataset.cora_dataset.labels.to(self.device, dtype=torch.long)
        dataset.cora_dataset.features = dataset.cora_dataset.features.to(self.device)
        dataset.cora_dataset.adj = dataset.cora_dataset.adj.to(self.device)
        for e in range(epoch):
            self.local_model.train()
            optimizer.zero_grad()
            output = self.local_model(dataset.cora_dataset.features, dataset.cora_dataset.adj)
            #loss_train = criterion(output[dataset.cora_dataset.idx_train], dataset.cora_dataset.labels[dataset.cora_dataset.idx_train])

            output = self.local_model(dataset.cora_dataset.features, dataset.cora_dataset.adj)
            global_output = model(dataset.cora_dataset.features, dataset.cora_dataset.adj)
            prev_output = self.prev_local_model(dataset.cora_dataset.features, dataset.cora_dataset.adj)

            posi = cos(output, global_output)
            logits = posi.reshape(-1, 1)
            nega = cos(output, prev_output)
            logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

            mask = dataset.cora_dataset.idx_train[dataset.client_idcs[self.client_id]]
            loss1 = criterion(output[mask], dataset.cora_dataset.labels[mask])

            labels = torch.zeros(logits.size(0)).cuda().long()
            loss2 = criterion(logits, labels)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            acc_val, loss_val = self.eval_model_graph(dataset=dataset)
        return acc_val, loss_val

    def backdoor_local_train(self, model, model_round=None):

        if model_round is not None:
            self.round = model_round + 1
            print("Client {} pull the {}-th round model for training".format(self.client_id, model_round))

        # for name, param in model.state_dict().items():
        #     self.local_model.state_dict()[name].copy_(param.clone())
        self.local_model.load_state_dict(copy.deepcopy(model.state_dict()))

        if not self.extractor is None:
            self.extractor.update_model(self.local_model)
            self.extractor.mean_list = np.zeros((1, self.extractor.num_channels))

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])

        #graph
        #optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'])

        criterion = torch.nn.CrossEntropyLoss()

        epoch = random.randint(self.conf["min_local_epochs"], self.conf["max_local_epochs"])

        self.local_model.to(self.device)

        #self.extractor.mean_list = np.zeros((1, self.extractor.num_channels))

        start = time.time()


        for e in range(epoch):
            self.local_model.train()
            total_loss = 0.0

            for batch_id, (normal_batch, backdoor_batch) in enumerate(tqdm(zip(self.train_loader, self.backdoor_train_loader), file=sys.stdout)):
            #for batch_id, batch in enumerate(tqdm(self.train_loader, file=sys.stdout)):
            #for batch_id, batch in enumerate(self.train_loader):
                normal_data, normal_target = normal_batch
                backdoor_data, backdoor_target = backdoor_batch

                normal_data = normal_data.to(self.device)
                normal_target = normal_target.to(self.device, dtype=torch.long)
                backdoor_data = backdoor_data.to(self.device)
                backdoor_target = backdoor_target.to(self.device, dtype=torch.long)

                optimizer.zero_grad()

                normal_output = self.local_model(normal_data)
                backdoor_output = self.local_model(backdoor_data)

                alpha = 0.7
                loss = alpha * criterion(normal_output, normal_target) + (1 - alpha) * criterion(backdoor_output, backdoor_target)

                if self.conf["prox"] == True:
                    proximal_term = 0.0
                    for w, w_t in zip(self.local_model.parameters(), model.parameters()):
                        proximal_term += (w - w_t.clone().to(self.device)).norm(2)
                    #print("{},{}".format(loss, proximal_term))
                    loss = loss + (0.1 / 2) * proximal_term

                loss.backward(retain_graph=True)
                total_loss += loss.item()

                if self.conf['dp'] == True:
                    torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0, norm_type=2)

                optimizer.step()

                if e == epoch - 1 and not self.extractor is None:
                    with torch.no_grad():
                        self.extractor(normal_data)

            end = time.time()
            acc, total_l = self.eval_model()
            print("Epoch {} done. Train loss {}. Valid accuracy {}, Consume: {}".format(e, total_loss, acc, end-start))

        print("client {} finish local train".format(self.client_id))

        if not self.extractor is None:

           self.extractor.old_all_vectors = self.extractor.mean_list.copy()
           temp = self.extractor.mean_list.mean(axis=0)
           self.extractor.mean_list = temp.reshape((1, self.extractor.num_channels))
           #print(self.extractor.old_all_vectors.shape)
        return acc, total_l

class ClientGroup(object):
    def __init__(self, conf, have_server=False):
        self.conf = conf
        self.test_data_loader=None
        self.clients= []

        self.have_server = have_server
        self.server = None

        self.dataset_allocation()


    def dataset_allocation(self):
        data = dataset.GetDataSet(dataset_name=self.conf["dataset"], is_iid=self.conf["iid"], beta=self.conf["niid_beta"])

        if self.conf["dataset"] == "cora":
            self.cora_data = data
            for i in range(self.conf["client_num"]):
                self.clients.append(Client(conf=self.conf, id=i, train_dataset=None, device=torch.device("cuda:"+str(0))))
            return


        self.test_data_loader = torch.utils.data.DataLoader(data.test_dataset, batch_size=self.conf["batch_size"], shuffle=False, num_workers=4)
        if self.have_server:
            self.server = Server(self.conf, data.test_dataset, device=torch.device(torch.device("cuda:"+str(0))))


        shard_size = data.train_data_size // self.conf["client_num"] // 2
        shard_id = np.random.permutation(data.train_data_size // shard_size)

        for i in range(self.conf["client_num"]):
            if self.conf["iid"] == False:
                subset = torch.utils.data.Subset(data.train_dataset, data.client_idcs[i])

            else:
                shards = list(range(shard_id[i] * shard_size, shard_id[i] * shard_size + shard_size))
                subset = torch.utils.data.Subset(data.train_dataset, shards)

                # labels = data.train_dataset.targets[shards]
                # distribution = {}
                # for label in labels:
                #     if label not in distribution:
                #         distribution[label] = 1
                #     else:
                #         distribution[label] += 1
                # print("Label Distribution : {}".format(distribution))
            self.clients.append(Client(conf=self.conf, train_dataset=subset, test_dataset=data.test_dataset, id=i, device=torch.device("cuda:"+str(0))))

if __name__ == "__main__":
    #torch.cuda.empty_cache()

    with open("./config/conf.json", 'r') as f:
        conf = json.load(f)

    client_group = ClientGroup(conf=conf, have_server=False)

    clients = client_group.clients
    attack(clients[:2])


    #server = client_group.server
    # client0 = client_group.clients[0]
    # client1 = client_group.clients[1]
    # client2 = client_group.clients[2]
    #client3 = client_group.clients[3]
    #client5 = client_group.clients[5]


    # cora_dataset = client_group.cora_data
    # avg = 0.0
    # for i in range(10):
    #     client_group.clients[i].local_train_graph(client_group.clients[i].local_model, cora_dataset)
    #     acc, _ = client_group.clients[i].eval_model_graph(cora_dataset.cora_dataset)
    #     avg += acc
    # print(avg / 10.0)

    # start = time.time()
    #client1.local_train_graph(client1.local_model, cora_dataset)
    # end = time.time()
    # print(end-start)
    # client1.save_model() # 12.79
    #
    # start = time.time()
    #client2.local_train_graph(client2.local_model, cora_dataset)
    # end = time.time()
    # print(end-start)
    # client2.save_model() # 13.65

    # client3.local_train(client3.local_model)
    # client3.save_model() # 39.87
    # client5.local_train(client5.local_model)
    # client5.save_model() # 28.31

    # client0.load_model(os.path.join("./models/clients/client0/", 'model.pth'))
    # client1.load_model(os.path.join("./models/clients/client1/", 'model.pth'))
    # client2.load_model(os.path.join("./models/clients/client2/", 'model.pth'))

    #client2.local_train(client2.local_model)
    # client2.load_model(os.path.join("./models/clients/client2/", 'model.pth'))
    #client2.fuse_model_by_teachers([[client0.client_id, client0.local_model], [client1.client_id, client1.local_model]])
    #
    # client2.load_model(os.path.join("./models/clients/client2/", 'model.pth'))
    # client2.fuse_model_by_teachers([[client0.client_id, client0.local_model]])
    #
    # client2.load_model(os.path.join("./models/clients/client2/", 'model.pth'))
    # client2.fuse_model_by_teachers([[client1.client_id, client1.local_model]])
    #
    # client0.load_model(os.path.join("./models/clients/client0/", 'model.pth'))
    # client1.load_model(os.path.join("./models/clients/client1/", 'model.pth'))
    # client2.load_model(os.path.join("./models/clients/client2/", 'model.pth'))
    #
    # client0.local_train(client0.local_model)
    # client0.load_model(os.path.join("./models/clients/client0/", 'model.pth'))
    # client0.fuse_model_by_teachers([(client1.client_id, client1.local_model), (client2.client_id, client2.local_model)])
    #
    # client0.load_model(os.path.join("./models/clients/client0/", 'model.pth'))
    # client0.fuse_model_by_teachers([(client1.client_id, client1.local_model)])
    #
    # client0.load_model(os.path.join("./models/clients/client0/", 'model.pth'))
    # client0.fuse_model_by_teachers([(client2.client_id, client2.local_model)])
    #
    #
    # client0.load_model(os.path.join("./models/clients/client0/", 'model.pth'))
    # client3.load_model(os.path.join("./models/clients/client3/", 'model.pth'))
    # client5.load_model(os.path.join("./models/clients/client5/", 'model.pth'))
    #
    # client5.local_train(client5.local_model)
    # client5.load_model(os.path.join("./models/clients/client5/", 'model.pth'))
    # client5.fuse_model_by_teachers([(client0.client_id, client0.local_model), (client3.client_id, client3.local_model)])
    #
    # client5.load_model(os.path.join("./models/clients/client5/", 'model.pth'))
    # client5.fuse_model_by_teachers([(client0.client_id, client0.local_model)])
    #
    # client5.load_model(os.path.join("./models/clients/client5/", 'model.pth'))
    # client5.fuse_model_by_teachers([(client3.client_id, client3.local_model)])

    # extractors = []
    # for id in range(conf["client_num"]):
    #     extractors.append(client_group.clients[id].extractor)
    #
    # similarity_vector = client1.extractor.sparsity_similarity_vector(client1.extractor.mean_list, extractors)
    # client1.extractor.similarity_vector = similarity_vector
    # matrix = np.zeros((3, 3))
    # sv = []
    # for e in range(2):
    #     client0.local_train(client0.local_model)
    #     client1.local_train(client1.local_model)
    #     client2.local_train(client2.local_model)
    #     sv.append(client0.extractor.sparsity_similarity_vector(client0.extractor.mean_list, extractors))
    #     sv.append(client1.extractor.sparsity_similarity_vector(client1.extractor.mean_list, extractors))
    #     sv.append(client2.extractor.sparsity_similarity_vector(client2.extractor.mean_list, extractors))
    #     print(sv)
    #
    #     # matrix[0] = sv0
    #     # matrix[1] = sv1
    #     # matrix[2] = sv2
    #     # print(matrix)
    #     server.cluster_model_weight_aggregate([(id, client_group.clients[id].local_model, sv[id]) for id in range(3)])
