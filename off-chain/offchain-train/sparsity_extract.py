import models
from torch import nn
import torch
import numpy as np
import dataset
import json
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, model, dataset_name):
        super(FeatureExtractor, self).__init__()

        self.dataset_name = dataset_name

        if dataset_name == "mnist" or dataset_name == "fashion_mnist":
            self.extracted_layer = 12
            self.num_channels = 128
            self.mods = torch.nn.ModuleList([model._modules["features"][i] for i in range(self.extracted_layer+1)])
            self.mean_list = np.random.random((1, self.num_channels))
        elif dataset_name == "cifar10" or dataset_name == "cifar100" or dataset_name == "imagenet":
            self.extracted_layer = 27
            self.num_channels = 512
            self.mods = torch.nn.ModuleList([model.features._modules["features"][i] for i in range(self.extracted_layer+1)])
            self.mean_list = np.random.random((1, self.num_channels))

        self.similarity_vector = None
        self.old_all_vectors = None

    def forward(self, x):
        mean_temp = np.zeros((x.size(0), 1))

        #目前只是一层的channel，如果多层需要拼接
        for module in self.mods:
            x = module(x)

        temp = x.cpu().detach().numpy()
        density = np.count_nonzero(temp, (2, 3))
        sparsity = (np.size(temp, 2) * np.size(temp, 3) - density) / (np.size(temp, 2) * np.size(temp, 3))

        mean_temp = np.concatenate((mean_temp, sparsity), axis=1)
        mean_temp = np.delete(mean_temp, 0, axis=1)

        # print(sparsity.shape)
        # print(sparsity)

        self.mean_list = np.concatenate((self.mean_list, mean_temp), axis=0)
        #print(self.mean_list.shape)

    def update_model(self, model):
        if self.dataset_name == "mnist" or self.dataset_name == "fashion_mnist":
            self.mods = torch.nn.ModuleList([model._modules["features"][i] for i in range(self.extracted_layer + 1)])
        elif self.dataset_name == "cifar10" or self.dataset_name == "cifar100" or self.dataset_name == "imagenet":
            self.mods = torch.nn.ModuleList([model.features._modules["features"][i] for i in range(self.extracted_layer + 1)])

    def sparsity_similarity(self, v1, v2):
        #print("{}, {}".format(v1.shape, v2.shape))
        # temp = np.sum(np.square(v2 - v1))
        # similarity = 1.0 / (np.sqrt(temp)) if temp != 0 else 1.0
        # return similarity

        temp = np.sum(np.square(v2 - v1))
        distance = np.sqrt(temp)
        return distance

        # num = float(np.dot(v1, v2.T))  # 向量点乘
        # denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
        # similarity = 0.5 + 0.5 * (num / denom) if denom != 0 else 0
        #return similarity

    def softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def sparsity_similarity_vector(self, v1, v2list):
        similarity_vector = np.zeros(len(v2list))
        for i in range(len(v2list)):
            similarity_vector[i] = self.sparsity_similarity(v1, v2list[i].mean_list)

        # temp = np.array([])
        # for similarity in similarity_vector:
        #     if similarity != 1.0:
        #         temp = np.append(temp, similarity)
        # temp = self.softmax(temp)
        # t = 0
        # for i in range(len(v2list)):
        #     if similarity_vector[i] == 1.0:
        #         continue
        #     similarity_vector[i] = temp[t]
        #     t = t + 1

        return similarity_vector

class NLPFeatureExtractor(FeatureExtractor):
    def __init__(self, model, dataset_name):
        super(NLPFeatureExtractor, self).__init__(model, dataset_name)
        self.model = model
        self.num_channels = 80
        self.mean_list = np.random.random((1, self.num_channels))
        self.similarity_vector = None

    def forward(self, x):
        output = self.model(x)
        temp = output.cpu().detach().numpy()
        self.mean_list = np.concatenate((self.mean_list, temp), axis=0)

    def update_model(self, model):
        self.model = model

class GCNFeatureExtractor(FeatureExtractor):
    def __init__(self, model, dataset_name):
        super(GCNFeatureExtractor, self).__init__(model, dataset_name)

        self.extracted_layer = 0
        self.mod = model.gc1
        self.mean_list = np.random.random((1, 3312))

        self.similarity_vector = None

    def forward(self, x, adj):
        mean_temp = np.zeros((1, x.size(0)))
        mean_temp += 16

        x = F.relu(self.mod(x, adj))
        temp = x.cpu().detach().numpy()
        density = np.count_nonzero(temp, axis=1)
        sparsity = (mean_temp - density) / 16
        self.mean_list = sparsity

    def update_model(self, model):
        self.mod = model.gc1

if __name__ == '__main__':
    with open("./config/test-conf.json", 'r') as f:
        conf = json.load(f)
    print(conf)
    dataset = dataset.GetDataSet(dataset_name=conf["dataset"], is_iid=conf["iid"], beta=conf["niid_beta"])
    test_loader = torch.utils.data.DataLoader(dataset.test_dataset, batch_size=32,
                                                   shuffle=False, num_workers=4)

    device = torch.device("cuda:"+str(0))
    model = models.CharLSTM()
    model = model.to(device)

    extractor = NLPFeatureExtractor(model=model, dataset_name=conf["dataset"])

    for batch in test_loader:
        data, target = batch
        data = data.to(device)
        target = target.to(device, dtype=torch.long)

        extractor(data)

    #512个channel的稀疏度向量表示
    extractor.mean_list = np.delete(extractor.mean_list, 0 ,0)
    print(extractor.mean_list.shape)
    extractor.mean_list = extractor.mean_list.mean(axis=0)
    extractor.mean_list = extractor.mean_list.reshape((1, extractor.num_channels))
    print(extractor.mean_list.shape)