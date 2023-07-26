import random
import torch
import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict
import gzip
import torchvision.utils
from torchvision import datasets, transforms
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import sys
from pathlib import Path
from tqdm import tqdm, trange
import scipy.sparse as sp
import networkx as nx
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("./models/bert-base-uncased")

# import models
# from fedlab_benchmarks.leaf.dataset.sent140_dataset import Sent140Dataset
# from torch_geometric.datasets import Planetoid

np.random.seed(33)
random.seed(33)

# def process_features(features):
#     row_sum_diag = np.sum(features, axis=1)
#     row_sum_diag_inv = np.power(row_sum_diag+1e-5, -1)
#     row_sum_diag_inv[np.isinf(row_sum_diag_inv)] = 0.
#     row_sum_inv = np.diag(row_sum_diag_inv)
#     return np.dot(row_sum_inv, features)
#
# def preprocess_adj(A):
#     '''
#     Pre-process adjacency matrix
#     :param A: adjacency matrix
#     :return:
#     '''
#     I = np.eye(A.shape[0])
#     A_hat = A + I  # add self-loops
#     D_hat_diag = np.sum(A_hat, axis=1)
#     D_hat_diag_inv_sqrt = np.power(D_hat_diag, -0.5)
#     D_hat_diag_inv_sqrt[np.isinf(D_hat_diag_inv_sqrt)] = 0.
#     D_hat_inv_sqrt = np.diag(D_hat_diag_inv_sqrt)
#     return np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
#
#
# def sample_mask(idx, l):
#     mask = np.zeros(l)
#     mask[idx] = 1
#     return np.array(mask, dtype=np.bool)
#
#
# def load_data(dataset):
#     ## get data
#     data_path = 'data/data'
#     suffixs = ['x', 'y', 'allx', 'ally', 'tx', 'ty', 'graph']
#     objects = []
#     for suffix in suffixs:
#         file = os.path.join(data_path, 'ind.%s.%s'%(dataset, suffix))
#         objects.append(pickle.load(open(file, 'rb'), encoding='latin1'))
#     x, y, allx, ally, tx, ty, graph = objects
#     x, allx, tx = x.toarray(), allx.toarray(), tx.toarray()
#
#     # test indices
#     test_index_file = os.path.join(data_path, 'ind.%s.test.index'%dataset)
#     with open(test_index_file, 'r') as f:
#         lines = f.readlines()
#     indices = [int(line.strip()) for line in lines]
#     min_index, max_index = min(indices), max(indices)
#
#     # preprocess test indices and combine all data
#     tx_extend = np.zeros((max_index - min_index + 1, tx.shape[1]))
#     features = np.vstack([allx, tx_extend])
#     features[indices] = tx
#     ty_extend = np.zeros((max_index - min_index + 1, ty.shape[1]))
#     labels = np.vstack([ally, ty_extend])
#     labels[indices] = ty
#
#     # get adjacency matrix
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).toarray()
#
#     idx_train = range(len(y))
#     idx_val = range(len(y), len(y) + 500)
#     idx_test = indices
#
#     train_mask = sample_mask(idx_train, labels.shape[0])
#     val_mask = sample_mask(idx_val, labels.shape[0])
#     test_mask = sample_mask(idx_test, labels.shape[0])
#     zeros = np.zeros(labels.shape)
#     y_train = zeros.copy()
#     y_val = zeros.copy()
#     y_test = zeros.copy()
#     y_train[train_mask, :] = labels[train_mask, :]
#     y_val[val_mask, :] = labels[val_mask, :]
#     y_test[test_mask, :] = labels[test_mask, :]
#     features = torch.from_numpy(process_features(features))
#
#     # train_mask = np.where(train_mask==True)[0]
#     # val_mask = np.where(val_mask==True)[0]
#     # test_mask = np.where(test_mask==True)[0]
#
#     y_train = np.argmax(y_train, axis=1)
#     y_val = np.argmax(y_val, axis=1)
#     y_test = np.argmax(y_test, axis=1)
#
#     y_train, y_val, y_test, train_mask, val_mask, test_mask = \
#         torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(y_test), \
#         torch.from_numpy(train_mask), torch.from_numpy(val_mask), torch.from_numpy(test_mask)
#
#     np.set_printoptions(threshold=np.inf)
#     print(len(labels))
#     labels = torch.LongTensor(np.where(labels)[1])
#
#     return adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask


class AgnewsDataset(Dataset):
    def __init__(self, train=True, data_dir="./data/AGNEWS/", bert_dir="./models/bert-base-uncased/"):
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir, do_lower_case=False)

        if train:
            df = pd.read_csv(os.path.join(data_dir, "train1.csv"))
        else:
            df = pd.read_csv(os.path.join(data_dir, "test.csv"))

        df['text'] = df.iloc[:, 1] + " " + df.iloc[:, 2]
        df = df.drop(df.columns[[1, 2]], axis=1)
        df.columns = ['label', 'text']
        df = df[['text', 'label']]
        df['text'] = df['text'].apply(lambda x: x.replace('\\', ' '))
        df['label'] = df['label'].apply(lambda x: x - 1)
        #print(df)

        self.ids = self.padAndTokenizeForBert(df.text, 128)
        self.masks = self.masker(self.ids)
        self.labels = np.array(df['label'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.ids[index], self.masks[index], self.labels[index])

    def padSentenceForBert(self, sent, cls_id, sep_id, lg):
        return np.array([cls_id] + sent[:lg - 2] + [sep_id] + [0] * (lg - len(sent[:lg - 2]) - 2))

    def padAndTokenizeForBert(self, sentences, lg):
        cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        return np.array([self.padSentenceForBert(self.tokenizer.encode(sent), cls_id, sep_id, lg) for sent in sentences])

    def masker(self, input_ids):
        return np.array([[float(i > 0) for i in seq] for seq in input_ids])

class CoraDataset(Dataset):
    def __init__(self, dataset_name="cora"):
        self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test = self.load_data()

        #self.adj, self.features, self.labels, self.y_train, self.y_val, self.y_test, self.idx_train, self.idx_val, self.idx_test = load_data(dataset_name)

    def load_data(self, path="./data/cora/", dataset="cora"):
        print('Loading {} dataset...'.format(dataset))
        # content数据加载
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        # 获取特征向量，并将特征向量转为稀疏矩阵，
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        # 获取标签
        labels = self.encode_onehot(idx_features_labels[:, -1])
        # 搭建图
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        # 搭建字典，论文编号-->索引
        idx_map = {j: i for i, j in enumerate(idx)}
        # cites数据加载，shape：5429,2
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
        # 边，将编号映射为索引，因为编号是非连续的整数
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
            edges_unordered.shape)
        print(edges.dtype)
        # 构建邻接矩阵
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # 转换为对称邻接矩阵
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # 归一化特征矩阵和邻接矩阵
        features = self.normalize(features)
        adj = self.normalize(adj + sp.eye(adj.shape[0]))

        # 设置训练、验证和测试的数量
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test
        # adj 		2708,2708
        # features 	2708,1433
        # labels	2708,      0~6

    def encode_onehot(self, labels):
        classes = list(set(labels))
        classes.sort(key=list(labels).index)
        print(classes)

        #cora
        #classes = ['Neural_Networks', 'Probabilistic_Methods', 'Genetic_Algorithms', 'Case_Based', 'Theory', 'Rule_Learning', 'Reinforcement_Learning']

        #citeseer


        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


class CiteseerDataset(Dataset):
    def __init__(self, dataset_name="citeseer"):
        self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test = self.load_data()

        #self.adj, self.features, self.labels, self.y_train, self.y_val, self.y_test, self.idx_train, self.idx_val, self.idx_test = load_data(dataset_name)

    def load_data(self, path="./data/citeseer/", dataset="citeseer"):
        print('Loading {} dataset...'.format(dataset))
        # content数据加载
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        # 获取特征向量，并将特征向量转为稀疏矩阵，
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        # 获取标签
        labels = self.encode_onehot(idx_features_labels[:, -1])
        # 搭建图
        #idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx = idx_features_labels[:, 0]
        # 搭建字典，论文编号-->索引
        idx_map = {j: i for i, j in enumerate(idx)}
        # cites数据加载，shape：5429,2
        #edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.str_)
        edges_unordered = pd.read_table("{}{}.cites".format(path, dataset), sep='\t', header=None)
        # 边，将编号映射为索引，因为编号是非连续的整数
        # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
        edges_unordered[0] = edges_unordered[0].map(lambda x: idx_map.get(x))
        edges_unordered[1] = edges_unordered[1].map(lambda x: idx_map.get(x))

        edges = np.array(edges_unordered, dtype=np.int32)
        edges = edges[(edges >= 0).all(axis=1)]

        # 构建邻接矩阵
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # 转换为对称邻接矩阵
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # 归一化特征矩阵和邻接矩阵
        features = self.normalize(features)
        adj = self.normalize(adj + sp.eye(adj.shape[0]))

        # 设置训练、验证和测试的数量
        idx_train = range(300)
        idx_val = range(500, 800)
        idx_test = range(800, 2500)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test
        # adj 		2708,2708
        # features 	2708,1433
        # labels	2708,      0~6

    def encode_onehot(self, labels):
        classes = list(set(labels))
        classes.sort(key=list(labels).index)
        print(classes)

        #cora
        #classes = ['Neural_Networks', 'Probabilistic_Methods', 'Genetic_Algorithms', 'Case_Based', 'Theory', 'Rule_Learning', 'Reinforcement_Learning']

        #citeseer

        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


class ShakeSpeare(Dataset):
    def __init__(self, train=True, client_nums = 20):
        train_clients, train_groups, train_data_temp, test_data_temp = self.read_data("./data/shakespeare/train", "./data/shakespeare/test")
        self.train = train
        self.ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        self.NUM_LETTERS = len(self.ALL_LETTERS)

        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                if i == client_nums:
                    break
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(cur_x[j])
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y

        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                if i == client_nums:
                    break
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(cur_x[j])
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index], self.label[index]
        indices = self.word_to_indices(sentence)
        target = self.letter_to_vec(target)
        # y = indices[1:].append(target)
        # target = indices[1:].append(target)
        indices = torch.LongTensor(np.array(indices))
        # y = torch.Tensor(np.array(y))
        # target = torch.LongTensor(np.array(target))
        return indices, target

    def letter_to_vec(self, letter):
        '''returns one-hot representation of given letter
        '''
        index = self.ALL_LETTERS.find(letter)
        return index

    def word_to_indices(self, word):
        '''returns a list of character indices
        Args:
            word: string

        Return:
            indices: int list with length len(word)
        '''
        indices = []
        for c in word:
            indices.append(self.ALL_LETTERS.find(c))
        return indices

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")

    def read_dir(self, data_dir):
        clients = []
        groups = []
        data = defaultdict(lambda: None)

        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]
        for f in files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            data.update(cdata['user_data'])

        clients = list(sorted(data.keys()))
        return clients, groups, data

    def read_data(self, train_data_dir, test_data_dir):
        '''parses data in given train and test data directories
        assumes:
        - the data in the input directories are .json files with
            keys 'users' and 'user_data'
        - the set of train set users is the same as the set of test set users
        Return:
            clients: list of client ids
            groups: list of group ids; empty list if none found
            train_data: dictionary of train data
            test_data: dictionary of test data
        '''
        train_clients, train_groups, train_data = self.read_dir(train_data_dir)
        test_clients, test_groups, test_data = self.read_dir(test_data_dir)

        assert train_clients == test_clients
        assert train_groups == test_groups

        return train_clients, train_groups, train_data, test_data

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, load_from_numpy=False):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self.images = []
        self.labels = []
        if load_from_numpy:
            self.images = np.load(os.path.join(self.root_dir, ("train" if self.Train else "val") + "_images.npy"))
            self.labels = np.load(os.path.join(self.root_dir, ("train" if self.Train else "val") + "_labels.npy"))
        else:
            self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):

        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]

            save_img_path = os.path.join(self.root_dir, "train_images.npy")
            save_label_path = os.path.join(self.root_dir, "train_labels.npy")
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

            save_img_path = os.path.join(self.root_dir, "val_images.npy")
            save_label_path = os.path.join(self.root_dir, "val_labels.npy")

        transform = transforms.Compose([transforms.ToTensor()])
        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)

                        with open(path, 'rb') as f:
                            img = Image.open(path)
                            img = img.convert('RGB')
                            #img = np.asarray(img)

                            img = transform(img)
                            img = img.numpy()
                            #img = np.transpose(img, (1, 2, 0))

                        self.images.append(img)
                        if Train:
                            self.labels.append(self.class_to_tgt_idx[tgt])
                        else:
                            self.labels.append(self.class_to_tgt_idx[self.val_img_to_class[fname]])

                        # if Train:
                        #     item = (path, self.class_to_tgt_idx[tgt])
                        # else:
                        #     item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        #self.images.append(item)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        print(self.images.shape)
        print(self.labels.shape)

        np.save(save_img_path, self.images)
        np.save(save_label_path, self.labels)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        sample = self.images[idx]
        tgt = self.labels[idx]
        # img_path, tgt = self.images[idx]
        # with open(img_path, 'rb') as f:
        #     sample = Image.open(img_path)
        #     sample = sample.convert('RGB')
        #
        # sample = np.asarray(sample)
        # if self.transform is not None:
        #     sample = self.transform(sample)

        return sample, tgt

BASE_DIR = Path(__file__).resolve().parents[1]
class PickleDataset:
    def __init__(self, dataset_name: str, data_root: str = None, pickle_root: str = None):
        self.dataset_name = dataset_name
        self.data_root = Path(data_root) if data_root is not None else BASE_DIR / "datasets"
        self.pickle_root = Path(pickle_root) if pickle_root is not None else Path(__file__).parent / "pickle_datasets"

    def get_dataset_pickle(self, dataset_type: str, users_range: int = 10, user_id: int = None):
        """load pickle dataset file for `dataset_name` `dataset_type` data based on client with client_id

        Args:
            dataset_type (str): Dataset type {train, test}
            client_id (int): client id. Defaults to None, which means get all_dataset pickle
        Raises:
            FileNotFoundError: No such file or directory {pickle_root}/{dataset_name}/{dataset_type}/{dataset_type}_{client_id}.pickle
        Returns:
            if there is no pickle file for `dataset`, throw FileNotFoundError, else return responding dataset
        """
        # check whether to get all datasets
        if user_id is None:
            pickle_files_path = self.pickle_root / self.dataset_name / dataset_type
            dataset_list = []
            index_list = []
            index = 0
            for i in trange(users_range):
                dataset = pickle.load(open(os.path.join(pickle_files_path, f"{dataset_type}_{i}.pkl"), 'rb'))
                dataset_list.append(dataset)
                index_list.append([index+i for i in range(len(dataset))])
                index += len(dataset)

            dataset = ConcatDataset(dataset_list)
        else:
            pickle_file = self.pickle_root / self.dataset_name / dataset_type / f"{dataset_type}_{user_id}.pkl"
            dataset = pickle.load(open(pickle_file, 'rb'))
            index_list = list(range(len(dataset)))
        return dataset, index_list


class GetDataSet(object):
    def __init__(self, dataset_name, is_iid=True, beta=0.05, client_num=10):
        self.name = dataset_name
        self.isIID = is_iid #数据是否满足独立同分布
        self.beta = beta
        self.client_num = client_num

        self.train_data = None  # 训练集
        self.train_label = None  # 标签
        self.train_data_size = None  # 训练数据的大小
        self.test_data = None  # 测试数据集
        self.test_label = None  # 测试的标签
        self.test_data_size = None  # 测试集数据大小


        if self.name == "mnist":
            self.get_mnist_dataset()

        if self.name == "fashion_mnist":
            self.get_fashion_mnist_dataset(client_num=self.client_num)

        if self.name == "cifar10":
            self.get_cifar10_dataset(client_num=self.client_num)

        if self.name == "cifar100":
            self.get_cifar100_dataset(client_num=self.client_num)

        if self.name == "imagenet":
            self.get_imagenet_dataset(client_num=self.client_num)

        if self.name == "agnews":
            self.get_agnews_dataset()

        if self.name == "shakespeare":
            self.get_shakespeare_dataset()

        if self.name == "sent140":
            self.get_sent140_dataset()

        if self.name == "cora":
            self.get_cora_dataset()

        if self.name == "citeseer":
            self.get_citeseer_dataset()

    def get_mnist_dataset(self, data_dir = "./data/"):
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor())

        train_images = train_dataset.data.numpy()
        train_labels = train_dataset.targets.numpy()
        test_images = test_dataset.data.numpy()
        test_labels = test_dataset.targets.numpy()

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        #train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])

        #test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        test_dataset.data = torch.tensor(test_images)

        if self.isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            train_dataset.data = torch.tensor(train_images[order])
            train_dataset.targets = torch.tensor(train_labels[order])
            #print(train_dataset.targets)

        else:
            order = np.argsort(train_labels)
            # print(order)
            #print(train_labels[order])
            distribution={}
            for label in order:
                    if train_labels[label] not in distribution:
                        distribution[train_labels[label]]=1
                    else:
                        distribution[train_labels[label]]+=1
            print("Label Distribution : {}".format(distribution))

            train_dataset.data = torch.tensor(train_images[order])
            train_dataset.targets = torch.tensor(train_labels[order])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        print("The shape of mnist data: {}".format(self.train_dataset.data.shape))
        print("The shape of mnist label: {}".format(self.train_dataset.targets.shape))

    def get_fashion_mnist_dataset(self, client_num=10, class_num=10, data_dir="./data/"):
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.FashionMNIST(data_dir, train=False, transform=transforms.ToTensor())

        train_images = train_dataset.data.numpy()
        train_labels = train_dataset.targets.numpy()
        test_images = test_dataset.data.numpy()
        test_labels = test_dataset.targets.numpy()

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        # train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])

        # test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        test_dataset.data = torch.tensor(test_images)

        if self.isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            train_dataset.data = torch.tensor(train_images[order])
            train_dataset.targets = torch.tensor(train_labels[order])
            # print(train_dataset.targets)

        else:
            # order = np.argsort(train_labels)
            # # print(order)
            # # print(train_labels[order])
            # distribution = {}
            # for label in order:
            #     if train_labels[label] not in distribution:
            #         distribution[train_labels[label]] = 1
            #     else:
            #         distribution[train_labels[label]] += 1
            # print("Label Distribution : {}".format(distribution))
            #
            # train_dataset.data = torch.tensor(train_images[order])
            # train_dataset.targets = torch.tensor(train_labels[order])

            train_dataset.data = torch.tensor(train_images)
            train_dataset.targets = torch.tensor(train_labels)

            label_distribution = np.random.dirichlet([self.beta] * client_num, class_num)
            class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(class_num)]

            client_idcs = [[] for _ in range(client_num)]
            for c, fracs in zip(class_idcs, label_distribution):
                for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                    client_idcs[i] += [idcs]

            client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
            self.client_idcs = client_idcs

            for idcs in client_idcs:
                a = train_labels[idcs]
                distribution = {}
                for label in a:
                    if label not in distribution:
                        distribution[label] = 1
                    else:
                        distribution[label] += 1
                print("Label Distribution : {}".format(distribution))

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        print("The shape of fashion_mnist data: {}".format(self.train_dataset.data.shape))
        print("The shape of fashion_mnist label: {}".format(self.train_dataset.targets.shape))

    def get_sent140_dataset(self, client_num=10, class_num=2):
        train_dataset = Sent140(train=True)
        test_dataset = Sent140(train=False)

        #train_order = np.random.choice(range(len(train_dataset)), 100000)
        train_data = np.array(train_dataset.data)
        train_dataset.data = train_data
        train_labels = np.array(train_dataset.label)
        train_dataset.label = train_labels

        #test_order = np.random.choice(range(len(test_dataset)), 10000)
        test_data = np.array(test_dataset.data)
        test_dataset.data = test_data
        test_labels = np.array(test_dataset.label)
        test_dataset.label = test_labels

        self.train_data_size = train_dataset.label.shape[0]
        self.test_data_size = test_dataset.label.shape[0]

        if self.isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            # print(train_data[order])
            train_dataset.data = train_data[order]
            train_dataset.label = train_labels[order]

        else:
            # train_dataset.data = train_data
            # train_dataset.label = train_labels

            # client_idcs = [0] * 10;
            # print(len(train_dataset.dic_users))
            # for i in range(10):
            #     client_idcs[i] = list(train_dataset.dic_users[i*2]) + list(train_dataset.dic_users[i*2+1])


            label_distribution = np.random.dirichlet([self.beta] * client_num, class_num)
            class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(class_num)]

            client_idcs = [[] for _ in range(client_num)]
            for c, fracs in zip(class_idcs, label_distribution):
                for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                    client_idcs[i] += [idcs]

            client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
            self.client_idcs = client_idcs

            # client_idcs = train_dataset.get_client_dic()
            # self.client_idcs = [[] for _ in range(client_num)]
            # for i in range(client_num):
            #     self.client_idcs[i] = list(client_idcs[i])

            for idcs in self.client_idcs:
                a = train_labels[idcs]
                distribution={}
                for label in a:
                        if label not in distribution:
                            distribution[label]=1
                        else:
                            distribution[label]+=1
                print("Label Distribution : {}".format(distribution))

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        print("The shape of shakespeare train data: {}".format(self.train_dataset.data.shape))
        print("The shape of shakespeare test data: {}".format(self.test_dataset.data.shape))

    def get_cora_dataset(self, client_num=10, class_num=7, data_dir="./data/cora"):
        cora_dataset = CoraDataset()

        client_idcs = []
        if self.isIID:
            for i in range(client_num):
                idcs = np.random.choice(cora_dataset.idx_train, (int)(len(cora_dataset.idx_train)/client_num))
                client_idcs.append(idcs)
        else:

            label_distribution = np.random.dirichlet([self.beta] * client_num, class_num)
            class_idcs = [np.argwhere(cora_dataset.labels[cora_dataset.idx_train] == y).flatten() for y in range(class_num)]

            client_idcs = [[] for _ in range(client_num)]
            for c, fracs in zip(class_idcs, label_distribution):
                for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                    client_idcs[i] += [idcs]

            # for i in range(client_num):
            #     for c in class_idcs:
            #         idcs = np.random.choice(c.numpy(), np.random.choice([0, 1], 1))
            #         client_idcs[i] += [idcs]

            client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

        for idcs in client_idcs:
            a = cora_dataset.labels.cpu().numpy()[idcs]
            distribution = {}
            for label in a:
                if label not in distribution:
                    distribution[label] = 1
                else:
                    distribution[label] += 1
            print("Label Distribution : {}".format(distribution))

        self.client_idcs = client_idcs
        self.cora_dataset = cora_dataset

    def get_citeseer_dataset(self, client_num=10, class_num=6, data_dir="./data/cora"):

        cora_dataset = CiteseerDataset()

        client_idcs = []
        if self.isIID:
            for i in range(client_num):
                idcs = np.random.choice(cora_dataset.idx_train, (int)(len(cora_dataset.idx_train) / client_num))
                client_idcs.append(idcs)
        else:

            label_distribution = np.random.dirichlet([self.beta] * client_num, class_num)
            class_idcs = [np.argwhere(cora_dataset.labels[cora_dataset.idx_train] == y).flatten() for y in
                          range(class_num)]

            client_idcs = [[] for _ in range(client_num)]
            for c, fracs in zip(class_idcs, label_distribution):
                for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                    client_idcs[i] += [idcs]

            # for i in range(client_num):
            #     for c in class_idcs:
            #         idcs = np.random.choice(c.numpy(), np.random.choice([0, 1, 2], 1))
            #         client_idcs[i] += [idcs]

            client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

        for idcs in client_idcs:
            a = cora_dataset.labels.cpu().numpy()[idcs]
            distribution = {}
            for label in a:
                if label not in distribution:
                    distribution[label] = 1
                else:
                    distribution[label] += 1
            print("Label Distribution : {}".format(distribution))

        self.client_idcs = client_idcs
        self.cora_dataset = cora_dataset


        # cora_dataset = CoraDataset(dataset_name="citeseer")
        #
        # if self.isIID:
        #
        #     idx_list = cora_dataset.idx_train.numpy()
        #     idx_list = np.where(idx_list == True)[0]
        #     client_idcs = []
        #     for i in range(client_num):
        #         idcs = np.random.choice(idx_list, int(len(idx_list) / client_num))
        #         client_mask = []
        #         for i in range(len(cora_dataset.idx_train)):
        #             if i in idcs:
        #                 client_mask.append(True)
        #             else:
        #                 client_mask.append(False)
        #         client_idcs.append(client_mask)
        #     # print(client_idcs)
        #
        # else:
        #     idx_list = [[] for _ in range(client_num)]
        #     label_distribution = np.random.dirichlet([self.beta] * client_num, class_num)
        #     class_idcs = [np.argwhere(cora_dataset.labels[cora_dataset.idx_train] == y).flatten() for y in range(class_num)]
        #
        #     for c, fracs in zip(class_idcs, label_distribution):
        #         for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
        #             idx_list[i] += [idcs]
        #
        #     idx_list = [np.concatenate(idcs) for idcs in idx_list]
        #
        #     client_idcs = []
        #     for i in range(client_num):
        #         client_mask = []
        #         idcs = idx_list[i]
        #         for j in range(len(cora_dataset.idx_train)):
        #             if j in idcs:
        #                 client_mask.append(True)
        #             else:
        #                 client_mask.append(False)
        #         client_idcs.append(client_mask)
        #
        # for idcs in client_idcs:
        #     print(cora_dataset.labels.shape)
        #     a = cora_dataset.labels[idcs]
        #     distribution = {}
        #     for label in a:
        #         if label not in distribution:
        #             distribution[label] = 1
        #         else:
        #             distribution[label] += 1
        #     print("Label Distribution : {}".format(distribution))
        #
        # self.client_idcs = client_idcs
        # self.cora_dataset = cora_dataset

    def get_cifar10_dataset(self, client_num=10, class_num=10, data_dir="./data/CIFAR"):
        transform_train = transforms.Compose([
                    #transforms.RandomCrop(32, padding=4),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform_test)

        train_data = train_dataset.data
        train_labels = np.array(train_dataset.targets)
        test_data = test_dataset.data
        test_labels = np.array(test_dataset.targets)
        test_dataset.targets = test_labels

        self.train_data_size = train_data.shape[0]
        self.test_data_size = test_data.shape[0]
        print(train_data.shape)
        if self.isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            train_dataset.data = train_data[order]
            train_dataset.targets = train_labels[order]

        else:
            #order = np.argsort(train_labels)
            # distribution={}
            # for label in order:
            #         if train_labels[label] not in distribution:
            #             distribution[train_labels[label]]=1
            #         else:
            #             distribution[train_labels[label]]+=1
            # print("Label Distribution : {}".format(distribution))
            #
            # iid_list = []
            # total_size = 5000
            # niid_size = int(total_size * self.beta)
            # iid_size = total_size - niid_size
            # for i in range(10):
            #     iid_list.extend(order[i*total_size+niid_size:(i+1)*total_size])
            # random.shuffle(iid_list)
            #
            # new_order = []
            # for i in range(10):
            #     new_order.extend(order[i*total_size:i*total_size+niid_size])
            #     new_order.extend(iid_list[i*iid_size:(i+1)*iid_size])
            #
            # train_dataset.data = train_data[new_order]
            # train_dataset.targets = train_labels[new_order]

            train_dataset.data = train_data
            train_dataset.targets = train_labels

            label_distribution = np.random.dirichlet([self.beta]*client_num, class_num)
            class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(class_num)]

            client_idcs = [[] for _ in range(client_num)]
            for c, fracs in zip(class_idcs, label_distribution):
                for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                    client_idcs[i] += [idcs]

            client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
            self.client_idcs = client_idcs


            for idcs in client_idcs:
                a=train_labels[idcs]
                distribution={}
                for label in a:
                        if label not in distribution:
                            distribution[label]=1
                        else:
                            distribution[label]+=1
                print("Label Distribution : {}".format(distribution))


        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        print("The shape of cifar10 train data: {}".format(self.train_dataset.data.shape))
        print("The shape of cifar10 train label: {}".format(self.train_dataset.targets.shape))
        print("The shape of cifar10 test data: {}".format(self.test_dataset.data.shape))

    def get_shakespeare_dataset(self, client_num=10, class_num=80):
        train_dataset = ShakeSpeare(train=True)
        test_dataset = ShakeSpeare(train=False)

        #train_order = np.random.choice(range(len(train_dataset)), 100000)
        train_data = np.array(train_dataset.data)
        train_dataset.data = train_data
        train_labels = np.array(train_dataset.label)
        train_dataset.label = train_labels

        #test_order = np.random.choice(range(len(test_dataset)), 10000)
        test_data = np.array(test_dataset.data)
        test_dataset.data = test_data
        test_labels = np.array(test_dataset.label)
        test_dataset.label = test_labels

        self.train_data_size = train_dataset.label.shape[0]
        self.test_data_size = test_dataset.label.shape[0]

        if self.isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            # print(train_data[order])
            train_dataset.data = train_data[order]
            train_dataset.label = train_labels[order]

        else:
            # train_dataset.data = train_data
            # train_dataset.label = train_labels

            # client_idcs = [0] * 10;
            # print(len(train_dataset.dic_users))
            # for i in range(10):
            #     client_idcs[i] = list(train_dataset.dic_users[i*2]) + list(train_dataset.dic_users[i*2+1])

            ALL_LETTERS = train_dataset.ALL_LETTERS

            label_distribution = np.random.dirichlet([self.beta] * client_num, class_num)
            class_idcs = [np.argwhere(train_labels == ALL_LETTERS[y]).flatten() for y in range(class_num)]

            client_idcs = [[] for _ in range(client_num)]
            for c, fracs in zip(class_idcs, label_distribution):
                for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                    client_idcs[i] += [idcs]

            client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
            self.client_idcs = client_idcs

            # client_idcs = train_dataset.get_client_dic()
            # self.client_idcs = [[] for _ in range(client_num)]
            # for i in range(client_num):
            #     self.client_idcs[i] = list(client_idcs[i])

            for idcs in self.client_idcs:
                a = train_labels[idcs]
                distribution={}
                for label in a:
                        if label not in distribution:
                            distribution[label]=1
                        else:
                            distribution[label]+=1
                print("Label Distribution : {}".format(distribution))

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        print("The shape of shakespeare train data: {}".format(self.train_dataset.data.shape))
        print("The shape of shakespeare test data: {}".format(self.test_dataset.data.shape))

    def get_agnews_dataset(self, client_num=10, class_num=4, data_dir="./data/AGNEWS/", bert_dir="./models/bert-base-uncased/"):
        train_dataset = AgnewsDataset(train=True, data_dir=data_dir, bert_dir=bert_dir)
        test_dataset = AgnewsDataset(train=False, data_dir=data_dir, bert_dir=bert_dir)

        self.train_data_size = train_dataset.labels.shape[0]
        self.test_data_size = test_dataset.labels.shape[0]

        if self.isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            train_dataset.ids = train_dataset.ids[order]
            train_dataset.masks = train_dataset.masks[order]
            train_dataset.labels = train_dataset.labels[order]

        else:
            label_distribution = np.random.dirichlet([self.beta] * client_num, class_num)
            class_idcs = [np.argwhere(train_dataset.labels == y).flatten() for y in range(class_num)]

            client_idcs = [[] for _ in range(client_num)]
            for c, fracs in zip(class_idcs, label_distribution):
                for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                    client_idcs[i] += [idcs]

            client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
            self.client_idcs = client_idcs

            for idcs in client_idcs:
                a = train_dataset.labels[idcs]
                distribution = {}
                for label in a:
                    if label not in distribution:
                        distribution[label] = 1
                    else:
                        distribution[label] += 1
                print("Label Distribution : {}".format(distribution))

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        print("The shape of agnews train data: {}".format(self.train_dataset.ids.shape))
        print("The shape of agnews train label: {}".format(self.train_dataset.labels.shape))
        print("The shape of agnews test data: {}".format(self.test_dataset.ids.shape))

    def get_cifar100_dataset(self, client_num=10, class_num=100, data_dir="./data/CIFAR100/"):
        transform_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                          transform=transform_train)
        test_dataset = datasets.CIFAR100(data_dir, train=False, transform=transform_test)

        train_data = train_dataset.data
        train_labels = np.array(train_dataset.targets)
        test_data = test_dataset.data
        test_labels = np.array(test_dataset.targets)
        test_dataset.targets = test_labels

        self.train_data_size = train_data.shape[0]
        self.test_data_size = test_data.shape[0]

        if self.isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            train_dataset.data = train_data[order]
            train_dataset.targets = train_labels[order]

        else:
            # order = np.argsort(train_labels)
            #
            # distribution = {}
            # for label in order:
            #     if train_labels[label] not in distribution:
            #         distribution[train_labels[label]] = 1
            #     else:
            #         distribution[train_labels[label]] += 1
            # print("Label Distribution : {}".format(distribution))
            # train_dataset.data = train_data[order]
            # train_dataset.targets = train_labels[order]

            train_dataset.data = train_data
            train_dataset.targets = train_labels

            label_distribution = np.random.dirichlet([self.beta]*client_num, class_num)
            class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(class_num)]

            client_idcs = [[] for _ in range(client_num)]
            for c, fracs in zip(class_idcs, label_distribution):
                for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                    client_idcs[i] += [idcs]

            client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
            self.client_idcs = client_idcs


            for idcs in client_idcs:
                a=train_labels[idcs]
                distribution={}
                for label in a:
                        if label not in distribution:
                            distribution[label]=1
                        else:
                            distribution[label]+=1
                print("Label Distribution : {}".format(distribution))

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        print("The shape of cifar100 train data: {}".format(self.train_dataset.data.shape))
        print("The shape of cifar100 train label: {}".format(self.train_dataset.targets.shape))
        print("The shape of cifar100 test data: {}".format(self.test_dataset.data.shape))

    def get_imagenet_dataset(self, client_num=10, class_num=200, data_dir="./data/tiny-imagenet-200/", load_from_numpy=True):
        train_dataset = TinyImageNet(data_dir, train=True, load_from_numpy=load_from_numpy)
        test_dataset = TinyImageNet(data_dir, train=False, load_from_numpy=load_from_numpy)

        train_data = train_dataset.images
        train_labels = train_dataset.labels

        self.train_data_size = train_dataset.images.shape[0]
        self.test_data_size = test_dataset.images.shape[0]
        #
        if self.isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            train_dataset.images = train_data[order]
            train_dataset.labels = train_labels[order]

        else:
            label_distribution = np.random.dirichlet([self.beta]*client_num, class_num)
            class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(class_num)]

            client_idcs = [[] for _ in range(client_num)]
            for c, fracs in zip(class_idcs, label_distribution):
                for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                    client_idcs[i] += [idcs]

            client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
            self.client_idcs = client_idcs

            for idcs in client_idcs:
                a=train_labels[idcs]
                distribution={}
                for label in a:
                        if label not in distribution:
                            distribution[label]=1
                        else:
                            distribution[label]+=1
                print("Label Distribution : {}".format(distribution))

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        print("The shape of imagenet train data: {}".format(self.train_dataset.images.shape))
        print("The shape of imagenet train label: {}".format(self.train_dataset.labels.shape))
        print("The shape of imagenet test data: {}".format(self.test_dataset.images.shape))

    def get_public_share_dataset(self, frac=0.15):
        if self.train_dataset is None:
            return

        order = np.random.choice(range(self.train_data_size), int(self.train_data_size * frac))
        return torch.utils.data.Subset(self.train_dataset, order)


class Sent140(Dataset):
    def __init__(self, train=True, client_nums=4000):
        train_clients, train_groups, train_data_temp, test_data_temp = self.read_data("./data/sent140/train", "./data/sent140/test")
        self.train = train

        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                if i == client_nums:
                    break
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']

                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(cur_x[j][4])
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y

        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                if i == client_nums:
                    break
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(cur_x[j][4])
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index], self.label[index]
        indices = tokenizer(sentence, padding="max_length", max_length=64, truncation=True)['input_ids']

        # y = indices[1:].append(target)
        # target = indices[1:].append(target)
        indices = torch.LongTensor(np.array(indices))
        # y = torch.Tensor(np.array(y))
        # target = torch.LongTensor(np.array(target))
        return indices, target

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")

    def read_dir(self, data_dir):
        clients = []
        groups = []
        data = defaultdict(lambda: None)

        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]
        for f in files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            data.update(cdata['user_data'])

        clients = list(sorted(data.keys()))
        return clients, groups, data

    def read_data(self, train_data_dir, test_data_dir):
        '''parses data in given train and test data directories
        assumes:
        - the data in the input directories are .json files with
            keys 'users' and 'user_data'
        - the set of train set users is the same as the set of test set users
        Return:
            clients: list of client ids
            groups: list of group ids; empty list if none found
            train_data: dictionary of train data
            test_data: dictionary of test data
        '''
        train_clients, train_groups, train_data = self.read_dir(train_data_dir)
        test_clients, test_groups, test_data = self.read_dir(test_data_dir)

        assert train_clients == test_clients
        assert train_groups == test_groups

        return train_clients, train_groups, train_data, test_data




if __name__ == "__main__":
    #mnist_dataset = GetDataSet("fashion_mnist", is_iid=False, beta=0.3)
    #mnist_dataset = GetDataSet("mnist", is_iid=False)
    #train_loader = torch.utils.data.DataLoader(mnist_dataset.test_dataset, batch_size=64)
    # for i, (images, labels) in enumerate(train_loader):
    #     if (i + 1) % 100 == 0:
    #         for j in range(len(images)):
    #             image = images[j].resize(28, 28)  # 将(1,28,28)->(28,28)
    #             plt.imshow(image)  # 显示图片,接受tensors, numpy arrays, numbers, dicts or lists
    #             plt.axis('off')  # 不显示坐标轴
    #             plt.title("$The {} picture in {} batch, label={}$".format(j + 1, i + 1, labels[j]))
    #             plt.show()

    cifar_dataset = GetDataSet("cifar10", is_iid=False, beta=0.3)
    # print(cifar_dataset.train_dataset)
    # train_loader = torch.utils.data.DataLoader(cifar_dataset.train_dataset, batch_size=64)
    # for image,label in  train_loader:
    #     img = image[0]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
    #     img = img.numpy()  # FloatTensor转为ndarray
    #     print(img.shape)
    #     img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
    #     # 显示图片
    #     plt.imshow(img)
    #     plt.show()

    #agnews_dataset = GetDataSet("agnews", is_iid=False, beta=0.3)

    #imagenet_dataset = GetDataSet("imagenet", is_iid=False, beta=0.3)
    # data_dir = './data/tiny-imagenet-200/'
    # dataset_train = TinyImageNet(data_dir, train=True, load_from_numpy=True)
    # dataset_val = TinyImageNet(data_dir, train=False, load_from_numpy=True)
    #sample, target = dataset_train.__getitem__(4)
    # print(target)
    # print(sample.shape)
    # plt.imshow(sample)
    # plt.axis('off')
    # plt.show()
    # print(sample.shape)
    # sample0, target0 = dataset_train.__getitem__(0)
    # sample1, target1 = dataset_train.__getitem__(1)
    # a = np.array([sample0, sample1])
    # print(a.shape)

    #shakespeare_dataset = GetDataSet("shakespeare", is_iid=False)

    # sent140_dataset = GetDataSet("sent140", is_iid=False)


    #cora_dataset = GetDataSet("cora", is_iid=False)
    citeseer_dataset = GetDataSet("citeseer", is_iid=False)


    # sent140_dataset = GetDataSet(dataset_name="sent140", is_iid=False)
    # #subset = torch.utils.data.Subset(sent140_dataset.train_dataset, sent140_dataset.client_idcs[-1])
    # train_dataloader = torch.utils.data.DataLoader(sent140_dataset.train_dataset, batch_size=32, shuffle=True)
    # test_dataloader = torch.utils.data.DataLoader(sent140_dataset.test_dataset, batch_size=32, shuffle=False)
    #
    # device = torch.device("cuda:0")
    # model = models.RNN(input_dim=50000, embedding_dim=256, hidden_dim=64, output_dim=2)
    # model.to(device)
    #
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #
    # criterion = torch.nn.CrossEntropyLoss()
    #
    # for e in range(3):
    #  total_loss = 0
    #  model = model.train()
    #  for batch_id, batch in enumerate(tqdm(train_dataloader)):
    #     data, target = batch
    #     data = data.to(device)
    #     target = target.to(device, dtype=torch.long)
    #
    #     optimizer.zero_grad()
    #     output = model(data)
    #
    #     loss = criterion(output, target)
    #     loss.backward()
    #     total_loss += loss.item()
    #     optimizer.step()
    #     print(total_loss)

    # subset = torch.utils.data.Subset(train_dataset, [4,5,6,7])
    # print(subset.__getitem__(0))
    # print(train_dataset.__getitem__(4))


