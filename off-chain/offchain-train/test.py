import random
import time
import traceback
import uuid
import logging
import models, torch
from torch import nn
import os
import re
import numpy as np
import json
import argparse
import copy
import rsa
import base64
import functools
import pika
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry
import requests
import threading
from threading import Timer
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from concurrent.futures import ThreadPoolExecutor, as_completed
from client import *
from server import *
import redis
from models import GCN
import torch.nn.functional as F
import dgl.data

def deleteAllModelFiles():
    path = "./models"
    for root, dir, files in os.walk(path):
        for file in files:
            if re.match(r"-\d{2}|\d{2}", file):
                model_file = os.path.join(root, file)
                print(model_file)
                os.remove(model_file)

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def sparsity_similarity(v1, v2):
    temp = np.sum(np.square(v2 - v1))
    similarity = 1.0 / (np.sqrt(temp)) if temp != 0 else 1.0
    return similarity

def calSimilarityMatrix(vlist):
    n = len(vlist)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity_matrix[i][j] = sparsity_similarity(vlist[i], vlist[j])
    # print(similarity_matrix)
    l = similarity_matrix[0]
    v = np.array([])
    for value in l:
        if value != 1.0:
            v = np.append(v, value)
    v = softmax(v)
    print(v)

    t = 0
    for i in range(len(l)):
        if l[i] == 1.0:
            continue
        l[i] = v[t]
        t = t + 1
    # print(l)
    return similarity_matrix

class mqclient(object):
    def __init__(self):
        self.credentials = pika.PlainCredentials("fedchain", "061910")
        self.connect = pika.BlockingConnection(
            pika.ConnectionParameters(host='114.212.83.207', port=5672, virtual_host='/', credentials=self.credentials))
        self.channel = self.connect.channel()

        result = self.channel.queue_declare(queue="", exclusive=True)
        self.callback_queue = result.method.queue
        self.response = {}
        executor.submit(self.run)

    def run(self):
        self.channel.basic_consume(on_message_callback=self.on_response, queue=self.callback_queue)
        self.channel.start_consuming()


    def grant_access_request_by_mq(self, msg):
        start = time.time()
        corr_id = str(uuid.uuid4())
        print("send msg {}".format(msg))
        self.channel.basic_publish(exchange='', routing_key="requestQueue", properties=pika.BasicProperties(reply_to=self.callback_queue, correlation_id=corr_id), body=msg)
        while corr_id not in self.response:
            time.sleep(0.1)
        end = time.time()
        grant_result = self.response[corr_id]
        del self.response[corr_id]
        print("response is {}, latency: {}s".format(grant_result, end-start))
        return grant_result

    def grant_access_request_by_chaincode(self, msg):
        start = time.time()
        grant_result = session.post(org_server + grant_access_api, data=msg, headers=headers).content
        grant_result = json.loads(grant_result)
        end = time.time()
        print("response is {}, latency: {}s".format(grant_result, end-start))
        return grant_result

    def on_response(self, ch, method, properties, body):
        self.response[properties.correlation_id] = body
        self.channel.basic_ack(delivery_tag=method.delivery_tag)

class mqserver(object):
    def __init__(self):
        self.credentials = pika.PlainCredentials("fedchain", "061910")
        self.connect = pika.BlockingConnection(pika.ConnectionParameters(
            host='114.212.83.207', port=5672, virtual_host='/', credentials=self.credentials))
        self.channel = self.connect.channel()
        self.channel.queue_declare(queue='requestQueue')

        self.timer = None
        self.timeout = 5
        self.batch_size = 10
        #self.channel.basic_qos(prefetch_count=0)
        self.msglist = {}
        self.message_count = 0
        self.executor = ThreadPoolExecutor()

        self.channel.basic_consume(queue="requestQueue", on_message_callback=self.on_message)
        self.channel.start_consuming()

    def on_message(self, ch, method, props, body):

        if self.timer and self.timer.isAlive():
            self.timer.cancel()

        msg = json.loads(body)
        if msg["requestee"] not in self.msglist:
            self.msglist[msg["requestee"]] = []
        self.msglist[msg["requestee"]].append([msg, props, method])
        self.message_count += 1

        if self.message_count >= self.batch_size:
            tasks = []
            for key in self.msglist:
                tasks.append(self.executor.submit(self.batch_process, self.msglist[key]))

            for _ in as_completed(tasks):
                try:
                    pass
                except Exception as e:
                    print(traceback.format_exc())
            self.message_count = 0

        elif len(self.msglist) > 0:
            self.timer = Timer(self.timeout, mqserver.timeout_process, (self,))
            self.timer.start()

        # print("cousume msg {}".format(json.loads(body)))
        # self.channel.basic_publish(exchange='', routing_key=props.reply_to,
        #                  properties=pika.BasicProperties(correlation_id=props.correlation_id), body="")
        # self.channel.basic_ack(delivery_tag=method.delivery_tag)

    def batch_process(self, messages):
        # if self.timer and self.timer.isAlive():
        #     self.timer.cancel()

        print("Batch consume {} messages".format(len(messages)))
        if len(messages) == 0:
            return

        requestee = messages[0][0]["requestee"]
        subject_ids = []
        orgs = []
        requestTimes = []
        for i in range(len(messages)):
            body, _, _ = messages[i]
            requestee = body["requestee"]
            subject_ids.append(body["subject_id"])
            orgs.append(body["org"])
            requestTimes.append(body["requestTime"])

        data = {
            "requestee": requestee,
            "subject_ids": subject_ids,
            "orgs": orgs,
            "requestTimes": requestTimes
        }
        grant_results = session.post(org_server + grant_group_access_api, data=json.dumps(data),
                                     headers=headers).content
        grant_results = json.loads(grant_results)
        print(grant_results)
        for i in range(len(messages)):
            _, props, method = messages[i]
            cb = functools.partial(self.ack_message, exchange="", props=props, method=method, body=json.dumps(grant_results))
            self.connect.add_callback_threadsafe(cb)

        self.msglist[requestee] = []

    def ack_message(self, exchange=None, props=None, method=None, body=None):
        self.channel.basic_publish(
            exchange=exchange,
            routing_key=props.reply_to,
            properties=pika.BasicProperties(correlation_id=props.correlation_id),
            body=body)
        self.channel.basic_ack(method.delivery_tag)

    def timeout_process(self):
        print("Time out for batch process")
        tasks = []
        for key in self.msglist:
            tasks.append(self.executor.submit(self.batch_process, self.msglist[key]))

        for _ in as_completed(tasks):
            try:
                pass
            except Exception as e:
                print(traceback.format_exc())
        self.message_count = 0

def start_mqserver():
    server = mqserver()


if __name__ == '__main__':
    deleteAllModelFiles()

    # vlist = [[0.1, 0.7], [0.15, 0.6], [0.5, 0.5]]
    # vlist = np.array(vlist)
    # similarity_matrix = calSimilarityMatrix(vlist)
    # print(similarity_matrix)
    # labels = SpectralClustering(n_clusters=2, affinity='precomputed').fit_predict(similarity_matrix)
    # print(labels)

    # similarity_matrix = np.array([[1, 0.3, 0.2], [0.3, 1, 0.9], [0.2, 0.9, 1]])
    # labels = SpectralClustering(n_clusters=2, affinity='precomputed').fit_predict(similarity_matrix)
    # print(labels)

    # distance_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # cluster_labels = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='complete').fit_predict(distance_matrix)
    # for i in range(2):
    #     cluster = np.where(cluster_labels == i)[0]
    #     print(cluster)

    # from scipy.spatial.distance import squareform
    # from scipy.cluster.hierarchy import linkage,dendrogram
    #
    # distance_matrix = np.array([[0, 0.3, 0.2], [0.3, 0, 0.11], [0.2, 0.11, 0]])
    # condensed_dist = squareform(distance_matrix)
    # linkresult = linkage(condensed_dist)
    # print(linkresult)

    # os.environ["DGL_DOWNLOAD_DIR"] = "./data"
    # cora_dataset = dgl.data.CoraGraphDataset()
    # g = cora_dataset[0]
    #
    # sg1 = g.subgraph([0, 2, 3])
    # print(sg1.ndata['feat'].shape)

    # model = GCN(g.ndata['feat'].shape[1], 16, cora_dataset.num_classes)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    #
    # features = g.ndata['feat']
    # labels = g.ndata['label']
    # train_mask = g.ndata['train_mask']
    # val_mask = g.ndata['val_mask']
    # test_mask = g.ndata['test_mask']
    #
    # for epoch in range(100):
    #     model.train()
    #     optimizer.zero_grad()
    #
    #     logits = model(g, features)
    #     pred = logits.argmax(1)
    #     loss = loss_fn(logits[train_mask], labels[train_mask])
    #
    #     loss.backward()
    #     optimizer.step()
    #
    #     train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
    #     val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
    #     test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
    #     print("{}, {}, {}".format(train_acc, val_acc, test_acc))


    # session = requests.Session()
    # session.mount('https://', HTTPAdapter(max_retries=Retry(total=5, allowed_methods=frozenset(['GET', 'POST']))))
    # org_server = "http://114.212.82.242:8080/"
    # grant_group_access_api = "grantGroupAccess"
    # grant_access_api = "grantAccess"
    # headers = {'content-type': 'application/json',
    #            'User-Agent': 'User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
    #
    # executor = ThreadPoolExecutor(max_workers=30)
    # task_list = []
    # task_list.append(executor.submit(start_mqserver))
    #
    # list = []
    # threads = 10
    # for i in range(threads):
    #     list.append(mqclient())
    #
    # requestees = ["000", "001", "002"]
    # for i in range(threads):
    #     data = {
    #         "requestee": random.choice(requestees),
    #         "subject_id": str(i),
    #         "org": "org"+str(i),
    #         "requestTime": int(time.time())
    #     }
    #
    #     task_list.append(executor.submit(list[i].grant_access_request_by_mq, json.dumps(data)))
    #     #executor.submit(list[i].grant_access_request_by_chaincode, json.dumps(data))
    #
    # for task in as_completed(task_list):
    #     try:
    #         pass
    #     except Exception as e:
    #         print(traceback.format_exc())

    # executor = ThreadPoolExecutor()
    # executor.submit(test)



