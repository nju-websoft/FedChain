import os
import random
import argparse
import time
import traceback

import schedule
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
#import threading
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry

import plot
from client import *
from threading import Timer
import uuid
import redis
from redis import WatchError

class ResettableTimer(object):
    def __init__(self, interval, function):
        self.interval = interval
        self.function = function
        self.timer = Timer(self.interval, self.function)

    def start(self):
        self.timer.start()

    def reset(self):
        self.timer.cancel()
        self.timer = Timer(self.interval, self.function)
        self.timer.start()

    def cancel(self):
        self.timer.cancel()

def pull_global_model(client):

    start = time.time()
    while True:
      global_model_info = session.get(org_server + get_global_api, headers=headers).content
      global_model_info = json.loads(global_model_info)

      if global_model_info["round"] >= client.round:
          client.round = global_model_info["round"]
          client.load_model("./models/server/" +global_model_info["curHashId"] + ".pth")
          break

      time.sleep(1)

    end = time.time()
    return end - start

def local_train(client):
    start = time.time()
    exchange_prob = random.random()
    if exchange_prob > conf["exchange_probability"]:
        try:
            client.local_train(client.local_model, client.round)
        except Exception as e:
            print(traceback.format_exc())
            pass

    else:
        print("client {} request other local models by blockchain".format(client.client_id))
        data = {
            "org": client.client_id,
            "loss_list": client.loss_list,
            "threshold": conf["threshold"]
        }

        collaborative_info = session.post(org_server + collaborative_api, data=json.dumps(data), headers=headers).content
        collaborative_info = json.loads(collaborative_info)
        print("Collaborative clients are {}".format(collaborative_info["collaborativeClients"]))

        acc1, _ = client.eval_model()
        print("Before : client {} valid acc {}".format(client.client_id, acc1))

        models = [[id, torch.load(url, map_location=client.device)] for id, url in
                  zip(collaborative_info["collaborativeClients"], collaborative_info["modelUrls"])]
        try:
            client.fuse_model_by_teachers(models, client.round)
            acc2, _ = client.eval_model()
        except Exception as e:
            print(traceback.format_exc())
            pass
        print("After : client {} valid acc {}".format(client.client_id, acc2))

    cur_hash_id = str(hash(frozenset(client.local_model.state_dict().values())))
    client.save_model(model_name=cur_hash_id)

    end = time.time()
    data = {
        "org": "org" + str(client.client_id),
        "cur_hash_id": cur_hash_id,
        "model_url": "./models/clients/client" + str(client.client_id) + "/" + cur_hash_id + '.pth',
        "round": client.round,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(round(time.time() * 1000)) / 1000)),
        "traintime": end-start,
        "sparse_vector": client.extractor.mean_list[0].tolist()
    }

    response = session.post(org_server + upload_local_api, data=json.dumps(data), headers=headers).content
    print("Response:{} at time {}".format(response, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(
        int(round(time.time() * 1000)) / 1000))))


def pull_cluster_model(client):
    global local_cluster_info_cache
    start = time.time()
    while True:

        if not local_cluster_info_cache:
            print("{}: local cluster info cache is empty; waiting...".format(pull_cluster_model.__name__))
            time.sleep(1)
            continue

        cluster_info = session.get(org_server + get_cluster_api, headers=headers, params=local_cluster_info_cache).content
        cluster_info = json.loads(cluster_info)
        if (cluster_info["cluster"] == local_cluster_info_cache["cluster"] and cluster_info["round"] <= local_cluster_info_cache["round"]):
            print("Cluster still not update: cluster_info:{}; local_cache:{}".format(cluster_info,local_cluster_info_cache))
            time.sleep(1)
            continue

        print("Cluster model has been updated; client pull cluster model from {} to update local model".format(cluster_info["cluster"]))
        client.round = cluster_info["round"]
        client.load_model(cluster_info["modelUrl"])
        local_cluster_info_cache = {}

        end = time.time()
        idle_time = end - start
        print("Idle Time between round {} and round {} is {}s".format(client.round, client.round-1, idle_time))

        print()
        break

    return idle_time


def monitor_cluster_change(client):
    global local_cluster_info_cache, local_global_info_cache
    belong_to_cluster = -1

    while True:
        while True:
            try:
                global_model_info = session.get(org_server + get_global_api, headers=headers).content
            except requests.exceptions.ConnectionError:
                print(traceback.format_exc())
                time.sleep(1)
                continue

            global_model_info = json.loads(global_model_info)

            if local_global_info_cache and local_global_info_cache["round"] < global_model_info["round"]:
                print("{}: global_info:{}; local_global_info_cache:{}".format(monitor_cluster_change.__name__,
                                                                              global_model_info["round"],
                                                                              local_global_info_cache["round"]))

                local_global_info_cache = global_model_info
                clusters = global_model_info["clusters"]
                for index, cluster in enumerate(clusters):
                    if client.client_id in cluster:
                        belong_to_cluster = index
                        print("{}: now belong to cluster {}".format(monitor_cluster_change.__name__, belong_to_cluster))
                        break
                break
            time.sleep(1)

        print("{}: wait for last round client finish pull cluster model for update".format(monitor_cluster_change.__name__))
        while local_cluster_info_cache:
            time.sleep(1)
            continue
        print("{}: finish waiting".format(monitor_cluster_change.__name__))

        cluster_info = session.get(org_server + get_cluster_api, headers=headers, params={"cluster": "cluster"+str(belong_to_cluster)}).content
        local_cluster_info_cache = json.loads(cluster_info)
        print("{}: {}".format(monitor_cluster_change.__name__, local_cluster_info_cache))

def server_thread(server):

    def cluster_aggregate():
        clusters = global_model_info["clusters"]
        for i in range(len(clusters)):

            update_ids = [int(key[3:]) for key in local_updates.keys()]
            cluster_ids = list(set(update_ids).intersection(set(clusters[i])))
            if len(cluster_ids) == 0 or cluster_ids[0] != client_id:
                continue

            cluster_info = session.get(org_server + get_cluster_api, headers=headers,
                                       params={"cluster": "cluster" + str(i)}).content
            cluster_info = json.loads(cluster_info)

            if cluster_info["round"] == global_model_info["round"] + 1:
                continue

            print("Cluster leader {} pull local models from {} to aggregate cluster {} model".format(client_id,
                                                                                                     cluster_ids, i))
            local_models = [
                torch.load(local_updates["org" + str(id)]["localModelBlock"]["modelUrl"], map_location=server.device)
                for id in cluster_ids]

            server.model_weight_aggregate(local_models)
            server.save_model(model_name="cluster" + str(i))
            data = {
                "cluster": "cluster" + str(i),
                "cluster_model_url": "./models/server/cluster" + str(i) + ".pth",
                "round": global_model_info["round"] + 1
            }

            try:
                response = session.post(org_server + upload_cluster_api, data=json.dumps(data), headers=headers)
                print(response.content)
            except requests.exceptions.RequestException as e:
                print(traceback.format_exc())

            break

    def timeout_aggregate():
        print("Timeout for starting aggregation")
        cluster_aggregate()
        old_local_updates.clear()

    old_local_updates = json.loads("{}")

    timer = ResettableTimer(interval=conf["timeout"], function=timeout_aggregate)
    timer.start()
    while True:
        local_updates = session.get(org_server + get_local_updates_api, headers=headers).content
        local_updates = json.loads(local_updates)
        #print(local_updates.keys())

        global_model_info = session.get(org_server + get_global_api, headers=headers).content
        global_model_info = json.loads(global_model_info)
        server.round = global_model_info["round"]
        #print(global_model_info["clusters"])

        if len(old_local_updates.keys()) < len(local_updates.keys()):
            timer.reset()
        old_local_updates = local_updates

        if len(local_updates) >= global_model_info["triggerAvgNum"]:
            # clusters = global_model_info["clusters"]
            # for i in range(len(clusters)):
            #
            #     update_ids = [int(key[3:]) for key in local_updates.keys()]
            #     cluster_ids = list(set(update_ids).intersection(set(clusters[i])))
            #     if len(cluster_ids) == 0 or cluster_ids[0] != client_id:
            #         continue
            #
            #     cluster_info = session.get(org_server + get_cluster_api, headers=headers, params={"cluster": "cluster"+str(i)}).content
            #     cluster_info = json.loads(cluster_info)
            #
            #     if cluster_info["round"] == global_model_info["round"] + 1:
            #         continue
            #
            #     print("Cluster leader {} pull local models from {} to aggregate cluster {} model".format(client_id, cluster_ids, i))
            #     local_models = [torch.load(local_updates["org"+str(id)]["localModelBlock"]["modelUrl"], map_location=server.device) for id in cluster_ids]
            #
            #     server.model_weight_aggregate(local_models)
            #     server.save_model(model_name="cluster"+str(i))
            #     data = {
            #         "cluster": "cluster" + str(i),
            #         "cluster_model_url": "./models/server/cluster" + str(i) + ".pth",
            #         "round": global_model_info["round"] + 1
            #     }
            #
            #     try:
            #         response = session.post(org_server + upload_cluster_api, data=json.dumps(data), headers=headers)
            #         print(response.content)
            #     except requests.exceptions.RequestException as e:
            #         print(traceback.format_exc())
            #     break

            timer.cancel()
            cluster_aggregate()
            old_local_updates.clear()

        time.sleep(1+random.random())

def client_thread(client):
    global local_global_info_cache, local_cluster_info_cache
    belong_to_cluster = -1

    global_model_info = session.get(org_server + get_global_api, headers=headers).content
    local_global_info_cache = json.loads(global_model_info)
    client.round = local_global_info_cache["round"]

    for index, cluster in enumerate(local_global_info_cache["clusters"]):
        if client.client_id in cluster:
            belong_to_cluster = index
            print("{}: Initial belong to cluster {}".format(monitor_cluster_change.__name__, belong_to_cluster))
            break
    cluster_info = session.get(org_server + get_cluster_api, headers=headers, params={"cluster": "cluster"+str(belong_to_cluster)}).content
    local_cluster_info_cache = json.loads(cluster_info)

    idle_time_list = []
    while True:
        local_train(client)
        idle_time = pull_cluster_model(client)

        idle_time_list.append(idle_time)
        plot.save_idletime(idle_time_list, client.client_id, suffix="fedchain")

        if client.round >= conf["global_epochs"]:
            break

    session.close()

def fedavg(client):
    idle_time_list = []
    while True:
        local_train(client)
        idle_time = pull_global_model(client)

        idle_time_list.append(idle_time)
        plot.save_idletime(idle_time_list, client.client_id, suffix="fedavg")

        if client.round >= conf["global_epochs"]:
            break


if __name__ == '__main__':
    if torch.cuda.is_available():
        print(torch.version.cuda)
        print(torch.__version__)

    with open("config/parallel-conf.json", 'r') as f:
        conf = json.load(f)
    print(conf)

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-client_id', dest='client_id')
    args = parser.parse_args()
    if args.client_id is not None:
        conf["client_id"] = int(args.client_id)
        print(conf["client_id"])

    client_id = conf["client_id"]
    dataset = dataset.GetDataSet(dataset_name=conf["dataset"], is_iid=conf["iid"], beta=conf["niid_beta"], client_num=conf["client_num"])
    subset = torch.utils.data.Subset(dataset.train_dataset, dataset.client_idcs[client_id])
    client_indices = list(range(conf["client_num"]))

    redis_client = redis.Redis(host="114.212.82.242", port=6379, decode_responses=True)
    session = requests.Session()
    #session.mount('https://', HTTPAdapter(max_retries=Retry(total=5, method_whitelist=frozenset(['GET', 'POST']))))
    org_server = "http://114.212.82.242:8080/"

    upload_local_api = "updateLocal"
    upload_global_api = "updateGlobal"
    upload_cluster_api = "updateClusterModel"

    get_global_api = "getGlobalModelMetaInfo"
    get_cluster_api = "getClusterModelMetaInfo"
    get_local_updates_api = "scanLocalUpdates"
    get_cluster_updates_api = "scanClusterUpdates"
    collaborative_api = "collaborative"
    headers = {'content-type': 'application/json',
               'User-Agent': 'User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}

    client = Client(conf=conf, train_dataset=subset, id=client_id, device=torch.device("cuda:"+str(client_id % 4)))
    server = Server(conf, device=torch.device("cuda:" + str(client_id % 4)))

    local_cluster_info_cache = {}
    local_global_info_cache = {}

    executor = ThreadPoolExecutor()
    task_list = []
    task_list.append(executor.submit(monitor_cluster_change, client))
    task_list.append(executor.submit(client_thread, client))
    task_list.append(executor.submit(server_thread, server))
    for task in as_completed(task_list):
        try:
            print(task.result())
        except Exception as e:
            print(traceback.format_exc())

    #server_thread(server)

    #fedavg(client)





