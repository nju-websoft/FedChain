import random
import time
import datetime
import numpy as np
import schedule
import requests
import json
from client import *
from server import *
import plot
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED, wait
from sklearn.cluster import AgglomerativeClustering
import uuid
import redis
from redis import WatchError
from threading import Thread

def fedasync():
    rounds = []
    accuracy = []
    # validloss = []
    times = []
    total_time_cost = 0
    start = time.time()

    while True:
      local_updates = session.get(org_server + get_local_updates_api, headers=headers).content
      local_updates = json.loads(local_updates)
      print(local_updates.keys())

      global_model_info = session.get(org_server + get_global_api, headers=headers).content
      global_model_info = json.loads(global_model_info)
      server.round = global_model_info["round"]
      print("global round: {}; clusters: {}".format(global_model_info["round"], global_model_info["clusters"]))

      if len(local_updates) >= global_model_info["triggerAvgNum"]:
            candidates = random.sample(local_updates.keys(), 1)
            print("Start model aggregate with {}".format(candidates))
            models = [torch.load(local_updates[candidate]["localModelBlock"]["modelUrl"], map_location=server.device) for candidate in candidates]
            server.model_weight_aggregate(models)

            end = time.time()
            total_time_cost += (end - start)

            acc, loss = server.model_eval()
            print("Round {}, global model acc: {}, loss: {}, time consume:{}".format(server.round, acc, loss, total_time_cost))

            cur_hash_id = str(hash(frozenset(server.global_model.state_dict().values())))
            server.save_model(model_name=cur_hash_id)
            clusters = [[]]
            data = {
              "cur_hash_id": cur_hash_id,
              "cur_model_url": "./models/server/" + cur_hash_id + ".pth",
              "clusters": clusters,
              "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            try:
              identifier = acquire_lock(lock_name="global")
              response = session.post(org_server + upload_global_api, data=json.dumps(data), headers=headers)
              release_lock(identifier, lock_name="global")
              print(response.content)
            except requests.exceptions.RequestException as e:
              print("Error is {}".format(e))

            start = time.time()
            rounds.append(server.round)
            accuracy.append(acc)
            times.append(total_time_cost)
            plot.save_array(rounds, accuracy, times, dir=save_dir, name=save_name)
            plot.plot(rounds, accuracy, label1="accuracy", dir=save_dir, name=save_name)
            candidates = random.sample(range(conf["client_num"]), int(conf["client_num"] * 0.4))
            #candidates = random.sample(range(5), 3)

      if server.round > 100:
          break
      time.sleep(1)



def fedavg():
    rounds = []
    accuracy = []
    # validloss = []
    times = []
    total_time_cost = 0
    start = time.time()
    candidates = random.sample(range(conf["client_num"]), int(conf["client_num"] * 0.4))
    #candidates = random.sample(range(5), 3)
    while True:
      local_updates = session.get(org_server + get_local_updates_api, headers=headers).content
      local_updates = json.loads(local_updates)
      print(local_updates.keys())

      global_model_info = session.get(org_server + get_global_api, headers=headers).content
      global_model_info = json.loads(global_model_info)
      server.round = global_model_info["round"]
      print("global round: {}; clusters: {}".format(global_model_info["round"], global_model_info["clusters"]))

      aggre = True

      for c in candidates:
          if "org" + str(c) not in list(local_updates.keys()):
              aggre = False
              break

      if aggre:
        print("Start model aggregate with {}".format(candidates))
        models = [torch.load(local_updates["org"+str(candidate)]["localModelBlock"]["modelUrl"], map_location=server.device) for candidate in candidates]
        server.model_weight_aggregate(models)

        end = time.time()
        total_time_cost += (end - start)

        acc, loss = server.model_eval()
        print("Round {}, global model acc: {}, loss: {}, time consume:{}".format(server.round, acc, loss, total_time_cost))

        cur_hash_id = str(hash(frozenset(server.global_model.state_dict().values())))
        server.save_model(model_name=cur_hash_id)
        clusters = [[]]
        data = {
          "cur_hash_id": cur_hash_id,
          "cur_model_url": "./models/server/" + cur_hash_id + ".pth",
          "clusters": clusters,
          "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        try:
          identifier = acquire_lock(lock_name="global")
          response = session.post(org_server + upload_global_api, data=json.dumps(data), headers=headers)
          release_lock(identifier, lock_name="global")
          print(response.content)
        except requests.exceptions.RequestException as e:
          print("Error is {}".format(e))

        start = time.time()
        rounds.append(server.round)
        accuracy.append(acc)
        times.append(total_time_cost)
        plot.save_array(rounds, accuracy, times, dir=save_dir, name=save_name)
        plot.plot(rounds, accuracy, label1="accuracy", dir=save_dir, name=save_name)
        candidates = random.sample(range(conf["client_num"]), int(conf["client_num"] * 0.4))
        #candidates = random.sample(range(5), 3)

      if server.round > 100:
          break
      time.sleep(1)


def acquire_lock(lock_name, acquire_time=100, time_out=10):
    identifier = str(uuid.uuid4())
    end = time.time() + acquire_time
    while time.time() < end:
        if redis_client.set(lock_name, identifier, ex=time_out, nx=True):
            return identifier
        time.sleep(0.5)
    return False

def release_lock(identifier, lock_name):
    with redis_client.pipeline() as pipe:
        while True:
            try:
                pipe.watch(lock_name)
                lock_value = pipe.get(lock_name)
                if lock_value and lock_value == identifier:
                    pipe.multi()
                    pipe.delete(lock_name)
                    pipe.execute()
                    return True

                pipe.unwatch()
                break
            except WatchError:
                pass
        return False

def check_cluster_aggregate():
    local_updates = session.get(org_server + get_local_updates_api, headers=headers).content
    local_updates = json.loads(local_updates)
    print(local_updates.keys())

    cluster_updates = session.get(org_server + get_cluster_updates_api, headers=headers).content
    cluster_updates = json.loads(cluster_updates)
    print(cluster_updates)

    global_model_info = session.get(org_server + get_global_api, headers=headers).content
    global_model_info = json.loads(global_model_info)
    server.round = global_model_info["round"]
    print("global round: {}; clusters: {}".format(global_model_info["round"], global_model_info["clusters"]))

    clusters = global_model_info["clusters"]
    orgs = [int(key[3:]) for key in local_updates.keys()]

    not_empty_count = 0
    for cluster in clusters:
        if len(list(set(orgs).intersection(set(cluster)))) != 0:
            not_empty_count += 1
    return not_empty_count != 0 and len(cluster_updates) == not_empty_count

    # if len(clusters) == 0:
    #     return False
    #
    # not_empty_count = 0
    # for cluster in clusters:
    #     if len(cluster) != 0:
    #         not_empty_count += 1
    #return len(clusterModelUrls) == not_empty_count

def cal_cluster():
    print("Start clustering")
    while True:
        similarity_matrix = session.get(org_server + get_similarity_matrix_api, headers=headers).content
        similarity_matrix = json.loads(similarity_matrix)
        if "transactionCode" in similarity_matrix:
            # print(similarity_matrix)
            continue

        similarity_matrix = np.array(similarity_matrix)
        print(similarity_matrix)
        break

    cluster_labels = AgglomerativeClustering(n_clusters=conf["cluster_centers"], affinity='precomputed', linkage='complete').fit_predict(
        similarity_matrix)

    org_ids = list(range(conf["client_num"]))
    clusters = []
    for i in range(conf["cluster_centers"]):
        cluster = np.intersect1d(np.where(cluster_labels == i), org_ids).tolist()
        clusters.append(cluster)

    return clusters

def aggregate():
    print("Start model aggregate")
    cluster_models = [torch.load("./models/server/cluster" + str(i) + ".pth", map_location=server.device) for i in
                      range(conf["cluster_centers"])]

    best_acc = 0.0
    best_loss = 0.0
    best_model = None
    tasks = []
    for i in range(conf["cluster_centers"]):
        tasks.append(executor.submit(server.model_eval, cluster_models[i]))
    wait(tasks, return_when=ALL_COMPLETED)

    for i in range(conf["cluster_centers"]):
        acc, loss = tasks[i].result()
        if acc > best_acc:
            best_acc = acc
            best_loss = loss
            best_model = cluster_models[i]

    cur_hash_id = str(hash(frozenset(best_model.state_dict().values())))
    server.save_model(model_name=cur_hash_id)
    return cur_hash_id, best_acc, best_loss

def server_thread():
    rounds = []
    accuracy = []
    # validloss = []
    times = []
    total_time_cost = 0
    start = time.time()
    while True:
        if check_cluster_aggregate():
            cal_cluster_task = executor.submit(cal_cluster)
            test_accuracy_task = executor.submit(aggregate)

            clusters = cal_cluster_task.result()
            end = time.time()
            total_time_cost += (end - start)
            start = end

            cur_hash_id, acc, loss = test_accuracy_task.result()
            print("Round {}, global model acc: {}, loss: {}, time consume:{}".format(server.round, acc, loss, total_time_cost))

            data = {
                "cur_hash_id": cur_hash_id,
                "cur_model_url": "./models/server/" + cur_hash_id + ".pth",
                "clusters": clusters,
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            try:
                identifier = acquire_lock(lock_name="global")
                response = session.post(org_server + upload_global_api, data=json.dumps(data), headers=headers)
                release_lock(identifier, lock_name="global")
                print(response.content)
            except requests.exceptions.RequestException as e:
                print("Error is {}".format(e))

            rounds.append(server.round)
            accuracy.append(acc)
            times.append(total_time_cost)
            plot.save_array(rounds, accuracy, times, dir=save_dir, name=save_name)
            plot.plot(rounds, accuracy, label1="accuracy", dir=save_dir, name=save_name)

        if server.round >= conf["global_epochs"]:
            break
        time.sleep(1)

if __name__ == '__main__':
    with open("./config/parallel-conf.json", 'r') as f:
        conf = json.load(f)
    print(conf)

    save_name = save_name = conf["dataset"] + "_result_" + ("iid" if conf["iid"] else "niid") + str(conf["niid_beta"]) + "_blockchain"
    save_dir = os.path.join("./results", save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_conf = json.dumps(conf)
    f2 = open(os.path.join(save_dir, "conf.json"), 'w')
    f2.write(save_conf)
    f2.close()

    dataset = dataset.GetDataSet(dataset_name=conf["dataset"], is_iid=conf["iid"], beta=conf["niid_beta"], client_num=conf["client_num"])
    server = Server(conf, eval_dataset=dataset.test_dataset, device=torch.device("cuda:2"))
    server.save_model(model_name="0")
    for i in range(conf["cluster_centers"]):
        server.save_model(model_name="cluster"+str(i))

    # server.save_model(model_name="0")
    # server.global_model = torch.load("./models/server/0.pth", map_location=server.device)
    # print(server.model_eval())

    executor = ThreadPoolExecutor(max_workers=8)

    redis_client = redis.Redis(host="114.212.82.242", port=6379, decode_responses=True)
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=5, method_whitelist=frozenset(['GET', 'POST']))))

    org_server = "http://114.212.82.242:8080/"
    upload_global_api = "updateGlobal"
    upload_cluster_api = "updateClusterInfo"

    get_global_api = "getGlobalModelMetaInfo"
    get_local_updates_api = "scanLocalUpdates"
    get_cluster_updates_api = "scanClusterUpdates"
    get_similarity_matrix_api = "getGlobalSimilarityMatrix"

    headers = {'content-type': 'application/json',
               'Connection': 'close',
               'User-Agent': 'User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}

    server_thread()

    #fedavg()

    # fedasync()