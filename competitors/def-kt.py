import argparse, json
import time

import torch, random
from server import *
from client import *
import dataset
import plot
from concurrent.futures import ThreadPoolExecutor, as_completed

#random.seed(42)

if __name__ == '__main__':

    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if torch.cuda.is_available():
        print(torch.version.cuda)
        print(torch.__version__)
        device0 = "cuda:0"
        device1 = "cuda:1"
        device2 = "cuda:2"
        device3 = "cuda:3"
    else:
        device0 = device1 = device2 = device3 = "cpu"

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)
    print(conf)

    best_acc = 0.0
    save_name = conf["dataset"] + "_result_" + ("iid" if conf["iid"] else ("niid"+str(conf["niid_beta"]))) + "_" + "onlyexhcange"
    save_dir = os.path.join("./results", save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_conf = json.dumps(conf)
    f2 = open(os.path.join(save_dir, "conf.json"), 'w')
    f2.write(save_conf)
    f2.close()

    records = {}
    for i in range(conf["client_num"]):
        records[i] = [0]*conf["client_num"]
    print(records)

    dataset = dataset.GetDataSet(dataset_name=conf["dataset"], is_iid=conf["iid"], beta=conf["niid_beta"], client_num=conf["client_num"])
    server = Server(conf,dataset.test_dataset, device=torch.device(device0))
    acc, loss = server.model_eval()

    epochs = []
    accuracy = []
    validloss = []
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

    #pool = ThreadPoolExecutor(max_workers=conf["k"])

    print("Start Training...")
    client_indices=list(range(conf["client_num"]))
    best_final_acc = 0
    bad_count = 0
    start = time.time()
    for e in range(conf["global_epochs"]):
        print("#######Global epoch {} start########".format(e))
        k = random.randint(conf["min_k"], conf["max_k"])

        candidates = random.sample(client_indices, k)
        not_selected = list(set(client_indices).difference(set(candidates)))
        print("candidates:{}; not_selects:{}\n".format(candidates, not_selected))

        best_acc = 0
        best_loss = 0
        if conf["exchange"]:

            for i in range(k):
                clients[candidates[i]].local_train(clients[candidates[i]].local_model)
                #clients[candidates[i]].local_train(server.global_model)
                print("mutual learning client: {}".format(not_selected[i]))

                print("The client {} load the model from clients {}".format(candidates[i], not_selected[i]))
                records[candidates[i]][not_selected[i]] += 1

                acc, loss = clients[candidates[i]].eval_model()
                print("Before : client {} valid acc {}".format(candidates[i], acc))

                clients[candidates[i]].fuse_model_by_teachers([[clients[not_selected[i]].client_id, clients[not_selected[i]].local_model]])
                acc, loss = clients[candidates[i]].eval_model()
                print("After : client {} valid acc {}".format(clients[candidates[i]].client_id, acc))

                if acc > best_acc:
                    best_acc = acc
                    best_loss = loss
        # end = time.time()
        # print("Epoch %d, best acc: %f, best loss: %f, time consume: %f\n" % (e, best_acc, best_loss, end - start))


        best_acc, best_lossloss = server.model_weight_aggregate([copy.deepcopy(clients[c].local_model) for c in candidates], client_datalen, total_len, [clients[c].client_id for c in candidates])

        end = time.time()
        print("Epoch %d, best acc: %f, best loss: %f, time consume: %f\n" % (e, acc, loss, end-start))
        task_list = []
        epochs.append(e)
        validloss.append(best_loss)
        accuracy.append(best_acc)

        if best_acc > best_final_acc:
            best_final_acc = best_acc
            bad_count = 0
        else:
            bad_count += 1

        if bad_count >= 45:
            plot.plot(epochs, accuracy, label1="accuracy", dir=save_dir, name=save_name)
            plot.save_array(epochs, accuracy, validloss, dir=save_dir, name=save_name)
            print("Finish training")
            break

        if e % 10 == 0 and e > 0:
            print(records)
            plot.plot(epochs, accuracy, label1="accuracy", dir=save_dir, name=save_name)
            plot.save_array(epochs, accuracy, validloss, dir=save_dir, name=save_name)