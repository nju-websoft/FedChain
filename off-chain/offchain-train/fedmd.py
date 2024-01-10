import os
import numpy as np
import torch
from tqdm import tqdm
import plot
import random
import json
import torch.nn.functional as F

from server import Server
from client import Client, ClientGroup
from dataset import GetDataSet


class FedMDserver(Server):
    def __init__(self, conf, eval_dataset, public_dataset, device=None):
        super(FedMDserver, self).__init__(conf, eval_dataset, device)

        self.public_loader = torch.utils.data.DataLoader(public_dataset,
                                                         batch_size=self.conf["batch_size"],
                                                         shuffle=False,
                                                         num_workers=4)
        self.global_logits = None

    def label_aggregate(self, models):
        self.model_weight_aggregate(models)

        self.global_logits = None

        for batch_id, batch in enumerate(tqdm(self.public_loader)):
            data, target = batch
            data = data.to(self.device)
            target = target.to(self.device, dtype=torch.long)

            models[0].eval()
            output = torch.zeros_like(models[0](data))

            for model in models:
                model.eval()
                output += model(data)

            output = output.cpu().detach().numpy()
            output /= len(models)
            output = np.argmax(output, axis=1)
            # ids = range(batch_id * self.conf["batch_size"], batch_id * self.conf["batch_size"] + len(target))
            # print(ids)

            if self.global_logits is None:
                self.global_logits = output
            else:
                self.global_logits = np.concatenate((self.global_logits, output), axis=0)
            #print(self.global_logits.shape)

class FedMdClient(Client):
    def __init__(self, conf, train_dataset, test_dataset, public_dataset, id=-1, global_model=None, device=None):
        super(FedMdClient, self).__init__(conf, train_dataset, test_dataset, id, global_model, device)

        self.public_loader = torch.utils.data.DataLoader(public_dataset,
                                                         batch_size=self.conf["batch_size"],
                                                         shuffle=False,
                                                         num_workers=4)

    def fedmd_local_train(self, global_model, global_logits=None):
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        if global_logits is not None:
            optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])
            #criterion = torch.nn.KLDivLoss(reduction="batchmean")
            criterion = torch.nn.CrossEntropyLoss()

            for e in range(self.conf["public_dataset_epochs"]):
                total_loss = 0.0
                self.local_model.train()

                for batch_id, batch in enumerate(tqdm(self.public_loader)):
                    data, target = batch
                    data = data.to(self.device)
                    target = target.to(self.device, dtype=torch.long)

                    optimizer.zero_grad()
                    output = self.local_model(data)

                    ids = range(batch_id * self.conf["batch_size"], batch_id * self.conf["batch_size"] + len(target))
                    global_output = global_logits[ids]
                    global_output = torch.from_numpy(global_output).to(self.device)

                    # T = 1
                    # loss = criterion(F.log_softmax(output / T, dim=1),
                    #                F.softmax(global_output / T, dim=1)) * T * T

                    #loss = criterion(output, global_output)
                    loss = criterion(output, target)

                    loss.backward()
                    total_loss += loss.item()
                    optimizer.step()

                acc, _ = self.eval_model()
                print("Epoch {} done. Public dataset loss {}. Valid accuracy {}".format(e, total_loss, acc))

            self.local_train(self.local_model)
        else:
            self.local_train(self.local_model)

if __name__ == "__main__":
    torch.cuda.empty_cache()

    with open("./config/fedmd-conf.json", 'r') as f:
        conf = json.load(f)
    print(conf)

    save_name = conf["dataset"] + "_result_" + ("iid" if conf["iid"] else "niid") + str(conf["niid_beta"]) + "_fedmd"
    save_dir = os.path.join("./results", save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_conf = json.dumps(conf)
    f2 = open(os.path.join(save_dir, "conf.json"), 'w')
    f2.write(save_conf)
    f2.close()

    dataset = GetDataSet(dataset_name=conf["dataset"], is_iid=conf["iid"], beta=conf["niid_beta"])
    public_dataset = dataset.get_public_share_dataset()

    server = FedMDserver(conf,
                         eval_dataset=dataset.test_dataset,
                         public_dataset=public_dataset,
                         device=torch.device(torch.device("cuda:"+str(2))))

    clients = []
    for i in range(conf["client_num"]):
        subset = torch.utils.data.Subset(dataset.train_dataset, dataset.client_idcs[i])

        clients.append(FedMdClient(
            conf=conf,
            train_dataset=subset,
            test_dataset=dataset.test_dataset,
            public_dataset=public_dataset,
            id=i,
            global_model=server.global_model,
            device=torch.device("cuda:"+str(2))))

    epochs = []
    accuracy = []
    validloss = []
    for e in range(conf["global_epochs"]):

        k = random.randint(conf["min_k"], conf["max_k"])
        candidates = random.sample(clients, k)

        for c in candidates:
            if e == 0:
                c.fedmd_local_train(server.global_model)
            else:
                c.fedmd_local_train(server.global_model, server.global_logits)

        server.label_aggregate([c.local_model for c in candidates])

        acc, loss = server.model_eval()
        print("Epoch %d, global model acc: %f\n" % (e, acc))
        epochs.append(e)
        validloss.append(loss)
        accuracy.append(acc)

        if e % 10 == 0 and e > 0:
            plot.plot(epochs, accuracy, label1="accuracy", dir=save_dir, name=save_name)
            plot.save_array(epochs, accuracy, validloss, dir=save_dir, name=save_name)