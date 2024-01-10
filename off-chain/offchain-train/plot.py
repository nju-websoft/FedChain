import copy
import sys
import os
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator, MultipleLocator
import seaborn as sns
import numpy as np
import pylab
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rc('font',family='Times New Roman')

tsne = TSNE(n_components=2, random_state=0)
pca = PCA(n_components=2)

def plot(x,y1,label1, dir="./results/", name="result"):
    plt.figure().clear()
    ax = plt.subplot(1,1,1)
    plt.sca(ax)
    plt.plot(x, y1, "r", label=label1)
    plt.legend()

    path = os.path.join(dir, name+".png")
    plt.savefig(path)

plt.rcParams.update({'font.size': 8})

def save_array(x, y1, y2, dir="./results/", name="result"):
    list = np.vstack((x,y1,y2))
    path = os.path.join(dir, name + ".npy")
    np.save(path, list)


def cal_avg_accuracy(list):
    return sum(list) / len(list)

def compare5_earlystop_twoplot(x1, y1, label1, y2, label2, y3, label3, y4, label4, y5, label5, y6, label6, y7, label7, y8, label8, y9, label9, y10, label10, dir="./results/", name="result", x2=None, x3=None, x4=None, x5=None, x6=None, x7=None, x8=None, x9=None, x10=None):

    plt.figure(figsize=(4, 3))
    plt.rcParams['ytick.direction'] = 'in'
    plt.ylim((0, 1.0))

    y_max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    maxlen = min(len(x1), len(x2), len(x3), len(x4), len(x5), len(x6), len(x7), len(x8), len(x9), len(x10))

    #maxlen = min(len(x1), len(x2), len(x3), len(x4), len(x5), len(x6))
    for i in range(maxlen):

        y1[i] = y1[i].item()
        y3[i] = y3[i].item()
        y4[i] = y4[i].item()
        y5[i] = y5[i].item()
        y7[i] = y7[i].item()
        y8[i] = y8[i].item()
        y9[i] = y9[i].item()
        y10[i] = y10[i].item()


        if y1[i] > y_max[0]:
            y_max[0] = y1[i]
        if y2[i] > y_max[1]:
            y_max[1] = y2[i]
        if y3[i] > y_max[2]:
            y_max[2] = y3[i]
        if y4[i] > y_max[3]:
            y_max[3] = y4[i]
        if y5[i] > y_max[4]:
            y_max[4] = y5[i]
        if y6[i] > y_max[5]:
            y_max[5] = y6[i]
        if y7[i] > y_max[6]:
            y_max[6] = y7[i]
        if y8[i] > y_max[7]:
            y_max[7] = y8[i]
        if y9[i] > y_max[8]:
            y_max[8] = y9[i]
        if y10[i] > y_max[9]:
            y_max[9] = y10[i]

    x_max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(maxlen):
        if y1[i] == y_max[0]:
            x_max[0] = i
        if y2[i] == y_max[1]:
            x_max[1] = i
        if y3[i] == y_max[2]:
            x_max[2] = i
        if y4[i] == y_max[3]:
            x_max[3] = i
        if y5[i] == y_max[4]:
            x_max[4] = i
        if y6[i] == y_max[5]:
            x_max[5] = i
        if y7[i] == y_max[6]:
            x_max[6] = i
        if y8[i] == y_max[7]:
            x_max[7] = i
        if y9[i] == y_max[8]:
            x_max[8] = i
        if y10[i] == y_max[9]:
            x_max[9] = i


    # a = plt.plot(x1[:x_max[0]], y1[:x_max[0]]/100.0, "slategrey", label="DFL")
    # b = plt.plot(x2[:x_max[1]], y2[:x_max[1]]/100.0, "hotpink", label="FedAvg")
    # c = plt.plot(x3[:x_max[2]], y3[:x_max[2]]/100.0, "orange", label="FedProx")
    # d = plt.plot(x9[:x_max[8]], y9[:x_max[8]]/100.0, "g", label="FedHiSyn")
    # e = plt.plot(x5[:x_max[4]], y5[:x_max[4]]/100.0, "r", label="FedChain")
    # f = plt.plot(x6[:x_max[5]], y6[:x_max[5]]/100.0, "y", label="ScaleSFL")
    # g = plt.plot(x7[:x_max[6]], y7[:x_max[6]]/100.0, "m", label="MOON")

    a = plt.plot(x1[:x_max[0]], y1[:x_max[0]], "slategrey", label="DFL")
    b = plt.plot(x1[:x_max[1]], y2[:x_max[1]], "hotpink", label="FedAvg")
    c = plt.plot(x1[:x_max[2]], y3[:x_max[2]], "orange", label="FedProx")
    d = plt.plot(x9[:x_max[8]], y9[:x_max[8]], "g", label="FedHiSyn")
    e = plt.plot(x1[:x_max[4]], y5[:x_max[4]], "r", label="FedChain")
    f = plt.plot(x1[:x_max[5]], y6[:x_max[5]], "y", label="ScaleSFL")
    g = plt.plot(x7[:x_max[6]], y7[:x_max[6]], "m", label="MOON")

    #plt.xlabel("Communication rounds", fontdict= {"size":16})
    #plt.ylabel("Accuracy", fontdict= {"size":16})
    #plt.title("CiteSeer", fontdict={"size":20})
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)

    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    ax2 = plt.twinx()
    #ax2.set_ylabel('β=0.1', fontdict= {"size":16})
    #ax2.set_ylabel('IID', fontdict={"size": 16})
    ax2.axes.yaxis.set_ticklabels([])
    ax2.set_yticks([])

    path = os.path.join(dir, "result_syn.svg")
    plt.savefig(path, format='svg', dpi=150, bbox_inches='tight', pad_inches=0.1, transparent=True)

    plt.clf()
    plt.figure(figsize=(4, 3))
    plt.rcParams['ytick.direction'] = 'in'
    plt.ylim((0, 1.0))
    a = plt.plot(x4[:x_max[3]], y4[:x_max[3]] / 100.0, "c", label="FedAsync")
    b = plt.plot(x8[:x_max[7]], y8[:x_max[7]] / 100.0, "sienna", label="CSAFL")
    d = plt.plot(x10[:x_max[9]], y10[:x_max[9]] / 100.0, "b", label="FedAT")
    e = plt.plot(x5[:x_max[4]], y5[:x_max[4]] / 100.0, "r", label="FedChain")

    # a = plt.plot(x1[:x_max[3]], y4[:x_max[3]], "c", label="FedAsync")
    # b = plt.plot(x1[:x_max[4]], y5[:x_max[4]], "r", label="FedChain")
    # c = plt.plot(x8[:x_max[7]], y8[:x_max[7]], "sienna", label="CSAFL")
    # e = plt.plot(x10[:x_max[9]], y10[:x_max[9]], "b", label="FedAT")

    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)

    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])
    ax2 = plt.twinx()
    ax2.axes.yaxis.set_ticklabels([])
    ax2.set_yticks([])
    path = os.path.join(dir, "result_asyn.svg")
    plt.savefig(path, format='svg', dpi=150, bbox_inches='tight', pad_inches=0.1, transparent=True)


if __name__ == '__main__':

    e = np.load("results/cifar100_result_iid_exchange_cluster/cifar100_result_iid_exchange_cluster.npy")
    d = np.load("results/cifar100_result_iid_fedasync/cifar100_result_iid_fedasync.npy")
    c = np.load("results/cifar100_result_iid_base/cifar100_result_iid_base.npy")
    b = np.load("results/cifar100_result_iid_base_prox/cifar100_result_iid_base_prox.npy")
    a = np.load("results/cifar100_result_iid_onlyexhcange/cifar100_result_iid_onlyexhcange.npy")
    f = np.load("results/cifar100_result_iid_scalesfl/cifar100_result_iid_scalesfl.npy")
    g = np.load("results/cifar100_result_iid_moon/cifar100_result_iid_moon.npy")
    h = np.load("results/cifar100_result_iid_csafl/cifar100_result_iid_csafl.npy")
    ii = np.load("results/cifar100_result_iid_fedhisyn/cifar100_result_iid_fedhisyn.npy")
    j = np.load("results/cifar100_result_iid_fedat/cifar100_result_iid_fedat.npy")

    # e = np.load("results/cifar100_result_niid0.05_exchange_cluster/cifar100_result_niid0.05_exchange_cluster.npy")
    # d = np.load("results/cifar100_result_niid0.05_fedasync/cifar100_result_niid0.05_fedasync.npy")
    # c = np.load("results/cifar100_result_niid0.05_base/cifar100_result_niid0.05_base.npy")
    # b = np.load("results/cifar100_result_niid0.05_base_prox/cifar100_result_niid0.05_base_prox.npy")
    # a = np.load("results/cifar100_result_niid0.05_onlyexhcange/cifar100_result_niid0.05_onlyexhcange.npy")
    # f = np.load("results/cifar100_result_niid0.05_scalesfl_1/cifar100_result_niid0.05_scalesfl.npy")
    # g = np.load("results/cifar100_result_niid0.05_moon/cifar100_result_niid0.05_moon.npy")
    # h = np.load("results/cifar100_result_niid0.05_csafl/cifar100_result_niid0.05_csafl.npy")
    # ii = np.load("results/cifar100_result_niid0.05_fedhisyn/cifar100_result_niid0.05_fedhisyn.npy")
    # j = np.load("results/cifar100_result_niid0.05_fedat/cifar100_result_niid0.05_fedat.npy")

    # a = np.load("results/cifar10_result_iid_onlyexhcange/cifar10_result_iid_onlyexhcange.npy")
    # b = np.load("results/cifar10_result_iid_base/cifar10_result_iid_base.npy")
    # c = np.load("results/cifar10_result_iid_base_prox/cifar10_result_iid_base_prox.npy")
    # d = np.load("results/cifar10_result_iid_fedasync/cifar10_result_iid_fedasync.npy")
    # e = np.load("results/cifar10_result_iid_exchange_cluster/cifar10_result_iid_exchange_cluster.npy")
    # f = np.load("results/cifar10_result_iid_scalesfl/cifar10_result_iid_scalesfl.npy")
    # g = np.load("results/cifar10_result_iid_moon/cifar10_result_iid_moon.npy")
    # h = np.load("results/cifar10_result_iid_csafl/cifar10_result_iid_csafl.npy")
    # ii = np.load("results/cifar10_result_iid_fedhisyn/cifar10_result_iid_fedhisyn.npy")
    # j = np.load("results/cifar10_result_iid_fedat/cifar10_result_iid_fedat.npy")

    # e = np.load("results/cifar10_result_niid0.05_exchange_cluster_1/cifar10_result_niid0.05_exchange_cluster.npy")
    # d = np.load("results/cifar10_result_niid0.05_fedasync_1/cifar10_result_niid0.05_fedasync.npy")
    # c = np.load("results/cifar10_result_niid0.05_base/cifar10_result_niid0.05_base.npy")
    # b = np.load("results/cifar10_result_niid0.05_base_prox/cifar10_result_niid0.05_base_prox.npy")
    # a = np.load("results/cifar10_result_niid0.05_onlyexhcange/cifar10_result_niid0.05_onlyexhcange.npy")
    # f = np.load("results/cifar10_result_niid0.05_scalesfl/cifar10_result_niid0.05_scalesfl.npy")
    # g = np.load("results/cifar10_result_niid0.05_moon/cifar10_result_niid0.05_moon.npy")
    # h = np.load("results/cifar10_result_niid0.05_csafl/cifar10_result_niid0.05_csafl.npy")
    # ii = np.load("results/cifar10_result_niid0.05_fedhisyn/cifar10_result_niid0.05_fedhisyn.npy")
    # j = np.load("results/cifar10_result_niid0.05_fedat/cifar10_result_niid0.05_fedat1.npy")

    # e = np.load("results/shakespeare_result_iid_exchange_cluster/shakespeare_result_iid_exchange_cluster.npy")
    # d = np.load("results/shakespeare_result_iid_fedasync/shakespeare_result_iid_fedasync.npy")
    # c = np.load("results/shakespeare_result_iid_base_prox/shakespeare_result_iid_base_prox.npy")
    # b = np.load("results/shakespeare_result_iid_base/shakespeare_result_iid_base.npy")
    # a = np.load("results/shakespeare_result_iid_onlyexhcange/shakespeare_result_iid_onlyexhcange.npy")
    # f = np.load("results/shakespeare_result_iid_scalesfl/shakespeare_result_iid_scalesfl.npy")
    # g = np.load("results/shakespeare_result_iid_moon/shakespeare_result_iid_moon.npy")
    # h = np.load("results/shakespeare_result_iid_csafl/shakespeare_result_iid_csafl.npy")
    # ii = np.load("results/shakespeare_result_iid_fedhisyn/shakespeare_result_iid_fedhisyn.npy")
    # j = np.load("results/shakespeare_result_iid_fedat/shakespeare_result_iid_fedat.npy")

    # e = np.load("results/shakespeare_result_niid0.05_exchange/shakespeare_result_niid0.05_exchange.npy")
    # d = np.load("results/shakespeare_result_niid0.05_fedasync/shakespeare_result_niid0.05_fedasync.npy")
    # c = np.load("results/shakespeare_result_niid0.05_base_prox/shakespeare_result_niid0.05_base_prox.npy")
    # b = np.load("results/shakespeare_result_niid0.05_base/shakespeare_result_niid0.05_base.npy")
    # a = np.load("results/shakespeare_result_niid0.05_onlyexhcange/shakespeare_result_niid0.05_onlyexhcange.npy")
    # # f = np.load("results/shakespeare_result_niid0.05_scalesfl/shakespeare_result_niid0.05_scalesfl.npy")
    # f = np.load("results/shakespeare_result_niid0.08_onlyexhcange_2/shakespeare_result_niid0.08_onlyexhcange.npy")
    # #g = np.load("results/shakespeare_result_niid0.05_moon/shakespeare_result_niid0.05_moon.npy")
    # g = np.load("results/shakespeare_result_niid0.08_onlyexhcange_3/shakespeare_result_niid0.08_onlyexhcange.npy")
    # #h = np.load("results/shakespeare_result_niid0.05_csafl/shakespeare_result_niid0.05_csafl.npy")
    # h = np.load("results/shakespeare_result_niid0.08_onlyexhcange_1/shakespeare_result_niid0.08_onlyexhcange_1.npy")
    # ii = np.load("results/shakespeare_result_niid0.05_fedhisyn/shakespeare_result_niid0.05_fedhisyn.npy")
    # j = np.load("results/shakespeare_result_niid0.05_fedat/shakespeare_result_niid0.05_fedat.npy")

    # e = np.load("results/cora_result_iid_exchange_prox/cora_result_iid_exchange_prox.npy", allow_pickle=True)
    # d = np.load("results/cora_result_iid_fedasync/cora_result_iid_fedasync.npy", allow_pickle=True)
    # c = np.load("results/cora_result_iid_base_prox/cora_result_iid_base_prox.npy", allow_pickle=True)
    # b = np.load("results/cora_result_iid_base/cora_result_iid_base.npy", allow_pickle=True)
    # a = np.load("results/cora_result_iid_onlyexhcange/cora_result_iid_onlyexhcange.npy", allow_pickle=True)
    # f = np.load("results/cora_result_iid_scalesfl/cora_result_iid_scalesfl.npy", allow_pickle=True)
    # g = np.load("results/cora_result_iid_moon/cora_result_iid_moon.npy", allow_pickle=True)
    # h = np.load("results/cora_result_iid_csafl/cora_result_iid_csafl.npy", allow_pickle=True)
    # ii = np.load("results/cora_result_iid_fedhisyn/cora_result_iid_fedhisyn.npy", allow_pickle=True)
    # j = np.load("results/cora_result_iid_fedat/cora_result_iid_fedat.npy", allow_pickle=True)

    # e = np.load("results/cora_result_niid0.05_exchange/cora_result_niid0.05_exchange.npy", allow_pickle=True)
    # d = np.load("results/cora_result_niid0.05_fedasync/cora_result_niid0.05_fedasync.npy", allow_pickle=True)
    # c = np.load("results/cora_result_niid0.05_base_prox/cora_result_niid0.05_base_prox.npy", allow_pickle=True)
    # b = np.load("results/cora_result_niid0.05_base/cora_result_niid0.05_base.npy", allow_pickle=True)
    # a = np.load("results/cora_result_niid0.05_onlyexhcange/cora_result_niid0.05_onlyexhcange.npy", allow_pickle=True)
    # f = np.load("results/cora_result_niid0.05_scalesfl/cora_result_niid0.05_scalesfl.npy", allow_pickle=True)
    # g = np.load("results/cora_result_niid0.05_moon/cora_result_niid0.05_moon.npy", allow_pickle=True)
    # h = np.load("results/cora_result_niid0.05_csafl/cora_result_niid0.05_csafl.npy", allow_pickle=True)
    # ii = np.load("results/cora_result_niid0.05_fedhisyn/cora_result_niid0.05_fedhisyn.npy", allow_pickle=True)
    # j = np.load("results/cora_result_niid0.05_fedat/cora_result_niid0.05_fedat.npy", allow_pickle=True)

    # e = np.load("results/citeseer_result_iid_exchange/citeseer_result_iid_exchange.npy", allow_pickle=True)
    # d = np.load("results/citeseer_result_iid_fedasync/citeseer_result_iid_fedasync.npy", allow_pickle=True)
    # c = np.load("results/citeseer_result_iid_base_prox/citeseer_result_iid_base_prox.npy", allow_pickle=True)
    # b = np.load("results/citeseer_result_iid_base/citeseer_result_iid_base.npy", allow_pickle=True)
    # a = np.load("results/citeseer_result_iid_onlyexhcange/citeseer_result_iid_onlyexhcange.npy", allow_pickle=True)
    # f = np.load("results/citeseer_result_iid_scalesfl/citeseer_result_iid_scalesfl.npy", allow_pickle=True)
    # g = np.load("results/citeseer_result_iid_moon/citeseer_result_iid_moon.npy", allow_pickle=True)
    # h = np.load("results/citeseer_result_iid_csafl/citeseer_result_iid_csafl.npy", allow_pickle=True)
    # ii = np.load("results/citeseer_result_iid_fedhisyn/citeseer_result_iid_fedhisyn.npy", allow_pickle=True)
    # j = np.load("results/citeseer_result_iid_fedat/citeseer_result_iid_fedat.npy", allow_pickle=True)

    # e = np.load("results/citeseer_result_niid0.05_exchange/citeseer_result_niid0.05_exchange.npy", allow_pickle=True)
    # d = np.load("results/citeseer_result_niid0.05_fedasync/citeseer_result_niid0.05_fedasync.npy", allow_pickle=True)
    # c = np.load("results/citeseer_result_niid0.05_base_prox/citeseer_result_niid0.05_base_prox.npy", allow_pickle=True)
    # b = np.load("results/citeseer_result_niid0.05_base/citeseer_result_niid0.05_base.npy", allow_pickle=True)
    # a = np.load("results/citeseer_result_niid0.05_onlyexhcange/citeseer_result_niid0.05_onlyexhcange.npy", allow_pickle=True)
    # f = np.load("results/citeseer_result_niid0.05_scalesfl/citeseer_result_niid0.05_scalesfl.npy", allow_pickle=True)
    # g = np.load("results/citeseer_result_niid0.05_moon/citeseer_result_niid0.05_moon.npy", allow_pickle=True)
    # h = np.load("results/citeseer_result_niid0.05_csafl/citeseer_result_niid0.05_csafl.npy", allow_pickle=True)
    # ii = np.load("results/citeseer_result_niid0.05_fedhisyn/citeseer_result_niid0.05_fedhisyn.npy", allow_pickle=True)
    # j = np.load("results/citeseer_result_niid0.05_fedat/citeseer_result_niid0.05_fedat.npy", allow_pickle=True)

    index = min(len(a[1]), len(b[1]), len(c[1]), len(d[1]), len(e[1]), len(f[1]), len(g[1]), len(h[1]), len(ii[1]), len(j[1]))
    mean_indices = index - 20
    if mean_indices < 0:
        mean_indices = 0

    print("fedchain max {}; mean {}".format(max(e[1][:index]), cal_avg_accuracy(e[1][mean_indices:index])))
    print("fedasync max {}; mean {}".format(max(d[1][:index]), cal_avg_accuracy(d[1][mean_indices:index])))
    print("fedprox max {}; mean {}".format(max(c[1][:index]), cal_avg_accuracy(c[1][mean_indices:index])))
    print("fedavg max {}; mean {}".format(max(b[1][:index]), cal_avg_accuracy(b[1][mean_indices:index])))
    print("mutual learning max {}; mean {}".format(max(a[1][:index]), cal_avg_accuracy(a[1][mean_indices:index])))
    print("scalesfl max{}; mean {}".format(max(f[1][:index]), cal_avg_accuracy(f[1][mean_indices:index])))
    print("moon max{}; mean {}".format(max(g[1][:index]), cal_avg_accuracy(g[1][mean_indices:index])))
    print("fedhisyn max{}; mean {}".format(max(ii[1][:index]), cal_avg_accuracy(ii[1][mean_indices:index])))
    print("fedat max{}; mean {}".format(max(j[1][:index]), cal_avg_accuracy(j[1][mean_indices:index])))
    compare5_earlystop_twoplot(a[0], a[1], "DFL", b[1], "FedAvg", c[1], "FedProx", d[1], "FedAsync", e[1], "FedChain", f[1],"ScaleSFL", g[1], "Moon", h[1], "CSAFL", ii[1], "FedHiSyn", j[1], "FedAT", x2=b[0], x3=c[0], x4=d[0], x5=e[0], x6=f[0], x7=g[0], x8=h[0], x9=ii[0], x10=j[0])
