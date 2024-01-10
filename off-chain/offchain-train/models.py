import torch
import os
import glob
import numpy as np
from torch.nn import Parameter
from torchvision import models
from torch import nn
from transformers import BertPreTrainedModel, BertModel
import math
import torch.nn.functional as F
import dgl.data
from dgl.nn.pytorch import GraphConv


class VGGMNIST(nn.Module):
    def __init__(self):
        super(VGGMNIST, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)

class VGGCifar(nn.Module):
    def __init__(self, num_classes=10, model_name="vgg16"):
        super(VGGCifar, self).__init__()
        net = get_model(model_name, load_from_local=True)
        net.classifier = nn.Sequential()

        self.features = net
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(512, num_classes)
        # )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class BertClassifer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClassifer, self).__init__(config)

        self.num_labels = 4

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        cls = outputs[0][:, 0, :]
        x = self.dropout(cls)

        x = self.classifier(x)
        return x, cls
        #outputs = (logits_tag,)

        # if labels is not None:
        #     loss_func = torch.nn.CrossEntropyLoss()
        #     loss_tag = loss_func(logits_tag, labels)
        #
        #     loss = loss_tag
        #     outputs = (loss,) + outputs
        # return outputs

class CharLSTM(nn.Module):
    # def __init__(self, input_size=80, embed_size=80, hidden_size=64, output_size=80):
    def __init__(self, input_size=80, embed_size=80, hidden_size=128, output_size=80):
        super(CharLSTM, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        self.rnn = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)
        # self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        # self.dropout = nn.Dropout(0.5)
        # self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq, prev_state=None):
        embedding = self.embedding(input_seq)
        output, _ = self.rnn(embedding)
        output = self.dropout(output)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)
        return output[:, :, -1]
        # return output[:, -1, :]
    #
    # def zero_state(self, batch_size):
    #     return (torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size))

class RNN(torch.nn.Module):
    def __init__(self, input_size=64, hidden_size=64, output_size=2):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)
        # self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        # self.dropout = nn.Dropout(0.5)
        # self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq, prev_state=None):
        input_seq = input_seq.view(len(input_seq), 1, -1).to(torch.float)
        output, _ = self.rnn(input_seq)
        output = self.dropout(output)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)
        return output[:, :, -1]



# class GCNLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, acti=True):
#         super(GCNLayer, self).__init__()
#         self.linear = nn.Linear(in_dim, out_dim) # bias = False is also ok.
#         if acti:
#             self.acti = nn.ReLU(inplace=True)
#         else:
#             self.acti = None
#     def forward(self, F):
#         output = self.linear(F)
#         if not self.acti:
#             return output
#         return self.acti(output)


#class GCN(nn.Module):
    # def __init__(self, input_dim, hidden_dim, num_classes, p):
    #     super(GCN, self).__init__()
    #     self.gcn_layer1 = GCNLayer(input_dim, hidden_dim)
    #     self.gcn_layer2 = GCNLayer(hidden_dim, num_classes, acti=False)
    #     self.dropout = nn.Dropout(p)

    # def forward(self, X, A):
    #     A = torch.from_numpy(self.preprocess_adj(A)).float()
    #     A = A.to("cuda:0")
    #     X = self.dropout(X.float())
    #     F = torch.mm(A, X)
    #     F = self.gcn_layer1(F)
    #     F = self.dropout(F)
    #     F = torch.mm(A, F)
    #     output = self.gcn_layer2(F)
    #     return output
    #
    # def preprocess_adj(self, A):
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

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')
    def forward(self, output, labels, mask):
        mask = mask.to("cuda:0")
        # labels = torch.argmax(labels, dim=1)
        loss = self.loss(output, labels)

        mask = mask.float()

        mask /= torch.mean(mask)

        loss *= mask
        return torch.mean(loss)


def build_optimizer(model, lr, weight_decay):
    gcn1, gcn2 = [], []
    for name, p in model.named_parameters():
        if 'layer1' in name:
            gcn1.append(p)
        else:
            gcn2.append(p)
    opt = torch.optimizer.Adam([{'params': gcn1, 'weight_decay': weight_decay},
                      {'params': gcn2}
                      ], lr=lr)
    return opt


def get_lr():
    pass


def get_loss(output, labels, mask):
    loss = Loss()
    return loss(output, labels, mask)


def get_accuracy(outputs, labels, mask):
    outputs = torch.argmax(outputs, dim=1)
    # labels = torch.argmax(labels, dim=1)
    outputs = outputs.cpu().numpy()
    labels = labels.cpu().numpy()
    correct = outputs == labels
    #print(correct)
    mask = mask.cpu().numpy()
    tp = np.sum(correct * mask)
    return tp / np.sum(mask)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, p):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, num_classes)
        self.dropout = p
        #self.dropout = nn.Dropout(p)

    def forward(self, x, adj):
        '''
        step1. gc
        step2. relu
        step3. dropout
        step4. gc
        step5. log_softmax 并输出
        '''
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, acti=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim) # bias = False is also ok.
        if acti:
            self.acti = nn.ReLU(inplace=True)
        else:
            self.acti = None
    def forward(self, F):
        output = self.linear(F)
        if not self.acti:
            return output
        return self.acti(output)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 这里设，特征矩阵feature为H，邻接矩阵adj为A，权重为W，则输出为，
    # step1. 求 H 和 W 乘积  HW
    # step2. 求 A 和 HW 乘积 AHW，这里 A = D_A ^ -1 · (A + I) 做了归一化， H = D_H^-1 · H
    # 对于维度
    '''
    # adj 		2708,2708   		A
    # features 	2708,1433			H0
    # labels	2708,      0~6
    第一次gc后为2708,nhid
    第二次gc后为2708,7 (7个类别)
    '''

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def get_model(name="vgg16",  model_dir="./models/checkpoints/", pretrained=True, load_from_local=False):
    os.environ['TORCH_HOME'] = './models/'
    if load_from_local:
        #print("Load model from local dir {}".format(model_dir))
        model = eval('models.%s(pretrained=False)' % name)
        path_format = os.path.join(model_dir, '%s-[a-z0-9]*.pth' % name)
        model_path = glob.glob(path_format)[0]
        model.load_state_dict(torch.load(model_path))

    else:
        print("Download model from Internet")
        if name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
        elif name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
        elif name == "densenet121":
            model = models.densenet121(pretrained=pretrained)
        elif name == "alexnet":
            model = models.alexnet(pretrained=pretrained)
        elif name == "vgg11":
            model = models.vgg11(pretrained=pretrained)
        elif name == "vgg16":
            model = models.vgg16(pretrained=pretrained)
        elif name == "vgg19":
            model = models.vgg19(pretrained=pretrained)
        elif name == "inception_v3":
            model = models.inception_v3(pretrained=pretrained)
        elif name == "googlenet":
            model = models.googlenet(pretrained=pretrained)

    return model
    # if torch.cuda.is_available():
    #     return model.cuda()
    # else:
    #     return model


if __name__ == '__main__':

    #model = get_model(name="resnet50", load_from_local=False)
    vgg = VGGCifar()
    print(vgg)

    # model = GCN(nfeat=1433, nhid=32, nclass=7, dropout=0.1)
    # print(model.gc1)

