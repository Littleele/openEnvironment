from __future__ import print_function
from sklearn.datasets import fetch_20newsgroups
from torch.autograd import Variable
from torch import autograd
from gensim.models import Word2Vec
import torch
import re
import math
import os
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy import data
from torchtext.data.utils import get_tokenizer
import jieba
from torchtext.vocab import Vectors
from gensim.models import doc2vec
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import utils
from gensim.models import keyedvectors
from sklearn.model_selection import train_test_split

import random

# hyper parameter
Batch_Size = 128
Embedding_Size = 100  # 词向量维度
Filter_Num = 20  # 卷积核个数
Dropout = 0.3
Epochs = 500
LR = 0.01
max_sent = 200
Label_Num = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data1 = fetch_20newsgroups(subset='all', categories=['comp.graphics'], shuffle=False)
data2 = fetch_20newsgroups(subset='all', categories=['comp.os.ms-windows.misc'], shuffle=False)
data3 = fetch_20newsgroups(subset='all', categories=['comp.sys.mac.hardware'], shuffle=False)
data4 = fetch_20newsgroups(subset='all', categories=['comp.windows.x'], shuffle=False)
data5 = fetch_20newsgroups(subset='all', categories=['rec.autos'], shuffle=False)
data6 = fetch_20newsgroups(subset='all', categories=['rec.motorcycles'], shuffle=False)
data7 = fetch_20newsgroups(subset='all', categories=['rec.sport.baseball'], shuffle=False)
data8 = fetch_20newsgroups(subset='all', categories=['rec.sport.hockey'], shuffle=False)
data9 = fetch_20newsgroups(subset='all', categories=['sci.crypt'], shuffle=False)
data10 = fetch_20newsgroups(subset='all', categories=['sci.electronics'], shuffle=False)
data11 = fetch_20newsgroups(subset='all', categories=['sci.med'], shuffle=False)
data12 = fetch_20newsgroups(subset='all', categories=['sci.space'], shuffle=False)
data13 = fetch_20newsgroups(subset='all', categories=['talk.politics.guns'], shuffle=False)
data14 = fetch_20newsgroups(subset='all', categories=['talk.politics.mideast'], shuffle=False)
data15 = fetch_20newsgroups(subset='all', categories=['talk.politics.misc'], shuffle=False)
data16 = fetch_20newsgroups(subset='all', categories=['talk.religion.misc'], shuffle=False)


def get_data(data_file, label_file):
    data = np.load(data_file)
    label = np.load(label_file)
    label = label % 4
    x = torch.from_numpy(data)  # [369, 63]
    x = x.type(torch.FloatTensor)
    y = torch.from_numpy(label)  # [369]
    y = torch.tensor(y).long()
    dataset = Data.TensorDataset(x, y)
    data_loader = Data.DataLoader(dataset=dataset, batch_size=Batch_Size, shuffle=True)
    return data_loader


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, Filter_Num, (2, Embedding_Size)),
            nn.ReLU(),
            nn.MaxPool2d((max_sent - 1, 1))
        )
        self.dropout = nn.Dropout(Dropout)
        self.fc = nn.Linear(Filter_Num, Label_Num)

    def forward(self, X):  # [batch_size, 63]
        batch_size = X.shape[0]
        # X = word2vec(X)  # [batch_size, 63, 50]
        X = X.unsqueeze(1)  # [batch_size, 1, 63, 50]
        X = self.conv(X)  # [batch_size, 10, 1, 1]
        X = X.view(batch_size, -1)  # [batch_size, 10]
        X = self.fc(X)  # [batch_size, 2]
        return X


def valid(model, test_loader):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for cur_data, cur_label in test_loader:
            output = model.forward(cur_data)
            # 损失函数参数一为网络输出，参数二为真实标签，size=[batch_size]
            test_loss += loss_fuc(output, cur_label).item()
            # print(output)
            # 准确率增加
            output = torch.max(output, dim=1)[1].numpy()
            label = cur_label.numpy()
            correct += sum(output == label)

        test_loss /= num_batches
        correct /= size

        # print('epoch:', epoch, ' |test loss:%.4f' % test_loss, ' | test accuracy: {}%'.format(100*correct))
        return correct


def cal_fisher(model, previous_loader):
    # 计算重要度矩阵
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}  # 模型的所有参数

    _means = {}  # 初始化要把参数限制在的参数域
    for n, p in params.items():
        _means[n] = p.clone().detach()

    precision_matrices = {}  # 重要度
    for n, p in params.items():
        precision_matrices[n] = p.clone().detach().fill_(0)  # 取zeros_like

    model.eval()
    for data, batch_y in previous_loader:
        model.zero_grad()
        data, batch_y = data.to(device), batch_y.to(device)
        output = model.forward(data)
        ############ 核心代码 #############
        loss = F.nll_loss(F.log_softmax(output, dim=1), batch_y)
        # 计算labels对应的(正确分类的)对数概率，并把它作为loss func衡量参数重要度
        loss.backward()  # 反向传播计算导数
        for n, p in model.named_parameters():
            precision_matrices[n].data += p.grad.data ** 2 / len(previous_loader)
        ########### 计算对数概率的导数，然后反向传播计算梯度，以梯度的平方作为重要度 ########

        return _means, precision_matrices


def ewc_train(model, optimizer, _means, precision_matrices, loader, lambda1, lambda2):
    model.train()
    model.zero_grad()
    loss_func = nn.CrossEntropyLoss()

    for step, (batch_x, batch_y) in enumerate(loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model.forward(batch_x)
        ce_loss = loss_func(outputs, batch_y)
        total_loss = ce_loss
        # 额外计算EWC的L2 loss
        ewc_loss = 0
        for n, p in model.named_parameters():
            _loss = precision_matrices[n] * (p - _means[n]) ** 2
            if n == 'conv.0.weight' or n == 'conv.0.bias':
                ewc_loss += lambda1 * _loss.sum()
            else:
                ewc_loss += lambda2 * _loss.sum()
        total_loss += ewc_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        predicted = torch.max(F.softmax(outputs, dim=1), dim=1)[1].cpu().numpy()
        label = batch_y.cpu().numpy()
        accuracy = sum(predicted == label) / label.size
        if (step + 1) % 5 == 0:
            print('epoch:', epoch, ' | train loss:%.4f' % total_loss.item(),
                  ' | test accuracy:{} %'.format(100 * accuracy))


def normal_train(model, optimizer, loader):
    model.train()
    model.zero_grad()
    loss_func = nn.CrossEntropyLoss()

    for step, (batch_x, batch_y) in enumerate(loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model.forward(batch_x)
        loss = loss_func(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predicted = model.forward(batch_x)
        predicted = torch.max(F.softmax(predicted, dim=1), dim=1)[1].cpu().numpy()
        label = batch_y.cpu().numpy()
        accuracy = sum(predicted == label) / label.size
        if (step + 1) % 20 == 0:
            print('epoch:', epoch, ' | train loss:%.4f' % loss.item(), ' | test accuracy:{} %'.format(100 * accuracy))


if __name__ == "__main__":
    text_cnn = TextCNN().to(device)
    optimizer = torch.optim.Adam(text_cnn.parameters(), lr=LR)
    loss_fuc = nn.CrossEntropyLoss()

    comp_train_loader = get_data('comp_train.npy', 'comp_train_label.npy')
    comp_test_loader = get_data('comp_test.npy', 'comp_test_label.npy')

    rec_train_loader = get_data('rec_train.npy', 'rec_train_label.npy')
    rec_test_loader = get_data('rec_test.npy', 'rec_test_label.npy')

    sci_train_loader = get_data('sci_train.npy', 'sci_train_label.npy')
    sci_test_loader = get_data('sci_test.npy', 'sci_test_label.npy')
    #
    talk_train_loader = get_data('talk_train.npy', 'talk_train_label.npy')
    talk_test_loader = get_data('talk_test.npy', 'talk_test_label.npy')

    x0 = range(20)
    x1 = range(20)
    x2 = range(20)
    x3 = range(20)

    acc_comp = []
    acc_rec = []
    acc_sci = []
    acc_talk = []

    for epoch in range(20):
        normal_train(text_cnn, optimizer, comp_train_loader)
        temp = valid(text_cnn, comp_test_loader)
        print('accuracy: {} % '.format(100 * temp))
        acc_comp.append(temp)

    for epoch in range(20):
        normal_train(text_cnn, optimizer, rec_train_loader)
        temp = valid(text_cnn, rec_test_loader)
        print('accuracy: {} % '.format(100 * temp))
        acc_rec.append(temp)

    for epoch in range(20):
        normal_train(text_cnn, optimizer, sci_train_loader)
        temp = valid(text_cnn, sci_test_loader)
        print('accuracy: {} % '.format(100 * temp))
        acc_sci.append(temp)

    for epoch in range(20):
        normal_train(text_cnn, optimizer, talk_train_loader)
        temp = valid(text_cnn, talk_test_loader)
        acc_talk.append(temp)
        print('accuracy: {} % '.format(100 * temp))


    plt.plot(x0, acc_comp, label='comp')
    plt.plot(x1, acc_rec, label='rec')
    plt.plot(x2, acc_sci, label='sci')
    plt.plot(x3, acc_talk, label='talk')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Multitask training without EWC')
    plt.show()


