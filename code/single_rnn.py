from __future__ import print_function
import math
import torch.nn as nn
import copy
import torch.nn.functional as F
import torch
import torch.utils.data as Data
import numpy as np
# 超参数
Batch_Size = 128
Embedding_Size = 100 # 词向量维度
Filter_Num = 30 # 卷积核个数
Dropout = 0.5
Epochs = 20
LR = 0.01
max_sent=250
T = 2  # 新旧模型比较的”温度“
alpha = 0.95  # alpha越小，旧模型需要保存越多
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device=",device)


def get_data(data, label):
    x = torch.from_numpy(data) # [369, 63]
    y = torch.from_numpy(label) # [369]
    y=torch.tensor(y).long()
    dataset = Data.TensorDataset(x, y)
    data_loader = Data.DataLoader(dataset=dataset, batch_size=Batch_Size,shuffle=True)
    return data_loader


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.lstm_layer = nn.LSTM(input_size=100, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
        # self.lstm_layer = nn.RNN(input_size=100, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
        # self.lstm_layer = nn.GRU(input_size=100, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(256,4)
        self.dropout = nn.Dropout(0.5)
        self.hidden_cell = None

    def attention_net(self,x, query,mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores, dim=-1)
        context = torch.matmul(alpha_n, x).sum(1)
        return context, alpha_n

    def forward(self, input):
        lstm_out, self.hidden_cell = self.lstm_layer(input)
        # output = lstm_out.permute(1,0,2)
        query = self.dropout(lstm_out)
        attn_output, alpha_n = self.attention_net(lstm_out, query)
        logit = self.fc(attn_output)
        return logit


def pre_train(train_loader, model, optimizer, loss_fuc):
    size = len(train_loader.dataset)
    num_batches = len(train_loader)
    avgloss,avgacc=0,0
    for step, (cur_data, cur_label) in enumerate(train_loader):
        cur_data = cur_data.to(device)
        cur_label = cur_label.to(device)
        # 前向传播
        predicted = model.forward(cur_data.float())
        # predicted = F.softmax(predicted,dim=1)
        loss = loss_fuc(predicted, cur_label)
        avgloss+=loss.item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算accuracy
        # dim=0表示取每列的最大值，dim=1表示取每行的最大值
        # torch.max()[0]表示返回最大值 torch.max()[1]表示返回最大值的索引
        predicted = torch.max(predicted, dim=1)[1].cpu().numpy()
        label = cur_label.cpu().numpy()

        accuracy = sum(predicted == label)
        avgacc+=accuracy

    avgloss/=num_batches
    avgacc/=size
    print('while training task 1:')
    print('train loss:%.4f' % avgloss, ' | train accuracy:', avgacc)




def pre_valid(test_loader, model):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for cur_data, cur_label in test_loader:
            cur_data = cur_data.to(device)
            cur_label = cur_label.to(device)
            output = model.forward(cur_data.float())
            output = F.softmax(output,dim=1)
            # 损失函数参数一为网络输出，参数二为真实标签，size=[batch_size]
            test_loss += loss_fuc(output, cur_label).item()
            # print(output)
            # 准确率增加
            output = torch.max(output, dim=1)[1].cpu().numpy()
            label = cur_label.cpu().numpy()
            correct += sum(output == label)

        test_loss /= num_batches
        correct /= size
        print('test loss:%.4f' % test_loss, ' | test accuracy:', correct)


if __name__ == '__main__':
    # 加载数据
    comp_train = np.load('comp_train.npy');     comp_train_label = np.load('comp_train_label.npy')
    comp_test = np.load('comp_test.npy');       comp_test_label = np.load('comp_test_label.npy')
    # 封装成dataloader
    train_loader1 = get_data(comp_train, comp_train_label)
    test_loader1 = get_data(comp_test, comp_test_label)

    # 先用comp训练初始网络
    pre_model = model().to(device)
    # 定义优化器以及损失函数
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=LR)
    loss_fuc = nn.CrossEntropyLoss()

    # 开始训练初始模型（第一个任务）
    for epoch in range(Epochs):
        print('comp Epoch =%d' % epoch)
        pre_train(train_loader1, pre_model, optimizer, loss_fuc)
        pre_valid(test_loader1, pre_model)
    print("single task done!")