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

# class model(nn.Module):
#     def __init__(self):
#         super(model, self).__init__()
#         self.dropout = nn.Dropout(Dropout)
#         self.conv = nn.Sequential(
#             nn.Conv1d(1, Filter_Num, (2, Embedding_Size)),
#             nn.ReLU(),
#             nn.MaxPool2d((max_sent - 1, 1))
#         )
#         self.fc = nn.Linear(Filter_Num, 4)

#     def forward(self, X):  # [batch_size, 63]
#         batch_size = X.shape[0]
#         X = X.unsqueeze(1)  # [batch_size, 1, 63, 50]
#         X = self.conv(X)  # [batch_size, 10, 1, 1]
#         X = X.view(batch_size, -1)  # [batch_size, 10]
#         X = self.fc(X)  # [batch_size, 2]
#         return X


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


def train(train_loader, new, old, optimizer, loss_fuc, pre_features, task_index):
    num_batches = len(train_loader)
    size = len(train_loader.dataset)
    avgloss = 0
    task_acc = 0
    for step, (cur_data, cur_label) in enumerate(train_loader):
        cur_data = cur_data.to(device)
        cur_label = cur_label.to(device) - 4 * (task_index - 1)
        # 这里需要将cur_label化为0-3以计算损失函数

        optimizer.zero_grad()
        # 得到新、旧模型输出
        output_new = new(cur_data.float())
        output_old = old(cur_data.float())
        # 计算loss1——新模型输出与结果的损失,T=1
        # output_new_val = F.softmax(output_new[:,pre_features:],dim=1)
        loss1 = loss_fuc(output_new[:, pre_features:], cur_label)

        # 计算loss2——新旧模型对旧任务输出的损失，softmax使用温度T
        output_new_pre = F.softmax(output_new[:, :pre_features] / T, dim=1)
        output_old_pre = F.softmax(output_old / T, dim=1)
        loss2 = output_old_pre.mul(-1 * torch.log(output_new_pre))
        loss2 = loss2.sum(1)
        loss2 = loss2.mean() * T * T

        # 计算总损失，其中alpha越小，表示保留旧模型越多
        loss = alpha * loss1 + (1 - alpha) * loss2
        avgloss += loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()

        task_acc += (output_new[:, pre_features:].cpu().argmax(1) == cur_label.cpu()).type(torch.float).sum().item()

    avgloss /= num_batches
    task_acc /= size
    print('while training task: %d ' % task_index)
    print('train loss:%.4f' % avgloss)
    print('train accuracy is %.4f' % task_acc)


def valid(test_loader, model, task_index):
    num_batches = len(test_loader)
    size = len(test_loader.dataset)
    avgloss = 0

    task_acc = 0
    with torch.no_grad():
        for step, (cur_data, cur_label) in enumerate(test_loader):
            cur_data = cur_data.to(device)
            cur_label = cur_label.to(device) - 4 * (task_index - 1)
            # 得到输出
            output = model(cur_data.float())
            # 计算loss
            output = F.softmax(output[:, (task_index - 1) * 4:task_index * 4], dim=1)

            loss = loss_fuc(output, cur_label)
            avgloss += loss.item()

            task_acc += (output.cpu().argmax(1) == cur_label.cpu()).type(torch.float).sum().item()

        avgloss /= num_batches
        task_acc /= size

        print('test accuracy is %.4f' % task_acc)


if __name__ == '__main__':
    # 加载数据
    comp_train = np.load('comp_train.npy');     comp_train_label = np.load('comp_train_label.npy')
    comp_test = np.load('comp_test.npy');       comp_test_label = np.load('comp_test_label.npy')
    rec_train = np.load('rec_train.npy');       rec_train_label = np.load('rec_train_label.npy')
    rec_test = np.load('rec_test.npy');         rec_test_label = np.load('rec_test_label.npy')
    sci_train = np.load('sci_train.npy');       sci_train_label = np.load('sci_train_label.npy')
    sci_test = np.load('sci_test.npy');         sci_test_label = np.load('sci_test_label.npy')
    talk_train = np.load('talk_train.npy');     talk_train_label = np.load('talk_train_label.npy')
    talk_test = np.load('talk_test.npy');       talk_test_label = np.load('talk_test_label.npy')
    # 封装成dataloader
    train_loader1 = get_data(comp_train, comp_train_label)
    test_loader1 = get_data(comp_test, comp_test_label)
    train_loader2 = get_data(rec_train, rec_train_label)
    test_loader2 = get_data(rec_test, rec_test_label)
    train_loader3 = get_data(sci_train, sci_train_label)
    test_loader3 = get_data(sci_test, sci_test_label)
    train_loader4 = get_data(talk_train, talk_train_label)
    test_loader4 = get_data(talk_test, talk_test_label)

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

    # 定义每次需要在输出层增加的类别数量
    new_class = 4
    # 依次训练rec、sci、talk任务
    for task_index in range(3):
        if task_index == 0:
            train_loader = train_loader2
            test_loader = test_loader2
        elif task_index == 1:
            train_loader = train_loader3
            test_loader = test_loader3
        else:
            train_loader = train_loader4
            test_loader = test_loader4
        # 拷贝模型并进行输出层的替换
        new_net = copy.deepcopy(pre_model)
        old_net = copy.deepcopy(pre_model)
        in_features = new_net.fc.in_features  # 提取new_net中最后一层的输入维数
        out_features = new_net.fc.out_features  # 提取new_net中最后一层的输出维数
        weight = new_net.fc.weight.data  # 提取最后一层的权重参数
        bias = new_net.fc.bias.data  # 提取最后一层的偏置参数

        new_out_features = out_features + new_class  # 新模型输出维数
        new_fc = nn.Linear(in_features, new_out_features)  # 创建新的输出层
        new_fc.weight.data[:out_features] = weight
        new_fc.bias.data[:out_features] = bias
        new_net.fc = new_fc
        new_net = new_net.to(device)
        old_net = old_net.to(device)

        # 这里需要重新定义优化器（用新模型的参数）
        optimizer = torch.optim.Adam(new_net.parameters(), lr=LR)
        for epoch in range(Epochs):
            print('comp Epoch =%d' % epoch)
            train(train_loader, new_net, old_net, optimizer, loss_fuc, out_features, task_index + 2)  # 传入2，3，4
            if task_index == 0:
                valid(test_loader1, new_net, 1)
                valid(test_loader2, new_net, 2)
            elif task_index == 1:
                valid(test_loader1, new_net, 1)
                valid(test_loader2, new_net, 2)
                valid(test_loader3, new_net, 3)
            else:
                valid(test_loader1, new_net, 1)
                valid(test_loader2, new_net, 2)
                valid(test_loader3, new_net, 3)
                valid(test_loader4, new_net, 4)
        # 新旧更替
        pre_model = copy.deepcopy(new_net)

    for task_index in range(4):
        if task_index == 0:
            test_loader = test_loader1
        elif task_index == 1:
            test_loader = test_loader2
        elif task_index == 2:
            test_loader = test_loader3
        else:
            test_loader = test_loader4
        print('testing task %d' % (task_index + 1))
        valid(test_loader, new_net, task_index + 1)  # 传入1,2,3,4