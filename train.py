#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Winter_Olympics_commentary_sentiment_classification 
@File ：train.py
@IDE  ：PyCharm 
@Author ：Eastforward
@Date ：2022/2/16 12:43 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import *
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb

print(torch.cuda.get_device_name(0))
# import numpy
# print(numpy.__version__)
# pip install --ignore-installed --upgrade tensorflow-gpu

MAX_WORDS = 89527  # 词汇表的大小
MAX_LEN = 200
BATCH_SIZE = 256
EMB_SIZE = 128
HID_SIZE = 128  # LSTM hidden size
DROPOUT = 0.2
DEVICE = torch.device("cuda:0")


def load_data():
    # 借助keras 加载imdb数据集
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
    x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")
    print(x_train.shape, x_test.shape)

    # 使用DataLoader加载数据
    # 转化为TensorDataset
    train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))

    # 转化为DataLoader
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    test_sampler = RandomSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    return train_loader, test_loader


# 定义LSTM模型用于文本二分类
class Model(nn.Module):
    def __init__(self, max_words, emb_size, hid_size, dropout):
        super(Model, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        self.Embedding = nn.Embedding(self.max_words, self.emb_size)
        self.LSTM = nn.LSTM(self.emb_size, self.hid_size, num_layers=2,
                            batch_first=True, bidirectional=True)  # 两层双向LSTM
        self.dp = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 2)

    def forward(self, x):
        """
        input : [bs,max_len]
        output : [bs,2]
        """
        x = self.Embedding(x)  # [bs,ml,emb_size]
        x = self.dp(x)
        x, _ = self.LSTM(x)  # [bs,ml,2*hid_size]
        x = self.dp(x)
        x = F.relu(self.fc1(x))
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()  # [bs, 1, hid_size] => [bs, hid_size]
        out = self.fc2(x)  # [bs,2]
        return out


def decode(text):
    # text = "<START> Enjoy your time in Beijing"
    # A dictionary mapping words to an integer index
    word_index = imdb.get_word_index()
    # The first indices are reserved
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# 定义训练函数和测试函数
def train(model, device, train_loader, optimizer, epoch):
    """
    训练模型
    :param model:
    :param device:
    :param train_loader:
    :param optimizer:
    :param epoch:
    :return:
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (x, y) in enumerate(train_loader):
        # print(x[0])
        # print(decode(x[0].cpu().numpy()))
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_ = model(x)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')  # 累加Loss
    test_loss = 0.0
    acc = 0
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            y_ = model(x)
        test_loss += criterion(y_, y)
        pred = y_.max(-1, keepdim=True)[1]  # .max() 2输出，分别为最大值和最小值的index
        acc += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, acc, len(test_loader.dataset),
                                                                               100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset)


def start_train(train_loader, test_loader):
    # 开始模型的训练（并保存最优模型权重），训练较快，2min左右
    model = Model(MAX_WORDS, EMB_SIZE, HID_SIZE, DROPOUT).to(DEVICE)
    print(model)
    optimizer = optim.Adam(model.parameters())

    best_acc = 0.0
    PATH = 'imdb model/model.pth'

    for epoch in range(1, 20 + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        acc = test(model, DEVICE, test_loader)
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), PATH)
        print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))


def evaluation(test_loader):
    PATH = 'imdb model/model.pth'
    best_model = Model(MAX_WORDS, EMB_SIZE, HID_SIZE, DROPOUT).to(DEVICE)
    best_model.load_state_dict(torch.load(PATH))
    test(best_model, DEVICE, test_loader)


if __name__ == '__main__':
    train_loader, test_loader = load_data()
    start_train(train_loader, test_loader)
    evaluation(test_loader)
