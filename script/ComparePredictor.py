#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
  @ Time     : 19-4-16 下午3:35
  @ Author   : Vodka
  @ File     : ComparePredictor.py
  @ Software : PyCharm Community Edition
"""

import sys

import torch
# import matplotlib.pyplot as plt
from torch import nn, optim

reload(sys)
sys.setdefaultencoding('utf8')

# 定义超参数
batch_size = 200
learning_rate = 1e-3
num_epoches = 1000
total_num = 10166
num_train = 8133
num_test = 2033


# 定义 Recurrent Network 模型
class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)

    def forward(self, x):
        """
        Forward propagation
        :param x: 
        :return: 
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return out

    def saveModel(self, model, filename):
        """
        Save model as one file
        :param model:
        :param filename:
        :return:
        """
        torch.save(model, filename)

    def train(self, model, X, Y, num_case):
        """
        Train for the model
        :param X: 
        :param Y: 
        :param num_case: 
        :return: 
        """
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        update_time = int(num_case / batch_size)
        # Begin to train
        loss_data = []
        for epoch in range(num_epoches):
            for j in range(update_time):
                X_ = (X[j * batch_size:num_case, :, :]) if (j == update_time - 1) else (
                    X[j * batch_size:(j + 1) * batch_size, :, :])
                Y_ = (Y[j * batch_size:num_case, :]) if (j == update_time - 1) else (
                    Y[j * batch_size:(j + 1) * batch_size, :])
                running_loss = 0.0
                running_acc = 0.0
                # forward
                X_ = X_.reshape(num_case - j * batch_size, 1, X_.shape[2]) if (j == update_time - 1) else (
                    X_.reshape(batch_size, 1, X_.shape[2]))
                Y_ = Y_.reshape(num_case - j * batch_size, Y_.shape[1]) if (j == update_time - 1) else (
                    Y_.reshape(batch_size, Y_.shape[1]))
                out = model(X_)
                loss = criterion(out, Y_)
                _, pred = torch.max(out, 1)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_data.append(loss.item())
                print("The {}th train， update {:.4}%  Loss: {}".format(epoch + 1, float((j + 1) / float
                (update_time)) * 100.0, loss.item()))
        # plt.figure()
        # iter = [i for i in range(len(loss_data))]
        # plt.plot(iter, loss_data)
        # plt.legend()
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.show()

        model.saveModel(model, 'model/compare_predictor.model')
#
#
# if __name__ == '__main__':
#     model = Rnn(50, 200, 2)
#     X = np.load('data/net_train_x.npy')  # 10166*50
#     Y = np.load('data/net_train_y.npy')  # 10166*200
#     X = X.reshape(X.shape[0],1,X.shape[1])
#     X = X.astype('float32')
#     Y = Y.astype('float32')
#     X = torch.from_numpy(X)
#     Y = torch.from_numpy(Y)
#     model.train(model,X,Y,total_num)
#
#     model = torch.load('model/compare_predictor.model')
#     # Test
#     _label = Y[-num_test:, :]
#     _input = X[-num_test:, :, :]
#     _input = torch.from_numpy(_input)
#     pred = model(_input)
#     _, out1 = torch.max(pred, 1)
#     out2 = np.argmax(_label, 1)
#     cnt = 0
#     for i in range(_label.shape[0]):
#         if out1.numpy()[i] == out2[i]:
#             cnt += 1
#     print cnt
