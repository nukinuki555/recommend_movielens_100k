#coding UTF-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from common.layers import *
# from common.gradient import numerical_gradient
from collections import OrderedDict

class AutoEocoder:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, drop_rotation=0.5, use_batchnorm=False):
        self.use_batchnorm = use_batchnorm
        #重みの初期化
        self.params = {}
        # self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # scale = {}
        scale_W1 = np.sqrt(2.0 / input_size)
        scale_W2 = np.sqrt(2.0 / hidden_size)
        self.params['W1'] = scale_W1 * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = scale_W2 * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        if use_batchnorm == True:
            self.params['gamma'] = np.ones(hidden_size)
            self.params['beta'] = np.ones(hidden_size)
            self.layers['BatchNorm'] = BatchNormalization(self.params['gamma'], self.params['beta'])
        self.layers['Relu'] = Relu()
        self.layers['Dropout'] = Dropout(drop_rotation)
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastlayer = IdentityMapingWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    #x: 入力データ、t: 教師データ
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)

    #RMSE(二乗誤差平均の平方根)
    def accuracy(self, origin, test):
        # print('origin: ', origin, 'test: ', test)
        # input('in acc func')
        sum_loss = np.sum(np.power(origin - test, 2))
        # print('sum_loss: ', sum_loss)
        #originの評価値数
        n = np.count_nonzero(origin)
        # print('n: ', n)
        out = np.sqrt(sum_loss/n)
        # print('out: ', out)
        # input('next is return to train step')
        return out
        # return np.sqrt((2.0 * self.loss(x, t)) / x.shape[1] )

    #x: 入力データ, t: 教師データ
    def gradient(self, x, t):
        #forward
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #設定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        if self.use_batchnorm:
                grads['gamma'] = self.layers['BatchNorm'].dgamma
                grads['beta'] = self.layers['BatchNorm'].dbeta


        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
