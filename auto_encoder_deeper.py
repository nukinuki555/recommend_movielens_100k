#coding UTF-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from common.layers import *
# from common.gradient import numerical_gradient
from collections import OrderedDict

#AEの層を5層にして深くしたもの(隠れ層は全て同じサイズでBNとdrpooutはあり)
class AutoEocoderDeep:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, drop_rotation=0.5, use_batchnorm=False):
        self.use_batchnorm = use_batchnorm
        #重みの初期化
        self.params = {}
        # self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # scale = {}
        scale_W1 = np.sqrt(2.0 / input_size)
        scale_W2 = np.sqrt(2.0 / hidden_size)
        scale_W3 = np.sqrt(2.0 / hidden_size)
        scale_W4 = np.sqrt(2.0 / hidden_size)
        self.params['W1'] = scale_W1 * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = scale_W2 * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = scale_W3 * np.random.randn(hidden_size, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)
        self.params['W4'] = scale_W4 * np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)

        #レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        if use_batchnorm == True:
            self.params['gamma1'] = np.ones(hidden_size)
            self.params['beta1'] = np.ones(hidden_size)
            self.layers['BatchNorm1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu1'] = Relu()
        self.layers['Dropout1'] = Dropout(drop_rotation)
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        if use_batchnorm == True:
            self.params['gamma2'] = np.ones(hidden_size)
            self.params['beta2'] = np.ones(hidden_size)
            self.layers['BatchNorm2'] = BatchNormalization(self.params['gamma2'], self.params['beta2'])
        self.layers['Relu2'] = Relu()
        self.layers['Dropout2'] = Dropout(drop_rotation)
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        if use_batchnorm == True:
            self.params['gamma3'] = np.ones(hidden_size)
            self.params['beta3'] = np.ones(hidden_size)
            self.layers['BatchNorm3'] = BatchNormalization(self.params['gamma3'], self.params['beta3'])
        self.layers['Relu3'] = Relu()
        self.layers['Dropout3'] = Dropout(drop_rotation)
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
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
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db
        grads['W4'] = self.layers['Affine4'].dW
        grads['b4'] = self.layers['Affine4'].db
        if self.use_batchnorm:
                grads['gamma1'] = self.layers['BatchNorm1'].dgamma
                grads['beta1'] = self.layers['BatchNorm1'].dbeta
                grads['gamma2'] = self.layers['BatchNorm2'].dgamma
                grads['beta2'] = self.layers['BatchNorm2'].dbeta
                grads['gamma3'] = self.layers['BatchNorm3'].dgamma
                grads['beta3'] = self.layers['BatchNorm3'].dbeta

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
