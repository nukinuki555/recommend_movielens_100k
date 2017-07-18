# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from common.optimizer import *

#評価済みの要素の一定の割合を0にする
def delete_for_rmse(origin, rate=0.8):

    for line in origin:
        hoge = np.where(line != 0)
        n = len(hoge[0])
        foo = np.random.choice(n, int(n * rate))
        del_where = hoge[0][foo]
        line[del_where] = 0

    return origin

class AE_Trainer:
    """ニューラルネットの訓練を行うクラス
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='Adam', optimizer_param={'lr':0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        #AEなので入力データと教師データは同じ、インスタンス時に
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        #元データをオリジナルデータと、80%の要素を欠損させたtestデータとに分ける
        self.x_origin = self.x_train
        self.t_origin = self.t_train
        #評価されている配列の要素の場所を参照して、その全体の0.8だけ0にしてRMSE(accuracy)のテストデータとする
        self.x_rmse = delete_for_rmse(self.x_origin)
        self.t_rmse = delete_for_rmse(self.t_origin)
        # optimzer
        optimizer_class_dict = {'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        # self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.iter_per_epoch = 100
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print("train loss:" + str(loss))

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            #accはoriginのデータが評価していたものに対してのみ計算する、生成したデータからそ例外の項目を削除する操作
            x_train_sample, t_train_sample = self.x_origin, self.network.predict(self.x_rmse)
            x_test_sample, t_test_sample = self.t_origin, self.network.predict(self.t_rmse)
            ####test
            for i in range(x_train_sample.shape[0]):
                hoge = np.where(x_train_sample[i, :] == 0.0)
                hogehoge = np.where(x_test_sample[i, :] == 0.0)

                t_train_sample[i,  hoge[0]] = 0.0
                t_test_sample[i,  hogehoge[0]] = 0.0
                # n = len(hoge[0])
                # foo = np.random.choice(n, int(n * rate))
                # del_where = hoge[0][foo]
                # _del = np.array(origin)
                # _del[del_where] = 0
            # print('x_train_sample: ', x_train_sample, 't_train_sample: ', t_train_sample)
            # input('next is acc')
            ###test
            # if not self.evaluate_sample_num_per_epoch is None:
            #     t = self.evaluate_sample_num_per_epoch
            #     x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
            #     x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

    def show_graf_data(self):
        #グラフの描画
        # self.test_acc_list /= np.max()
        markers = {'train': 'o', 'test': 's'}
        x = np.arange(self.epochs)
        plt.plot(x, self.train_acc_list, marker='o', label='train', markevery=2)
        plt.plot(x, self.test_acc_list, marker='s', label='test', markevery=2)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        # plt.ylim(0, np.max(self.test_acc_list) + 1.0)
        plt.ylim(0, 3.0)
        plt.legend(loc='lower right')
        plt.show()
