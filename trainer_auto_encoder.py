# coding: utf-8
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from load_movie_lens import loadMovielensByPandas
from auto_encoder_layer_mnist import AutoEocoder
# from auto_encoder_deeper import AutoEocoderDeep
from common.AE_trainer import AE_Trainer

#データの読み込み
(x_train, x_test) = loadMovielensByPandas(com_miss_flag=True)
test_size, test_items_size = x_test.shape
x_train = x_train[:, : test_items_size]

print('x_trian.shape: ', x_train.shape, 'x_test.shape: ', x_test.shape)

train_size, train_items_size  = x_train.shape
network = AutoEocoder(input_size=train_items_size, hidden_size=3000, output_size=train_items_size, drop_rotation=0.5, use_batchnorm=True)
iters_num = 5500

#ハイパーパラメータ
batch_size = 100
learning_rate = 0.01

# iter_per_epoch = max(train_size / batch_size, 1)
iter_per_epoch = 100
max_epochs = iters_num / iter_per_epoch

# 学習経過を保存
# train_loss_list = []
# train_acc_list = []
# test_acc_list = []

trainer = AE_Trainer(network, x_train, x_train, x_test, x_test,
                  epochs=max_epochs, mini_batch_size=batch_size,
                  optimizer='Adam', optimizer_param={'lr':learning_rate},
                  evaluate_sample_num_per_epoch=iter_per_epoch)
trainer.train()

trainer.show_graf_data()
# raw_input('graf is showing')
print('learning has finished')
# pre_test = network.predict(x_test) #予想点数
output = network.predict(x_test) #予想点数
# output = []
# for u_test, u_prediction in x_test, pre_test:
#     output.append(u_test)
#     output.append(u_prediction)
# for user in pre_test:
#     output.append(user)
# output = np.array(output) #リストに保存していた教師データ-予測データをnumpy配列にする

#もともとのデータで評価済みの映画は、評価0にしておく
# mask = (x_test !=  0.0)
# output[mask] = 0.0
name = input('出力結果の保存するファイル名を入力してください')
np.savetxt(name + '.csv',output , delimiter=',') #outputをcsvとして保存する


# パラメータの保存
weight_name = input('重みのファイル名を入力してください')
network.save_params(weight_name + ".pkl")
print("Saved Network Parameters!")
