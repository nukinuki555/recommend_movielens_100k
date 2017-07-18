#coding UTF-8
#オートエンコーダで評点を生成する

import numpy as np
from load_movie_lens import loadMovielensByPandas
from auto_encoder_layer_mnist import AutoEocoder
import matplotlib.pyplot as plt
from recommendations import predict

#データの読み込み
(x_train, x_test) = loadMovielensByPandas(com_miss_flag=True)
# train_size, train_items_size  = x_train.shape
# x_test = x_test[: , : train_items_size]
#x_testのitem_sizeがよりx_trainよりも大きいので、x_testのアイテム項目数に合わせて、x_trainを読み込む
test_size, test_items_size = x_test.shape
x_train = x_train[:, : test_items_size]

print('x_trian.shape: ', x_train.shape, 'x_test.shape: ', x_test.shape)

#ネットワークの初期設定
train_size, train_items_size  = x_train.shape
network = AutoEocoder(input_size=train_items_size, hidden_size=300, output_size=train_items_size, drop_rotation=0.5)
iters_num = 1000

#ハイパーパラメータ
batch_size = 10
learning_rate = 0.1

# iter_per_epoch = max(train_size / batch_size, 1)
iter_per_epoch = 100
max_epochs = iters_num / iter_per_epoch

#学習経過を保存
train_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = x_batch

    #誤差逆伝播法によって勾配を求める
    grad = network.gradient(x_batch, t_batch)

    #更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    #誤差を算出する
    # loss = network.loss(x_batch, t_batch)
    # train_loss_list.append(loss)
    # print('i: ', i, 'loss: ', loss)

    #1epoch毎に、誤差と出力されたmnist画像を表示する
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, x_train)
        test_acc = network.accuracy(x_test, x_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('i: ', i, 'train_acc: ', train_acc, 'test_acc: ', test_acc)
#
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
val = x_test == 0.0
output[val] = 0.0
np.savetxt('AE_output_movielens.csv',output , delimiter=',') #outputをcsvとして保存する


#グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, np.max(test_acc_list) + 1.0)
plt.legend(loc='lower right')
plt.show()

print('all has finished')
