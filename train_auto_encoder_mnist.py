#coding UTF-8
#オートエンコーダでmnist画像を生成する

import numpy as np
from data.mnist.mnist import load_mnist
from auto_encoder_layer_mnist import AutoEocoder
import matplotlib.pyplot as plt

#データの読み込み
(x_train, x_test), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = AutoEocoder(input_size=784, hidden_size=500, output_size=784, drop_rotation=0.5)

iters_num = 10000
train_size = x_train.shape[0]

batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
max_epochs = iters_num / iter_per_epoch

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
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #1epoch毎に、誤差と出力されたmnist画像を表示する
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, x_train)
        test_acc = network.accuracy(x_test, x_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

print('learning has finished')
#データの例示
# outpux_test = network.predict(x_train[:81])
output_test = network.predict(x_test[:81])
#MNISTデータの表示
# fig_train = plt.figure(figsize=(9, 9))
fig_test = plt.figure(figsize=(9, 9))
# fig_train.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)
fig_test.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)
for i in range(81):
    # ax_train = fig_train.add_subplot(9, 9, i + 1, xticks=[], yticks=[])
    ax_test = fig_test.add_subplot(9, 9, i + 1, xticks=[], yticks=[])
    # ax_train.imshow(output_train[i].reshape((28, 28)), cmap='gray')
    ax_test.imshow(output_test[i].reshape((28, 28)), cmap='gray')
plt.show()
print('generated data was showed')
row_input()
# #グラフの描画
# markers = {'train': 'o', 'test': 's'}
# x = np.arange(max_epochs)
# plt.plot(x, train_acc_list, marker='o', label='train', markevery=2)
# plt.plot(x, test_acc_list, marker='s', label='test', markevery=2)
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.ylim(0, np.max(test_acc_list) + 1.0)
# plt.legend(loc='lower right')
# plt.show()

print('all has finished')
