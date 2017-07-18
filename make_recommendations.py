#coding UTF-8

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from load_movie_lens import loadMovielensByPandas
from load_movie_lens import get_movie_titles

def draw_3D_Scatter(path = 'outputs/'):
    train, test = loadMovielensByPandas()
    #出力した推測データを読み込み
    # data = np.loadtxt(path + "[BAE-output-2000k-L2500.csv", delimiter=",")
    data = test

    print('data.shape: ', data.shape)

    #
    shape_x, shape_y = data.shape
    sum = shape_x * shape_y
    x = np.arange(sum).reshape(shape_x, shape_y)
    for i in range(shape_x):
        x[i, :] = i + 1
    x.reshape(sum)

    y = np.arange(sum).reshape(shape_x, shape_y)
    for i in range(shape_x):
        y[i, :] = np.arange(1, shape_y + 1)

    z = data
    z.reshape(sum)

    # 3Dでプロット
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(x, y, z, "o", color="#00aa00", ms=4, mew=0.5)
    # 軸ラベル
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # 表示
    plt.show()

#CM用のRMSE
def RMSE(path = 'outputs/'):
    #CMに使った元データと、それを欠損させて生成させたデータを読み込む
    origin = np.loadtxt(path + 'CM_origin_test_movielens.csv', delimiter=",")
    test = np.loadtxt(path + 'CM_output_for_rmse_test_movielens.csv', delimiter=",")
    #互いに片方が0のところは0にする
    for i in range(origin.shape[0]):
        hoge = np.where(origin[i, :] == 0.0)
        hogehoge = np.where(test[i, :] == 0.0)

        test[i, hoge[0]] = 0.0
        origin[i, hogehoge[0]] = 0.0

    #RMSE部分
    sum_loss = np.sum(np.power(origin - test, 2))
    # originの評価値数
    n = np.count_nonzero(origin)
    # print('n: ', n)
    out = np.sqrt(sum_loss / n)
    # print('out: ', out)
    # input('next is return to train step')
    return out

def recommendations(path = 'outputs/', file_name = None, n=10):
    #出力結果読み込み
    file_name = "CM_output_test_movielens.csv"
    data = np.loadtxt(path + file_name, delimiter=",")
    # _, origin = loadMovielensByPandas()
    # #元データで評価されていたら無視
    # mask = origin != 0.0
    # data[mask] = 0.0
    #映画のタイトル辞書を読み込み
    movies_dict = get_movie_titles()
    x_shape = data.shape[0]
    y_shape = n
    # output = np.arange(x_shape * y_shape).reshape(x_shape, y_shape)
    output = [[0 for i in range(y_shape)] for j in range(x_shape)]
    for i in range(x_shape):
        line = data[i, :]
        where = np.argsort(line)
        where = where[::-1]
        for j in range(y_shape):
            output[i][j] = movies_dict[where[j] + 1]

    name = 'Recommendations_'
    # np.savetxt(name + file_name, output, delimiter=',')  # outputをcsvとして保存する
    # ファイルを書き込みモードでオープン
    with open(name + file_name, 'w') as f:

        writer = csv.writer(f)  # writerオブジェクトを作成
        # writer.writerow(header)  # ヘッダーを書き込む
        writer.writerows(output)  # 内容を書き込む

    return output





