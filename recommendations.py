#coding UTF-8
import numpy as np
import math
from common.AE_trainer import delete_for_rmse
#二つの座標の類似度を算出する関数軍

#p1とp2のユークリッド距離を返す
def sim_distance(prefs, p1, p2):
    return 0

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

#p1とp2のピアソン係数を返す
def sim_pearson(prefs, p1, p2):
    #両者が互いに評価しているアイテムのリストを取得
    si = {}
    for item in prefs[p1]:
        if item in prefs[p1]:
            si[item] = 1
    #要素の数の調べる
    n = len(si)

    #共に評価しているアイテムがなければ0を返す
    if n == 0:
        return 0


    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])

    #平方を合計する
    sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])

    #積を合計する
    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])

    #ピアソンによるスコアを計算する
    num = pSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1, 2)/n) * (sum2q - pow(sum2, 2) / n))
    if den == 0:
        return 0

    r = num / den
    return r

#movielens用に改変版：p1とp2のピアソン係数を返す
def sim_pearson_movielens(prefs, p1, p2):
    #prefsは縦軸：user_id, 横軸：item_idで評価が書いてあるデータ配列
    #p1, p2はユーザidの番号
    #両者が互いに評価しているアイテムのリストを取得
    si = []
    for i in range(prefs.shape[1]):
        if prefs[p1, i] != 0.0 and prefs[p2, i] != 0.0:
            si.append(i)
        #要素の数の調べる
    n = len(si)

    #共に評価しているアイテムがなければ0を返す
    if n == 0:
        return 0


    # sum1 = sum([prefs[p1, it] for it in si])
    # sum2 = sum([prefs[p2, it] for it in si])

    sum1 = np.sum(prefs[p1, si])
    sum2 = np.sum(prefs[p2, si])

    #平方を合計する
    # sum1Sq = sum([pow(prefs[p1, it], 2) for it in si])
    # sum2Sq = sum([pow(prefs[p2, it], 2) for it in si])
    sum1Sq = np.sum(pow(prefs[p1, si], 2))
    sum2Sq = np.sum(pow(prefs[p2, si], 2))

    #積を合計する
    # pSum = sum([prefs[p1, it] * prefs[p2, it] for it in si])
    pSum = np.sum(prefs[p1, si] - prefs[p2, si])

    #ピアソンによるスコアを計算する
    num = pSum - (sum1 * sum2 / n)
    den = math.sqrt((sum1Sq - pow(sum1, 2)/n) * (sum2Sq - pow(sum2, 2) / n))
    if den == 0:
        return 0

    r = num / den
    return r


def cross_entropy(item1, item2):
    #item1とitem2のどちらも評価済みユーザの集合
    common = np.logical_and(item1 != 0, item2 != 0)
    v1 = item1[common]
    v2 = item2[common]

    sim = 0.0
    #共通評価者が2以上と言う制約にしている
    if v1.size > 1:
        sim = 1.0 - cos_sim(v1, v2)

    return sim

#自分で改変したやつ
def cross_entropy02(item1, item2):
    #item1とitem2のどちらも評価済みユーザの集合
    # common = np.logical_and(item1 != 0, item2 != 0)
    si = []
    for i in range(item1.shape[0]):
        if item1[i, :] != 0.0 and item2[i,:] != 0.0:
            si.append(i)
        #要素の数の調べる
    n = len(si)

    #共に評価しているアイテムがなければ0を返す
    if n == 0:
        return 0
    v1 = item1[si]
    v2 = item2[si]

    sim = 0.0
    #共通評価者が2以上と言う制約にしている
    if n > 1:
        sim = 1.0 - cos_sim(v1, v2)

    return sim

def compute_item_similarities(R, similarity=cross_entropy):
    #n: movie counts
    n = R.shape[1]
    sims = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim = 1.0
            else:
                #R[:, i]はアイテムiに関する全ユーザの評価を並べた列ベクトル
                sim = similarity(R[:, i], R[:, j])
            sims[i][j] = sim
            sims[j][i] = sim
    return sims

#自分で改変したやつ
def compute_item_similarities02(R, similarity=cross_entropy):
    #n: movie counts
    n = R.shape[1]
    sims = np.zeros((n, n))
    R_t = R.T
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim = 1.0
            else:
                #R[:, i]はアイテムiに関する全ユーザの評価を並べた列ベクトル
                sim = similarity(R_t, i, j)
            sims[i][j] = sim
            sims[j][i] = sim
    return sims

#あるuser_idに対して、(類似度 * 評価)/類似度の合計でまだ評価していない映画の評価を予測する
def predict(u, sims):
    #未評価は0,評価済みは1となるベクトル.normalizersの計算のために
    x = np.zeros(u.size)
    x[u > 0] = 1

    scores = sims.dot(u)
    normalizers = sims.dot(x)

    prediction = np.zeros(u.size)
    for i in range(u.size):
        #分母が0になるケースと評価済みアイテムは予測値を0とする
        if normalizers[i] == 0 or u[i] > 0:
            prediction[i] = 0
        else:
            prediction[i] = scores[i] / normalizers[i]
    #ユーザuのアイテムiに対する評価の予測
    return prediction

#意味ない
def accuracy(u, prediction):
    val = u != 0.0
    not_val = u == 0.0
    val_u = u[val]
    val_pre = u[val]
    error = val_u - val_pre
    error = np.sqrt(np.sum(pow(error, 2)))
    n = val_u.shape[0]
    acc = (1 - error/(5.0 * n))

    return acc

#テスト評価

#ディクショナリprefsから、personにもっともマッチする者たちを返す
def topMatches(prefs, person, n=5, similarity=sim_pearson):
    scores = [(similarity(prefs, person, other), other) \
              for other in prefs if other != person]
    #高スコアがリストの最初にくるように並び替える
    scores.sort()
    scores.reverse()
    return scores[0:n]

#person以外の全ユーザの氷点の重み付き平均を使い、personへの推薦を算出する
def getRecommendations(prefs, person, similarity=sim_pearson):
    totals = {}
    simSums = {}
    for other in prefs:
        #自分自身とは比較しない
        if other == person:
            continue
        sim = similarity(prefs, person, other)

        #0以外のスコアは無視する
        if sim <= 0:
            continue

        for item in prefs[other]:
            #まだ見ていない映画の得点のみを算出
            if item not in prefs[person] or prefs[person][item] == 0:
                #類似度　* スコア
                totol.setdefault(item, 0)
                total[item] += prefs[other][item] * sim
                #類似度を計算
                simSums.setdefault(item, 0)
                simSums[item] += sim

    #正規化したリストを作る
    rankings = [(total/simSums[item], item) for item, total in totlals.item()]

    #ソート済みのリストを返す
    rankings.sort()
    rankings.reverse()
    return rankings



#人々とアイテムを交換したディクショナリにする
def transformPrefs(prefs):
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            #itemとpersonを入れ替える
            result[item][person] = prefs[person][item]
    return result

#movielens用に改変版：人々とアイテムを交換したリストにする
def transformPrefs(prefs):
    return prefs.T #numpyを想定


#アイテムベースのフィルタリング
def calculatesSimilarItems(prefs, n=10):
    #アイテムをキーとして持ち、それぞれのアイテムににている
    #アイテムのリストを値をして持つディクショナリを作る
    result = {}

    #嗜好の行列をアイテム中心な形に反転させる
    itemPrefs = transformPrefs(prefs)
    c = 0
    for item in itemPrefs:
        #巨大なデータセット用にステータスを表示
        c += 1
        if c % 100 == 0:
            print('%d' / '%d' % (c, len(itemPrefs)))
        #このアイテムに最も似ているアイテムたちを探す
        scores = topMatches(itemPrefs, item, n=n, similarity=sim_person)
        result[item] = scores
    return result

def list_index(list):
    index = []
    for i in range(len(list)):
        if list[i] == True:
            index.append(i)
    return index

def main():
    from load_movie_lens import loadMovielensByPandas as f
    #訓練データとテストデータを分ける
    train, test = f(com_miss_flag=False)
    test_size, test_items_size = test.shape
    train = train[:, : test_items_size]

    # print('R.shape; ', R.shape)
    print('train.shape: ', train.shape)
    print('test.shape: ', test.shape)

    #要素の80&を削除してちゃんと推測しているか試す
    # sample = test
    # sample = delete_for_rmse(sample)

    sim = compute_item_similarities(test)#類似度計算
    #訓練データに対しての予測、正確性テスト
    # acc = []
    output = []
    i = 0
    for user in train:
        i += 1
        # val = user != 0
        # val_list = list_index(val)
        # # val_list = val_list[::10]
        check = user
        # check[val_list] = 0
        prediction = predict(check, sim)
        output.append(prediction)
        # acc_data = accuracy(user, prediction)
        # acc.append(acc_data)

        # if i % 10 == 0:
        #     print('i : ', i)
    # sum_acc = np.sum(acc) / train.shape[0]
    # print('sum_acc: ', sum_acc)
    print('CM has finished')
    output = np.array(output)
    np.savetxt('similarity_test__movielens.csv', sim, delimiter=',')
    np.savetxt('CM_origin_test_movielens.csv',test , delimiter=',') #outputをcsvとして保存する
    np.savetxt('CM_output_test_movielens.csv',output , delimiter=',') #outputをcsvとして保存する




if __name__ == '__main__':
    main()
