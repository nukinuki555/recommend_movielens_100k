#coding UTF-8
import numpy as np
import pandas as pd
from scipy import sparse
import chardet

import csv
import operator
from math import sqrt, pow
from operator import itemgetter
from itertools import combinations
from collections import namedtuple, defaultdict

#使ってない
def loadMovieLens(path='./data/movielens'):

    #映画のタイトルを得る
    movies = {}
    for line in open(path+'/u_item_by_exel.txt'):
        (id, title) = line.split('|')[0:2]
        str_id = ''
        for i in id:
            if i != '\"':
                str_id += i
        int_id = int(str_id)
        movies[int_id] = title

    #データの読み込み
    # prefs = {}
    # for line in open(path+'/u.data'):
    #     (user, movieid, rating, ts) = line.split('\t')
    #     prefs.setdafault(user, {})
    #     prefs[user][movies[movieid]] = float(rating)
    return movies

#これで映画のタイトル受け取れる
def get_movie_titles(path='./data/movielens'):
    # 映画のタイトルを得る
    movies = {}
    for line in open(path + '/u_item_decoded.txt'):
        (id, title) = line.split('|')[0:2]
        #idに紛れ込んでる"のフィルター処理
        str_id = ''
        for i in id:
            if i != '\"':
                str_id += i
        int_id = int(str_id)
        movies[int_id] = title
    return movies

#これでu.itemのよくわからん文字コードをunicodeに直せた
def change_to_unicode(path='./data/movielens'):
    #バイナリモードで映画のタイトルファイルを読み込み
    #decode_to_unicode = None
    with open(path + '/u_item_by_exel.txt', mode='rb') as f:
         binary = f.read()
         #u.itemのencodingを読み取って、それでdecode
         file_info = chardet.detect(binary)
         encoding = file_info['encoding']
         decode_to_unicode = binary.decode(encoding)
         #ファイルを書き込み
         f2 = open('u_item_decoded.txt', 'w')
         f2.write(decode_to_unicode)
         f2.close()

    # movies = {}
    # for line in decode_to_unicode:
    #     (id, title) = line.split('|')[0:2]
    #     str_id = ''
    #     for i in id:
    #         if i != '\"':
    #             str_id += i
    #     int_id = int(str_id)
    #     movies[int_id] = title

    # return movies


#映画の評価データはこれで読み込みできる
def loadMovielensByPandas(path='./data/movielens', com_miss_flag=True):
    #pandasでmovielens-100kデータの読み込み
    df_train = pd.read_csv(path + '/ua.base', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    shape = (df_train.max().ix['user_id'], df_train.max().ix['item_id'])
    R_train = np.zeros(shape)

    df_test = pd.read_csv(path + '/ua.test', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    shape = (df_test.max().ix['user_id'], df_test.max().ix['item_id'])
    R_test = np.zeros(shape)

    #pandasで読み込んだデータをnumpy配列になおす(user-itemの二次元配列)
    for i in df_train.index:
        row = df_train.ix[i]
        R_train[row['user_id'] - 1, row['item_id'] - 1] = row['rating']

    for i in df_test.index:
        row = df_test.ix[i]
        R_test[row['user_id'] - 1, row['item_id'] - 1] = row['rating']

    #com_miss_flagがTrueならユーザ平均に則って、欠損値補完を行う
    if com_miss_flag == True:
        for user in R_train:
            val = user != 0.0
            not_val = user == 0.0
            u_com_miss = np.sum(user[val]) / len(user[val])
            user[not_val] = u_com_miss

        for user in R_test:
            val = user != 0.0
            not_val = user == 0.0
            u_com_miss = np.sum(user[val]) / len(user[val])
            user[not_val] = u_com_miss

    return R_train, R_test

##使ってない
def hoge(path='./data/movielens'):
    df = pd.read_csv(path + '/u.item', sep='\t', names=['movie_id', 'movie_title', 'release_data', 'video_release_date', 'IMDb_URL',\
                                                        'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',\
                                                        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', \
                                                        'Mystery', 'Romance', 'Sci-Fi', 'Thriller','War', 'Western'])
    shape = (df.max().ix['movie_id'], 1)
    R = np.zeros(shape)

    for i in df.index:
        row = df.ix[i]
        R[row['movie_id'] - 1] = row['movie_title']

    return R

#使ってない
def movie_name(path='./data/movielens'):
	lookup = defaultdict()
	for line in open(path + '/u.item'):
	    record = line.strip().split('|')
	    movie_id, movie_name = record[0], record[1]
	    lookup[movie_id] = movie_name
	return lookup

#
def loadMovieTitles(path='./data/movielens'):
    # 映画のタイトルを取得する処理です
    # movies = {}
    # for line in open(path + '/u.item'):
    # # u.item内のデータから映画IDと映画名のペアを作成していきます
    #     (id, title) = line.split('|')[0:2]
    #     movies[id] = title
    df = pd.read_csv(path + '/u.item', sep='\t', names=['movie_id', 'movie_title', 'release_data', 'video_release_date', 'IMDb_URL',\
                                                        'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',\
                                                        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', \
                                                        'Mystery', 'Romance', 'Sci-Fi', 'Thriller','War', 'Western'])
    n = df.max().ix['movie_id']
    titles = {}
    for movie_id in range(n):
        row = df.ix[movie_id]
        titles[movie_id] =  row['movie_title']

    return 0


#pandasをさらにスパース化(使ってない、)
def loadMovielensByPandasSparce(path='./data/movielens'):

    df = pd.read_csv(path + '/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    shape = (df.max().ix['user_id'] + 1, df.max().ix['item_id'] + 1)
    # R = np.zeros(shape)
    R = sparse.lil_matrix(shape)

    for i in df.index:
        row = df.ix[i]
        R[row['user_id'] - 1, row['item_id'] - 1] = row['rating']

    return R
