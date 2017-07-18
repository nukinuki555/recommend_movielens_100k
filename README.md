# recommend_movielens_100k
レコメンドシステムを強調フィルタリングとオートエンコーダで実装して比較してみました

## Description

movielensの評点を予想して、その精度をRMSE(二乗平均誤差平方根)で評価
予想評点からレコメンドの生成を行います
アルゴリズムは
- 強調フィルタリング
- オートエンコーダ

##Explains for items

root/
- trainer_auto_encoder.py : オートエンコーダのmainコード
- auto_encoder_layer_mnist.py: オートエンコーダのクラス
- auto_encoder_deeper.py: 深くしたオートエンコーダのクラス
- load_movie_lens.py: movielensのデータの読み込み、処理
- recommendatios.py: 強調フィルタリングのmainコード
- make_recommendations.py: 予想評点からレコメンド生成

output/
- CM_output_test_movielens.csv: 協調フィルタリングの評点推測結果(行: user_id, 列: item_id)
- CM_origin_test_movielens.csv: 協調フィルタリングに使ったデータ(行: user_id, 列: item_id)
- similarity_test__movielens.csv: 協調フィルタリングで作成したアイテムベースの類似度(行: item_id, 列: item_id)
- [BAE-output-2000k-L2500.csv: オートエンコーダで生成した予想評点、隠れ層2000で学習は2500回(行: user_id, 列: item_id)
- AE_output_3000k_L5500.csv: オートエンコーダで生成した予想評点、隠れ層3000で学習は5500回(行: user_id, 列: item_id)

グラフ/
- AE_2000k_L10000_output.png: オートエンコーダのRMSE評価のエポックごとの移動(隠れ層2000、学習10000回)
- AE_gragh_300k_L5500.png: オートエンコーダのRMSE評価のエポックごとの移動(隠れ層3000(名前ミスってました...)、学習5500回)

Recommendations/
Recommendations_CM_output_test_movielens.csv: 協調フィルタリングで予想評点から高かった順に10項目をレコメンド((行: user_id, 列: 映画のタイトル)
Recommendations_[BAE-output-2000k-L2500.csv:　オートエンコーダの隠れ層2000学習2500で予想評点から高かった順に10項目をレコメンド((行: user_id, 列: 映画のタイトル))
Recommendations_AE_output_2000k_L10000.csv: オートエンコーダの隠れ層2000学習10000で予想評点から高かった順に10項目をレコメンド((行: user_id, 列: 映画のタイトル))
Recommendations_AE_output_3000k_L5500.csv: オートエンコーダの隠れ層3000学習5500で予想評点から高かった順に10項目をレコメンド((行: user_id, 列: 映画のタイトル))
