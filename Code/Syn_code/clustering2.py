import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. データの読み込み
data = pd.read_csv('pre_dataset/anime_with_embeddings.csv')
# 要素間にカンマを追加
data['Genre_Embedding'] = data['Genre_Embedding'].str.replace(r'(?<=\d)\s(?=\d)', ', ')


# 2. Genre_Embedding と Sypnopsis_Embedding を取り出して配列に変換
genre_embeddings = np.array(data['Genre_Embedding'].apply(eval).tolist())  # evalでリスト型に変換
sypnopsis_embeddings = np.array(data['Sypnopsis_Embedding'].apply(eval).tolist())

# 3. 埋め込みベクトルを結合
combined_embeddings = np.hstack((genre_embeddings, sypnopsis_embeddings))

# 4. L2ノルム正規化
normalized_embeddings = normalize(combined_embeddings, norm='l2', axis=1)

# 5. クラスタリングの実行
num_clusters = 193  # クラスタ数を指定
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(normalized_embeddings)

# クラスタリングの評価（オプション）
silhouette_avg = silhouette_score(normalized_embeddings, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

# 6. クラスタリング結果をデータフレームに追加
data['Cluster_Label'] = cluster_labels

# 7. 新しいCSVファイルとして保存
output_path = 'pre_dataset/anime_with_cluster_ids.csv'
data.to_csv(output_path, index=False)

print(f"クラスタリング結果を追加したCSVファイルが作成されました: {output_path}")
