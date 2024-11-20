from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Genreとsypnopsisを考慮した場合が性能が良い
loaded_embeddings = np.load("Embedding_data/syn_embedding_TV_Genre_all-MiniLM-L12-v2.npy")

# 1. データの読み込み
anime = pd.read_csv('Pre_data/new_anime_dataset_TV_Genre.csv').dropna()
"""
# 2. Genre_Embedding と Sypnopsis_Embedding を取り出して配列に変換
genre_embeddings = np.array(data['Genre_Embedding'].apply(eval).tolist())  # evalでリスト型に変換
sypnopsis_embeddings = np.array(data['Sypnopsis_Embedding'].apply(eval).tolist())

# 3. 埋め込みベクトルを結合
combined_embeddings = np.hstack((genre_embeddings, sypnopsis_embeddings))
"""

normalized_embeddings = normalize(loaded_embeddings, norm='l2', axis=1)

print(normalized_embeddings)

pca = PCA(n_components=50)
pca_result = pca.fit_transform(normalized_embeddings)
umap = TSNE(n_components=2, random_state=42)
umap_result = umap.fit_transform(pca_result)

"""
# シルエットスコアを格納するリスト
silhouette_scores = []
cluster_range = range(2, 200)  # クラスタ数を2～10まで試す

for n_clusters in cluster_range:
    # KMeansクラスタリングの適用
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(umap_result)

    # シルエットスコアの計算
    silhouette_avg = silhouette_score(loaded_embeddings, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    #print(f"クラスタ数: {n_clusters}, シルエットスコア: {silhouette_avg:.4f}")

# 最適なクラスタ数の特定
optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
print(f"最適なクラスタ数: {optimal_clusters}")

# シルエットスコアのプロット
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Various Numbers of Clusters")
plt.show()
"""

num_clusters = 143
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(umap_result)

#gmm = GaussianMixture(n_components=num_clusters, covariance_type='full', random_state=42)
#cluster_labels = gmm.fit_predict(umap_result)

#dbscan = DBSCAN(eps=5, min_samples=5)
#cluster_labels = dbscan.fit_predict(umap_result)

silhouette_avg = silhouette_score(umap_result, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")


# 各サンプルのシルエットスコア
sample_silhouette_values = silhouette_samples(loaded_embeddings, cluster_labels)

# シルエット図の描画
fig, ax = plt.subplots(figsize=(10, 8))
y_lower = 10
for i in range(num_clusters):
    # クラスタiのシルエットスコアを抽出
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()

    # クラスタごとのシルエット図の描画
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = plt.cm.nipy_spectral(float(i) / num_clusters)
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

    # クラスタラベルの表示
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10  # 次のクラスタに移動

ax.set_title("Silhouette plot for the clusters")
ax.set_xlabel("Silhouette coefficient")
ax.set_ylabel("Cluster label")
ax.axvline(x=silhouette_avg, color="red", linestyle="--")
ax.set_yticks([])  # y軸のラベルを非表示
ax.set_xticks(np.arange(-1, 1.1, 0.2))

plt.show()


clustered_data = pd.DataFrame({
    'MAL_ID': anime['MAL_ID'],  
    'Name': anime['Name'],    
    'Genre': anime['Genres'],
    'sypnopsis': anime['sypnopsis'],
    'Cluster_Label': cluster_labels,
    'Embedding': list(loaded_embeddings)  
})
unique_labels = sorted(clustered_data['Cluster_Label'].unique())
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
clustered_data['Cluster_Label'] = clustered_data['Cluster_Label'].map(label_mapping)

print(clustered_data.head())
print(clustered_data.shape)

clustered_data.to_csv("Clustered_data/clustered_synopsis_data.csv", index=False)

plt.figure(figsize=(10, 8))
for i in range(num_clusters):
    cluster_points = umap_result[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
plt.legend()
plt.title("t-SNE Visualization of Clusters")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()