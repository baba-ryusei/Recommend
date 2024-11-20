import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

class ClusteringPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.loaded_embeddings = None
        self.normalized_embeddings = None
        self.umap_result = None
        self.cluster_labels = None
        self.silhouette_scores = []

    def load_data(self):
        """データをロードしてL2ノルム正規化"""
        self.loaded_embeddings = np.load(self.data_path)
        self.normalized_embeddings = normalize(self.loaded_embeddings, norm='l2', axis=1)

    def dimensionality_reduction(self, pca_components=50):
        """PCAとt-SNEで次元削減"""
        pca = PCA(n_components=pca_components)
        pca_result = pca.fit_transform(self.normalized_embeddings)
        tsne = TSNE(n_components=2, random_state=42)
        self.umap_result = tsne.fit_transform(pca_result)

    def find_optimal_clusters(self, cluster_range=range(2, 20)):
        """最適なクラスタ数を探索"""
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.umap_result)
            silhouette_avg = silhouette_score(self.umap_result, cluster_labels)
            self.silhouette_scores.append(silhouette_avg)
        
        optimal_clusters = cluster_range[np.argmax(self.silhouette_scores)]
        return optimal_clusters

    def plot_silhouette_scores(self, cluster_range):
        """シルエットスコアをプロット"""
        plt.figure(figsize=(8, 6))
        plt.plot(cluster_range, self.silhouette_scores, marker='o')
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score for Various Numbers of Clusters")
        plt.show()

    def perform_clustering(self, n_clusters):
        """KMeansクラスタリングを実行"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = kmeans.fit_predict(self.umap_result)
        return silhouette_score(self.umap_result, self.cluster_labels)

    def plot_silhouette_diagram(self, n_clusters):
        """シルエット図を描画"""
        sample_silhouette_values = silhouette_samples(self.umap_result, self.cluster_labels)
        fig, ax = plt.subplots(figsize=(10, 8))
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[self.cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = plt.cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        ax.set_title("Silhouette plot for the clusters")
        ax.set_xlabel("Silhouette coefficient")
        ax.axvline(x=np.mean(sample_silhouette_values), color="red", linestyle="--")
        ax.set_yticks([])
        plt.show()

    def save_clustered_data(self, output_path):
        """クラスタリング結果を保存"""
        clustered_data = pd.DataFrame({
            'Cluster_Label': self.cluster_labels,
            'Synopsis_Embedding': list(self.loaded_embeddings)
        })
        clustered_data.to_csv(output_path, index=False)

    def plot_clusters(self, n_clusters):
        """クラスタリング結果を可視化"""
        plt.figure(figsize=(10, 8))
        for i in range(n_clusters):
            cluster_points = self.umap_result[self.cluster_labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
        plt.legend()
        plt.title("t-SNE Visualization of Clusters")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.show()

if __name__ == '__main__':
    # クラスの初期化
    clustering = ClusteringPipeline(data_path="pre_dataset/syn_embedding_TV_Genre_all-MiniLM-L12-v2.npy")

    # データのロードと正規化
    clustering.load_data()

    # 次元削減
    clustering.dimensionality_reduction(pca_components=50)

    # 最適なクラスタ数を探す
    cluster_range = range(2, 20)
    optimal_clusters = clustering.find_optimal_clusters(cluster_range=cluster_range)
    print(f"最適なクラスタ数: {optimal_clusters}")

    # シルエットスコアをプロット
    clustering.plot_silhouette_scores(cluster_range=cluster_range)

    # 最適なクラスタ数でクラスタリング
    silhouette_avg = clustering.perform_clustering(n_clusters=optimal_clusters)
    print(f"Silhouette Score for {optimal_clusters} clusters: {silhouette_avg}")

    # シルエット図を描画
    clustering.plot_silhouette_diagram(n_clusters=optimal_clusters)

    # クラスタリング結果を保存
    clustering.save_clustered_data(output_path="pre_dataset/clustered_synopsis_data.csv")

    # クラスタリング結果を可視化
    clustering.plot_clusters(n_clusters=optimal_clusters)
