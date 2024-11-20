import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# 1. BERT埋め込みの読み込み（既存の埋め込みがあると仮定）
embeddings = np.load("Recommend/anime_kai/pre_dataset/syn_embedding_TV_all-MiniLM-L12-v2.npy")
embeddings = torch.tensor(embeddings, dtype=torch.float32)

# 2. オートエンコーダの定義
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# 3. IDECのクラスタリング層を追加
class IDEC(nn.Module):
    def __init__(self, autoencoder, n_clusters, hidden_dim):
        super(IDEC, self).__init__()
        self.autoencoder = autoencoder
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, hidden_dim))
        nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x):
        z, decoded = self.autoencoder(x)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2))
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q, z, decoded

# ハイパーパラメータ
input_dim = embeddings.shape[1]
hidden_dim = 10
n_clusters = 10
alpha = 0.1  # クラスタリング損失の重要度を調整するパラメータ

# モデルの初期化
autoencoder = AutoEncoder(input_dim, hidden_dim)
idec = IDEC(autoencoder, n_clusters, hidden_dim)

# 最適化の設定
optimizer = optim.Adam(idec.parameters(), lr=0.001)
reconstruction_criterion = nn.MSELoss()

# 4. K-Meansで初期クラスタリング
with torch.no_grad():
    embeddings_encoded, _ = autoencoder(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    initial_cluster_centers = kmeans.fit(embeddings_encoded.numpy()).cluster_centers_
    idec.cluster_layer.data = torch.tensor(initial_cluster_centers, dtype=torch.float32)

# 5. トレーニングループ
for epoch in range(100):
    q, embeddings_encoded, decoded = idec(embeddings)
    
    # クラスタソフトアサインメント（ターゲット分布の計算）
    p = q ** 2 / q.sum(0)
    p = (p.t() / p.sum(1)).t()
    
    # 再構成損失とクラスタリング損失を組み合わせた損失
    reconstruction_loss = reconstruction_criterion(decoded, embeddings)
    kl_loss = torch.mean(torch.sum(p * torch.log(p / q), dim=1))
    loss = reconstruction_loss + alpha * kl_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}, Reconstruction Loss: {reconstruction_loss.item()}, KL Loss: {kl_loss.item()}")

# 6. クラスタリング結果と評価
cluster_labels = torch.argmax(q, dim=1).numpy()
silhouette_avg = silhouette_score(embeddings_encoded.detach().numpy(), cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")
