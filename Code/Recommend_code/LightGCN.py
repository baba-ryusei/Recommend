import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


# LightGCNモデルの定義
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # ユーザーとアイテムの埋め込み
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 埋め込みの初期化
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(self, edge_index):
        # ノードの埋め込みを取得
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # ユーザーとアイテムの埋め込みを結合
        embeddings = torch.cat([user_emb, item_emb], dim=0)
        
        # メッセージパッシング (エッジを通じて埋め込みを更新)
        for _ in range(2):  # 2層のLightGCN
            embeddings = self.propagate(edge_index, x=embeddings)
        
        return embeddings
    
    def propagate(self, edge_index, x):
        # メッセージパッシング (平均埋め込み)
        row, col = edge_index
        x_agg = torch.zeros_like(x)
        x_agg[row] += x[col]
        x_agg[col] += x[row]
        return x_agg / 2

def train(device, data):
    
    num_users = data['user_id'].nunique()
    num_animes = data['MAL_ID'].nunique()
    # モデルの初期化
    embedding_dim = 64
    model = LightGCN(num_users, num_animes, embedding_dim)

    # モデルの訓練 (例: Matrix FactorizationベースのLoss)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(10):
        optimizer.zero_grad()
        embeddings = model(graph_data.edge_index)
        
        # ユーザーとアイテム埋め込みを分割
        user_emb = embeddings[:num_users]
        item_emb = embeddings[num_users:]
        
        # 例: ユーザー0とアイテム3の内積
        user_id = torch.tensor(0)
        anime_id = torch.tensor(3)
        score = torch.dot(user_emb[user_id], item_emb[anime_id])
        
        # 損失の計算 (例: MSE Loss)
        target_score = torch.tensor(1.0)  # ラベル
        loss = F.mse_loss(score, target_score)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
