import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
import ast

device = "cuda"

clustered_data = pd.read_csv("Clustered_data/clustered_synopsis_data.csv")
num_labels = clustered_data['MAL_ID'].shape[0]

# Embedding列の修正
def fix_embedding_format(embedding_str):
    # 数値の間にカンマを追加
    return re.sub(r'(?<=\d)\s+(?=-?\d)', ', ', embedding_str)

# カンマを追加して修正
clustered_data['Embedding'] = clustered_data['Embedding'].apply(fix_embedding_format)
print(clustered_data)
clustered_data['Embedding'] = clustered_data['Embedding'].apply(ast.literal_eval)  # 埋め込みをリスト形式に変換
embeddings = clustered_data['Embedding'].tolist()
labels = clustered_data['Cluster_Label'].values

train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
    embeddings,
    labels,
    test_size=0.2,
    random_state=42
)

#print(train_embeddings)
train_embeddings = torch.tensor(train_embeddings, dtype=torch.float32)
test_embeddings = torch.tensor(test_embeddings, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# データセットの作成
class AnimeEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'embeddings': self.embeddings[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

train_dataset = AnimeEmbeddingDataset(train_embeddings, train_labels)
test_dataset = AnimeEmbeddingDataset(test_embeddings, test_labels)

class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 入力 -> 隠れ層
        self.relu = nn.ReLU()  # 活性化関数
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)  # 隠れ層 -> 出力層
        self.softmax = nn.Softmax(dim=1)  # 出力層の確率化

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TransformerClassifier, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Linear(input_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        x = self.transformer_encoder(x)
        x = self.fc(x[:, 0, :])  # クラス分類用
        x = self.softmax(x)
        return x
    

# 入力サイズ（埋め込みの次元数）
input_size = len(train_embeddings[0])
hidden_size = 1024  # 隠れ層のユニット数
num_classes = len(torch.unique(train_labels))+1  # クラス数
learning_rate = 0.0001
batch_size = 64
num_epochs = 20

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# モデルの初期化
#model = DNNModel(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)
model = TransformerClassifier(input_size=input_size, num_classes=num_classes).to(device)

# 損失関数
criterion = nn.CrossEntropyLoss()

# 最適化アルゴリズム
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_classes = len(torch.unique(train_labels))  # ユニークなラベル数
#print(num_classes) # 142
print(f"Max train label: {train_labels.max()}, Max test label: {test_labels.max()}")
print(f"Number of classes (num_classes): {num_classes}")


# モデルのトレーニング
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        embeddings = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)
        # 順伝播
        outputs = model(embeddings)
        loss = criterion(outputs, labels)

        # 逆伝播とパラメータ更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# モデルの評価
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        embeddings = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)

        # 予測
        outputs = model(embeddings)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 精度と評価指標
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
