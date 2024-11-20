import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

data = pd.read_csv("Dataset/")

class TwoTowerDataset(Dataset):
    def __init__(self, data):
        # ユーザーとアイテムの特徴量、ラベルを設定
        self.user_ids = data['user_id'].values
        self.anime_ids = data['ANIME_ID'].values
        self.ratings = data['rating'].values.astype(np.float32)
        
        # アイテムの特徴量（例: 数値データやカテゴリデータ）
        self.item_features = data[['Score', 'Favorites', 'Popularity']].values.astype(np.float32)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        anime_id = self.anime_ids[idx]
        item_features = self.item_features[idx]
        rating = self.ratings[idx]
        return user_id, anime_id, item_features, rating


# ユーザーIDとアニメIDを連続値にエンコード
user_ids = data['user_id'].unique()
anime_ids = data['ANIME_ID'].unique()

user_id_map = {id: idx for idx, id in enumerate(user_ids)}
anime_id_map = {id: idx for idx, id in enumerate(anime_ids)}

data['user_id'] = data['user_id'].map(user_id_map)
data['ANIME_ID'] = data['ANIME_ID'].map(anime_id_map)

# データ分割
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# データセットの作成
train_dataset = TwoTowerDataset(train_data)
test_dataset = TwoTowerDataset(test_data)

# DataLoaderの作成
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
