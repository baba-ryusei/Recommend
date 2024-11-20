from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np

"""
anime_syn = pd.read_csv('Recommend/anime_kai/pre_dataset/anime_with_synopsis.csv')
syn_id = anime_syn[['MAL_ID', 'Name', 'sypnopsis']].dropna()

# 新しいIDを作成
unique_ids = syn_id['MAL_ID'].unique()
id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
syn_id['new_id'] = syn_id['MAL_ID'].map(id_map)

# 新しい連続IDとアニメ名の対応を保存
id_to_name = dict(zip(syn_id['new_id'], syn_id['Name']))
# アニメの名前とあらすじを対応
name_to_sypnopsis = dict(zip(syn_id['Name'], syn_id['sypnopsis']))
"""

#new_anime = pd.read_csv('pre_dataset/new_anime_dataset_TV_One.csv')
new_anime = pd.read_csv("tv_merged.csv")
new_anime = new_anime.dropna()
new_anime = new_anime[new_anime['Genres'].str.contains('Action', na=False)]
#new_anime = new_anime[new_anime['Episodes'] != 'Unknown']

# BERT
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#model = AutoModel.from_pretrained("bert-base-uncased")

# Sentence-BERT
model = SentenceTransformer('all-MiniLM-L12-v2')

# あらすじを埋め込みベクトルに変換する関数 (BERTモデルの時)

"""
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    # 最後の隠れ層のベクトルの平均を取る
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
    return embedding
"""

batch_size = 64
all_embeddings = []
genre_embeddings = []
syn_embeddings = []

# バッチごとに埋め込みを生成
for i in range(0, len(new_anime), batch_size):
    batch_sypnopsis = new_anime['sypnopsis'][i:i+batch_size].tolist()
    batch_genre = new_anime['Genres'][i:i+batch_size].tolist()
    #batch_embeddings = get_embedding(batch_texts)
    batch_syn_embeddings = model.encode(batch_sypnopsis, batch_size=batch_size, show_progress_bar=True) #Sentence-BERTモデルを用いるとき
    batch_genre_embeddings = model.encode(batch_genre, batch_size=batch_size, show_progress_bar=True)
    batch_combined_embeddings = [np.concatenate((synopsis, genres)) for synopsis, genres in zip(batch_syn_embeddings, batch_genre_embeddings)]
    all_embeddings.extend(batch_combined_embeddings)
    genre_embeddings.extend(batch_genre_embeddings)
    syn_embeddings.extend(batch_syn_embeddings)
    
# 埋め込みをNumPy配列に変換
all_embeddings = np.array(all_embeddings)
genre_embeddings = np.array(genre_embeddings)
syn_embeddings = np.array(syn_embeddings)    

# 埋め込みを保存
new_anime['Genre_Embedding'] = list(genre_embeddings)
new_anime['Sypnopsis_Embedding'] = list(syn_embeddings)
    
"""    
# 数値データ（EpisodesとSource）を追加
episodes = new_anime['Episodes'].to_numpy().reshape(-1, 1)
source = new_anime.iloc[:, 5:].to_numpy()  # ワンホットエンコーディングされたSource列

# 全てを結合
final_dataset = np.hstack((all_embeddings, episodes, source))
"""

# データセットを保存
output_data = new_anime[['MAL_ID','Name','Score','Completed','Favorites','Genre_Embedding', 'Sypnopsis_Embedding']]
output_data.to_csv('anime_tv_action_embeddings.csv', index=False)
#np.save("pre_dataset/final_anime_dataset.npy", final_dataset)
#print("Final dataset shape:", final_dataset.shape)
