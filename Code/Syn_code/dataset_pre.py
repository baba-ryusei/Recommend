import pandas as pd

syn = pd.read_csv('Recommend/anime_kai/pre_dataset/anime_with_synopsis.csv')
missing_data = syn[syn.isnull().any(axis=1)]

mal_ids = [34755, 34794, 38475, 40714, 42717, 44848, 45731, 46095]

# MAL_IDが特定の値に一致する行をフィルタリング
anime_names = syn[syn["MAL_ID"].isin(mal_ids)][["MAL_ID", "Name"]]

# 結果の表示
print(anime_names)


print(missing_data)