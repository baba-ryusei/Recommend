import pandas as pd

anime = pd.read_csv("Recommend/anime_kai/pre_dataset/anime.csv")
syn = pd.read_csv('Recommend/anime_kai/pre_dataset/anime_with_synopsis.csv')

#print(round(anime['Score'].describe(),2))
#print(round(syn['Score'].describe(),2))
print(anime['Name'].value_counts())
print(syn['Name'].value_counts())
print(anime.shape)
print(syn.shape)