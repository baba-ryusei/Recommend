import pandas as pd
import torch
from dataset_gen import *
from torch.utils.data import Dataset, DataLoader

anime_tv = pd.read_csv('anime_tv.csv')

anime_tv_action = anime_tv[anime_tv['Genres'].str.contains('Action', na=False)]
anime_tv_action.to_csv('anime_tv_action.csv', index=False)