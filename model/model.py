import pandas as pd
import numpy as np
import os
from sklearn import model_selection
import xgboost as xgb
import matplotlib.pyplot as plt


ROOT = os.getenv('ROOT')

train_home_team_statistics_df = pd.read_csv(ROOT + '/train_home_team_statistics_df.csv', index_col=0)
train_away_team_statistics_df = pd.read_csv(ROOT + '/train_away_team_statistics_df.csv', index_col=0)

train_scores = pd.read_csv(ROOT + '/Y_train_1rknArQ.csv', index_col=0)

train_home = train_home_team_statistics_df.iloc[:,2:]
train_away = train_away_team_statistics_df.iloc[:,2:]

train_home.columns = 'HOME_' + train_home.columns
train_away.columns = 'AWAY_' + train_away.columns

