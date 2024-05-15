import wandb
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt

FOOTBALL_PROJECT = "Football-Analysis"


# start a new wandb run to track this script
run = wandb.init(
    # set the wandb project where this run will be logged
    project=FOOTBALL_PROJECT,
    job_type='train-model',
)

root = '/home/abelaid/Random/ML/Foot/train'

train_home_team_statistics_df = pd.read_csv(root + '/train_home_team_statistics_df.csv', index_col=0)
train_away_team_statistics_df = pd.read_csv(root + '/train_away_team_statistics_df.csv', index_col=0)

train_scores = pd.read_csv(root + '/Y_train_1rknArQ.csv', index_col=0)

train_home = train_home_team_statistics_df.iloc[:,2:]
train_away = train_away_team_statistics_df.iloc[:,2:]

train_home.columns = 'HOME_' + train_home.columns
train_away.columns = 'AWAY_' + train_away.columns

train_data =  pd.concat([train_home,train_away],join='inner',axis=1)
train_scores = train_scores.loc[train_data.index]

train_data = train_data.replace({np.inf:np.nan,-np.inf:np.nan})

# Benschmark model

train_new_y = train_scores['AWAY_WINS']

X_train, X_test, y_train, y_test = model_selection.train_test_split(train_data, train_new_y, train_size=0.8, random_state=42)
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_train, y_train, train_size=0.8, random_state=42)


d_train = xgb.DMatrix(X_train.replace({0:np.nan}), y_train)
d_valid = xgb.DMatrix(X_valid.replace({0:np.nan}), y_valid)

num_round = 10000
early_stopping_rounds=100

evallist = [(d_train, 'train'), (d_valid, 'eval')]

bst_params = {
    'booster': 'gbtree',
    'tree_method':'hist',
    'max_depth': 8, 
    'learning_rate': 0.025,
    'objective': 'multi:softprob',
    'num_class': 2,
    'eval_metric':['auc','mlogloss'],
    'nthread': 10,
    'tree_method':'gpu_hist',
    }

wandb.config.update(dict(bst_params))
run.config.update({'early_stopping_rounds':early_stopping_rounds})

from wandb.integration.xgboost import WandbCallback

bst = xgb.train(bst_params, d_train, num_round, evallist, early_stopping_rounds=early_stopping_rounds, callbacks=[WandbCallback(log_model=True)])

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()