import os
import time
import sklearn
import xgboost as xgb
import pandas as pd
import numpy as np

train_data = pd.DataFrame()
train_label = pd.DataFrame()

test_data = pd.DataFrame()
test_label = pd.DataFrame()

g = os.walk('/data/luckytiger/shengliOilWell/train_data')
for _, _, file_list in g:
    for file in file_list:
        item = pd.read_excel('/data/luckytiger/shengliOilWell/train_data/{}'.format(file))
        t_input = item[['DEPTH', 'Por', 'Perm', 'AC', 'SP', 'COND', 'ML1', 'ML2']]
        # t_input = item[['Por', 'Perm', 'AC', 'SP', 'COND', 'ML1', 'ML2']]
        # t_input['ML2'] = t_input['ML2'] / t_input['ML2'].mean()
        t_input['Well'] = file[:2]

        t_output = item['level']
        t_output.loc[t_output != 60] = 61  # change to 2 class
        t_output = t_output - 60

        train_data = pd.concat([train_data, t_input], ignore_index=True)
        train_label = pd.concat([train_label, t_output], ignore_index=True)

        print('read in train file {}'.format(file))

t = os.walk('/data/luckytiger/shengliOilWell/test_data')
for _, _, file_list in t:
    for file in file_list:
        item = pd.read_excel('/data/luckytiger/shengliOilWell/test_data/{}'.format(file))
        t_input = item[['DEPTH', 'Por', 'Perm', 'AC', 'SP', 'COND', 'ML1', 'ML2']]
        # t_input = item[['Por', 'Perm', 'AC', 'SP', 'COND', 'ML1', 'ML2']]
        # t_input = t_input / t_input.mean()
        t_input['Well'] = file[:2]

        t_output = item['level']
        t_output.loc[t_output != 60] = 61  # change to 2 class
        t_output = t_output - 60

        test_data = pd.concat([test_data, t_input], ignore_index=True)
        test_label = pd.concat([test_label, t_output], ignore_index=True)

        print('read in test file {}'.format(file))

params = {
    'gpu_id': 0,
    'tree_method': 'gpu_hist',
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 2,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,  # doubt rationality
    'silent': 0,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

np_train_input = np.array(train_data)
np_train_output = np.array(train_label)
np_test_input = np.array(test_data)
np_test_output = np.array(test_label)

dtrain = xgb.DMatrix(np_train_input, label=np_train_output)

num_round = 100
bst = xgb.train(params, dtrain, num_round)

file_flag = str(time.time()).replace('.', "_")
bst.save_model('/data/luckytiger/shengliOilWell/train_result/model/xgboost/{}.model'.format(file_flag))

ans = bst.predict(np_test_input)

level_count = 0
no_level_count = 0
total_count = 0

