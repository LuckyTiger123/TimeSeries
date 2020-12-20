import os
import time
import lightgbm as lgb
import pandas as pd
import numpy as np

statistic = pd.DataFrame(
    columns=['proportion', 'level_acc', 'no_level_acc', 'total_acc', 'precision', 'recall', 'f1', 'model_flag'])

# define max sample proportion
proportion = 8

for i in range(1, proportion + 1):
    train_data = pd.DataFrame()
    train_label = pd.DataFrame()

    test_data = pd.DataFrame()
    test_label = pd.DataFrame()

    g = os.walk('/data/luckytiger/shengliOilWell/train_data')
    for _, _, file_list in g:
        for file in file_list:
            item = pd.read_excel('/data/luckytiger/shengliOilWell/train_data/{}'.format(file))

            t_level_item = item[item['level'] != 60]
            t_no_level_item = item[item['level'] == 60].sample(int(t_level_item.shape[0] * i), replace=True)
            item = pd.concat([t_level_item, t_no_level_item], ignore_index=True)

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

    np_train_input = np.array(train_data)
    np_train_output = np.array(train_label)
    np_train_output = np_train_output.flatten()
    np_test_input = np.array(test_data)
    np_test_output = np.array(test_label)
    np_test_output = np_test_output.flatten()

    params = {
        'objective': 'multiclass',
        'num_class': 2,
        'max_depth': 6,
        'num_threads': 4,
        # 'device_type': 'gpu',
        'seed': 0,
        'min_split_gain': 0.1,
        'bagging_fraction': 0.7,
        'feature_fraction': 0.7,
        'lambda_l2': 2
    }

    dtrain = lgb.Dataset(np_train_input, label=np_train_output)

    num_round = 50

    bst = lgb.train(params, dtrain, num_round)

    file_flag = str(time.time()).replace('.', "_")
    bst.save_model('/data/luckytiger/shengliOilWell/train_result/model/xgboost/{}.model'.format(file_flag))

    ans = bst.predict(np_test_input)
    ans = np.argmax(ans, axis=1)

    level_count = 0
    no_level_count = 0
    total_count = 0
    level_correct_count = 0
    no_level_correct_count = 0
    total_correct_count = 0

    for j in range(len(np_test_output)):
        if np_test_output[j] == 1:
            level_count += 1
            if np_test_output[j] == ans[j]:
                level_correct_count += 1
                total_correct_count += 1
        elif np_test_output[j] == 0:
            no_level_count += 1
            if np_test_output[j] == ans[j]:
                no_level_correct_count += 1
                total_correct_count += 1
        total_count += 1

    level_acc = level_correct_count / level_count
    no_level_acc = no_level_correct_count / no_level_count
    total_acc = total_correct_count / total_count

    print('Model name:{}.model'.format(file_flag))
    print('level acc:{} . no level acc:{} . total acc:{} .'.format(level_acc, no_level_acc, total_acc))

    # total
    TP = FN = FP = TN = 0
    for j in range(len(np_test_output)):
        if np_test_output[j] == 1 and ans[j] == 1:
            TP += 1
        elif np_test_output[j] == 1 and ans[j] == 0:
            FN += 1
        elif np_test_output[j] == 0 and ans[j] == 1:
            FP += 1
        elif np_test_output[j] == 0 and ans[j] == 0:
            TN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    print('precision:{} . recall:{} .'.format(precision, recall))
    print('f1:{} .'.format(f1))
    statistic.loc[statistic.shape[0]] = {'proportion': i,
                                         'level_acc': level_acc,
                                         'no_level_acc': no_level_acc, 'total_acc'
                                         : total_acc, 'precision': precision, 'recall': recall, 'f1': f1,
                                         'model_flag': file_flag}

statistic.to_excel('/data/luckytiger/shengliOilWell/f1_statistic.xlsx')
print('mission complete!')
