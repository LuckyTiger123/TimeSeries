import lightgbm as lgb
import pandas as pd
import numpy as np
import seaborn as sns

model_flag = '1608189601_6708531'
test_file = '31-215'

item = pd.read_excel('/data/luckytiger/shengliOilWell/test_data/{}.xlsx'.format(test_file))
bst = lgb.Booster(model_file='/data/luckytiger/shengliOilWell/train_result/model/xgboost/{}.model'.format(model_flag))

t_input = item[['DEPTH', 'Por', 'Perm', 'AC', 'SP', 'COND', 'ML1', 'ML2']]
t_input['Well'] = test_file[:2]
t_output = item['level']
t_output.loc[t_output != 60] = 61  # change to 2 class
t_output = t_output - 60

np_input = np.array(t_input)
np_label = np.array(t_output).flatten()

ans = bst.predict(np_input)
ans = np.argmax(ans, axis=1)

# compare in same graph
# item['level'] -= 60
# item['type'] = 'reality'
#
# item2 = item.copy()
# item2['level'] = ans
# item2['type'] = 'predict'
#
# item = pd.concat([item, item2], axis=0, ignore_index=True)
#
# sns.relplot(x='DEPTH', y='level', data=item, height=3, aspect=10, hue='type')

# compare minus
# item['level'] -= 60
# item['predict'] = ans
# item['level'] = item['level'] - item['predict']
# sns.relplot(x='DEPTH', y='level', data=item, height=3, aspect=10)

# statistic
# acc
level_count = 0
no_level_count = 0
total_count = 0
level_correct_count = 0
no_level_correct_count = 0
total_correct_count = 0

for i in range(len(np_label)):
    if np_label[i] == 1:
        level_count += 1
        if np_label[i] == ans[i]:
            level_correct_count += 1
            total_correct_count += 1
    elif np_label[i] == 0:
        no_level_count += 1
        if np_label[i] == ans[i]:
            no_level_correct_count += 1
            total_correct_count += 1
    total_count += 1

level_acc = level_correct_count / level_count
no_level_acc = no_level_correct_count / no_level_count
total_acc = total_correct_count / total_count

print('level acc:{} . no level acc:{} . total acc:{} .'.format(level_acc, no_level_acc, total_acc))

# total
TP = FN = FP = TN = 0
for i in range(len(np_label)):
    if np_label[i] == 1 and ans[i] == 1:
        TP += 1
    elif np_label[i] == 1 and ans[i] == 0:
        FN += 1
    elif np_label[i] == 0 and ans[i] == 1:
        FP += 1
    elif np_label[i] == 0 and ans[i] == 0:
        TN += 1

print('precision:{} . recall:{} .'.format(TP / (TP + FP), TP / (TP + FN)))
