import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import time
import random
from Model.FCN import fcn_model


def cal_level_index(d):
    other_level = d[d['level'] != 60]
    return list(other_level.index)


def cal_loss_index(d):
    zero_level = d[d['level'] == 60]
    other_level = d[d['level'] != 60]
    other_size = int(other_level.shape[0])
    zero_size = int(zero_level.shape[0])
    result = list(other_level.index)
    if zero_size > other_size:
        result.extend(random.sample(list(zero_level.index), int(other_size)))
    else:
        result.extend(list(zero_level.index))
    # result.extend(list(zero_level.index))
    return result


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = list()
train_result = list()

# loss_index = list()

train_item = list()
level_index = list()
train_level_index = list()
zero_level_index = list()
train_zero_level_index = list()
well_name_list = list()

g = os.walk('/data/luckytiger/shengliOilWell/train_data')
for _, _, file_list in g:
    for file in file_list:
        item = pd.read_excel('/data/luckytiger/shengliOilWell/train_data/{}'.format(file))
        train_item.append(item)

        # loss_index_item = cal_loss_index(item)
        # loss_index.append(loss_index_item)

        train_level_index.append(cal_level_index(item))
        train_zero_level_index.append(list(set(item.index) - set(cal_level_index(item))))
        t_input = item[['DEPTH', 'Por', 'Perm', 'AC', 'SP', 'COND', 'ML1', 'ML2']]
        # t_input = item[['Por', 'Perm', 'AC', 'SP', 'COND', 'ML1', 'ML2']]
        t_input = t_input / t_input.mean()
        t_output = item['level']
        t_output.loc[t_output != 60] = 61  # change to 2 class
        train_input = np.array(t_input)
        train_output = np.array(t_output)
        train_data.append(train_input)
        train_result.append(train_output)
        print('read in train file {}'.format(file))

test_data = list()
test_result = list()

t = os.walk('/data/luckytiger/shengliOilWell/test_data')
for _, _, file_list in t:
    for file in file_list:
        item = pd.read_excel('/data/luckytiger/shengliOilWell/test_data/{}'.format(file))
        level_index.append(cal_level_index(item))
        zero_level_index.append(list(set(item.index) - set(cal_level_index(item))))
        t_input = item[['DEPTH', 'Por', 'Perm', 'AC', 'SP', 'COND', 'ML1', 'ML2']]
        # t_input = item[['Por', 'Perm', 'AC', 'SP', 'COND', 'ML1', 'ML2']]
        t_input = t_input / t_input.mean()
        t_output = item['level']
        t_output.loc[t_output != 60] = 61  # change to 2 class
        test_input = np.array(t_input)
        test_output = np.array(t_output)
        test_data.append(test_input)
        test_result.append(test_output)
        well_name_list.append(file.replace('.las', ''))
        print('read in test file {}'.format(file))

model = fcn_model.FCN8s().to(device)

# optim = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

optim = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

model.train()
for epoch in range(500):
    optim.zero_grad()
    loss_total = torch.tensor(0).to(device, dtype=torch.float)
    for i in range(len(train_data)):
        # optim.zero_grad()
        tensor_in = torch.tensor(train_data[i])
        tensor_in = tensor_in.unsqueeze(0).permute(0, 2, 1).to(device, dtype=torch.float)
        tensor_out = torch.tensor(train_result[i]).to(device, dtype=torch.long)
        tensor_out = tensor_out - 60
        out = model(tensor_in)
        out = out.squeeze(0).permute(1, 0)

        loss_index = cal_loss_index(train_item[i])
        # loss_index_item= loss_index[i]

        loss = F.nll_loss(out[loss_index], tensor_out[loss_index])
        loss_total += loss
        # loss.backward()
        # optim.step()
        print('train with the {} th item on the {} round.'.format(i, epoch, end='\r'))
    loss_total = loss_total / int(len(train_data))
    loss_total.backward()
    optim.step()
    print('after {} th training, the loss total is {}'.format(epoch, loss_total))
model.eval()

ac_data = pd.DataFrame(columns=['well_name', 'total_accurate', 'level_accurate', 'zero_accurate'])

# see train data set preference
# for j in range(len(train_data)):
#     tensor_in = torch.tensor(train_data[j])
#     tensor_in = tensor_in.unsqueeze(0).permute(0, 2, 1).to(device, dtype=torch.float)
#     tensor_out = torch.tensor(train_result[j]).to(device, dtype=torch.long)
#     tensor_out = tensor_out - 60
#     _, pred = model(tensor_in).max(dim=1)
#     correct = int(pred.eq(tensor_out).sum().item())
#     acc = correct / int(len(tensor_out))
#     level_correct = int(pred[0][train_level_index[j]].eq(tensor_out[train_level_index[j]]).sum().item())
#     level_total = int(len(train_level_index[j]))
#     l_acc = level_correct / level_total
#     zero_correct = int(pred[0][train_zero_level_index[j]].eq(tensor_out[train_zero_level_index[j]]).sum().item())
#     zero_level_total = int(len(train_zero_level_index[j]))
#     zl_acc = zero_correct / zero_level_total
#     ac_data.loc[ac_data.shape[0]] = {'well_name': j, 'total_accurate': acc, 'level_accurate': l_acc,
#                                      'zero_accurate': zl_acc}
#     print(ac_data.tail(1))

# test validate
for j in range(len(test_data)):
    tensor_in = torch.tensor(test_data[j])
    tensor_in = tensor_in.unsqueeze(0).permute(0, 2, 1).to(device, dtype=torch.float)
    tensor_out = torch.tensor(test_result[j]).to(device, dtype=torch.long)
    tensor_out = tensor_out - 60
    _, pred = model(tensor_in).max(dim=1)
    correct = int(pred.eq(tensor_out).sum().item())
    acc = correct / int(len(tensor_out))
    level_correct = int(pred[0][level_index[j]].eq(tensor_out[level_index[j]]).sum().item())
    level_total = int(len(level_index[j]))
    l_acc = level_correct / level_total
    zero_correct = int(pred[0][zero_level_index[j]].eq(tensor_out[zero_level_index[j]]).sum().item())
    zero_level_total = int(len(zero_level_index[j]))
    zl_acc = zero_correct / zero_level_total

    ac_data.loc[ac_data.shape[0]] = {'well_name': well_name_list[j], 'total_accurate': acc, 'level_accurate': l_acc,
                                     'zero_accurate': zl_acc}
    print(ac_data.tail(1))

file_flag = str(time.time())
torch.save(model, '/data/luckytiger/shengliOilWell/train_result/model/{}.pkl'.format(file_flag.replace('.', '_')))
ac_data.to_excel('/data/luckytiger/shengliOilWell/train_result/accurate/{}.xlsx'.format(file_flag.replace('.', '_')))
print(ac_data[['total_accurate', 'level_accurate', 'zero_accurate']].mean())
print('file flag: {}'.format(file_flag.replace('.', '_')))
print('Mission complete!')
