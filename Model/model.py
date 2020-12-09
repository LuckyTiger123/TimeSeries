import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import time
import random


def cal_level_index(d):
    other_level = d[d['level'] != 60]
    return list(other_level.index)


def cal_loss_index(d):
    zero_level = d[d['level'] == 60]
    other_level = d[d['level'] != 60]
    other_size = int(other_level.shape[0])
    zero_size = int(zero_level.shape[0])
    result = list(other_level.index)
    if zero_size > other_size / 3:
        result.extend(random.sample(list(zero_level.index), int(other_size / 3)))
    else:
        result.extend(list(zero_level.index))
    return result


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = list()
train_result = list()
loss_index = list()
level_index = list()
train_level_index = list()
well_name_list = list()

g = os.walk('/data/luckytiger/shengliOilWell/train_data')
for _, _, file_list in g:
    for file in file_list:
        item = pd.read_excel('/data/luckytiger/shengliOilWell/train_data/{}'.format(file))
        loss_index_item = cal_loss_index(item)
        loss_index.append(loss_index_item)
        train_level_index.append(list(cal_level_index(item)))
        t_input = item[['DEPTH', 'Por', 'Perm', 'AC', 'SP', 'COND', 'ML1', 'ML2']]
        # t_input = item[['Por', 'Perm', 'AC', 'SP', 'COND', 'ML1', 'ML2']]
        t_input = t_input / t_input.mean()
        t_output = item['level']
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
        t_input = item[['DEPTH', 'Por', 'Perm', 'AC', 'SP', 'COND', 'ML1', 'ML2']]
        # t_input = item[['Por', 'Perm', 'AC', 'SP', 'COND', 'ML1', 'ML2']]
        t_input = t_input / t_input.mean()
        t_output = item['level']
        test_input = np.array(t_input)
        test_output = np.array(t_output)
        test_data.append(test_input)
        test_result.append(test_output)
        well_name_list.append(file.replace('.las', ''))
        print('read in test file {}'.format(file))

# simple convolution operation
hidden_layer = 8


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv1d(8, hidden_layer, 5, padding=2)
        self.conv2 = torch.nn.Conv1d(hidden_layer, 9, 5, padding=2)

    def forward(self, data):
        x = self.conv1(data)
        x = F.relu(x)
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)


model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

model.train()
for epoch in range(20):
    for i in range(len(train_data)):
        optimizer.zero_grad()
        tensor_in = torch.tensor(train_data[i])
        tensor_in = tensor_in.unsqueeze(0).permute(0, 2, 1).to(device, dtype=torch.float)
        tensor_out = torch.tensor(train_result[i]).to(device, dtype=torch.long)
        tensor_out = tensor_out - 60
        out = model(tensor_in)
        out = out.squeeze(0).permute(1, 0)
        loss = F.nll_loss(out[loss_index[i]], tensor_out[loss_index[i]])
        loss.backward()
        optimizer.step()
        print('train with the {} th item on the {} round.'.format(i, epoch))
model.eval()

ac_data = pd.DataFrame(columns=['well_name', 'total_accurate', 'level_accurate'])

# see train data set preference
for j in range(len(train_data)):
    tensor_in = torch.tensor(train_data[j])
    tensor_in = tensor_in.unsqueeze(0).permute(0, 2, 1).to(device, dtype=torch.float)
    tensor_out = torch.tensor(train_result[j]).to(device, dtype=torch.long)
    tensor_out = tensor_out - 60
    _, pred = model(tensor_in).max(dim=1)
    correct = int(pred.eq(tensor_out).sum().item())
    acc = correct / int(len(tensor_out))
    level_correct = int(pred[0][train_level_index[j]].eq(tensor_out[train_level_index[j]]).sum().item())
    level_total = int(len(train_level_index[j]))
    l_acc = level_correct / level_total

    ac_data.loc[ac_data.shape[0]] = {'well_name': j, 'total_accurate': acc, 'level_accurate': l_acc}
    print(ac_data.tail(1))

# test validate
# for j in range(len(test_data)):
#     tensor_in = torch.tensor(test_data[j])
#     tensor_in = tensor_in.unsqueeze(0).permute(0, 2, 1).to(device, dtype=torch.float)
#     tensor_out = torch.tensor(test_result[j]).to(device, dtype=torch.long)
#     tensor_out = tensor_out - 60
#     _, pred = model(tensor_in).max(dim=1)
#     correct = int(pred.eq(tensor_out).sum().item())
#     acc = correct / int(len(tensor_out))
#     level_correct = int(pred[0][level_index[j]].eq(tensor_out[level_index[j]]).sum().item())
#     level_total = int(len(level_index[j]))
#     l_acc = level_correct / level_total
#
#     ac_data.loc[ac_data.shape[0]] = {'well_name': well_name_list[j], 'total_accurate': acc, 'level_accurate': l_acc}
#     print(ac_data.tail(1))

file_flag = str(time.time())
# torch.save(model, '/data/luckytiger/shengliOilWell/train_result/model/{}.pkl'.format(file_flag.replace('.', '_')))
# ac_data.to_excel('/data/luckytiger/shengliOilWell/train_result/accurate/{}.xlsx'.format(file_flag.replace('.', '_')))
print(ac_data[['total_accurate', 'level_accurate']].mean())
print('file flag: {}'.format(file_flag.replace('.', '_')))
print('Mission complete!')
