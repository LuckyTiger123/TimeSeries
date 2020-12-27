import pandas as pd
import numpy as np
import torch
from dtw import dtw

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

# statistic
statistic_form = pd.DataFrame(columns=['zoom_rate', 'acc_rate', 'best_params'])

# loc set
node_list = list()

for i in range(0, 6):
    attr_set = list()
    node_list.append(attr_set)

for i in (range(1, 9)):
    f = open('/data/luckytiger/shengliOilWell/TSC/train_data/level{}/loc_set'.format(i), 'r')
    file_name = f.readline()
    while file_name:
        file_name = file_name.replace('\n', '')
        item = pd.read_excel('/data/luckytiger/shengliOilWell/TSC/train_data/level{}/{}'.format(i, file_name))
        item = item.drop(['Perm', 'Por'], axis=1)
        np_item = np.array(item)
        np_item = np_item.T
        item_list = [i - 1, int(file_name[:2])]
        for j in range(1, 7):
            np_row = np_item[j]
            if np.sum(np_row == -9999) != 0:
                continue
            attr_item_list = item_list.copy()
            attr_item_list.append(np_item[j])
            node_list[j - 1].append(attr_item_list)
        print('loc file {}\'s level {} has been read in'.format(file_name, i))
        file_name = f.readline()
        # break
    f.close()

# train set
train_list = list()

for i in (range(1, 9)):
    f = open('/data/luckytiger/shengliOilWell/TSC/train_data/level{}/adjust_set'.format(i), 'r')
    file_name = f.readline()
    while file_name:
        file_name = file_name.replace('\n', '')
        item = pd.read_excel('/data/luckytiger/shengliOilWell/TSC/train_data/level{}/{}'.format(i, file_name))
        item = item.drop(['Perm', 'Por'], axis=1)
        np_item = np.array(item)
        np_item = np_item.T
        item_list = [i - 1, int(file_name[:2]), np_item[1:]]
        train_list.append(item_list)
        print('train file {}\'s level {} has been read in'.format(file_name, i))
        file_name = f.readline()
        # break
    f.close()

sum = int(len(train_list))

# train setting
manhattan_distance = lambda x, y: np.abs(x - y)

K = 20

cross_entropy = torch.nn.CrossEntropyLoss()

# generate the dist set
dist_list_label_pair = list()

for item in train_list:
    item_attr = item[2]
    item_dist = list()
    for i in range(6):
        vect_dist = list()
        if np.sum(item_attr[i] == -9999) != 0:
            item_dist.append(vect_dist)
            continue
        for loc_item in node_list[i]:
            level_gap = abs(item[1] - loc_item[1])
            dist, _, _, _ = dtw(item_attr[i], loc_item[2], dist=manhattan_distance)
            vect_dist.append([loc_item[0], level_gap, dist])
        item_dist.append(vect_dist)
    dist_list_label_pair.append(item_dist)
    print('level {} item has been caled origin dist.'.format(item[0]))

for i in range(11):
    # parameters
    zoom_rate = torch.tensor(0.1 * i, device=device)
    attr_weight = torch.full([6, 1], 1 / 6, device=device, requires_grad=True)

    # label matrix
    label_agg_matrix = list()

    # update label matrix
    for j in range(len(train_list)):
        label_matrix = list()
        for k in range(6):
            current_list = dist_list_label_pair[j][k].copy()
            if len(current_list) == 0:
                label_matrix.append([0, 0, 0, 0, 0, 0, 0, 0])
                continue
            for item in current_list:
                item[2] = item[2] * (1 + item[1] / 22 * float(zoom_rate))
            sort_list = sorted(current_list, key=lambda s: s[2])[:K]
            count = [0, 0, 0, 0, 0, 0, 0, 0]
            for item in sort_list:
                count[item[0]] += 1
            label_matrix.append(count)
        np_label = np.array(label_matrix)
        np_label = np_label.T
        label_agg_matrix.append(np_label)

    # optimizer
    optim = torch.optim.Adam([attr_weight], lr=0.005)

    # adjust params
    for epoch in range(100):
        total_loss = torch.zeros(1, device=device, dtype=torch.float)
        optim.zero_grad()
        for k in range(sum):
            item_agg = label_agg_matrix[k]
            item_label = train_list[k][0]
            tensor_label = torch.tensor(item_agg, dtype=torch.float, requires_grad=True, device=device)
            result = torch.mm(tensor_label, attr_weight)
            result = result.T
            loss = cross_entropy(result, torch.tensor([item_label]).to(device=device))
            total_loss += loss
        total_loss = total_loss / sum
        total_loss.backward()
        optim.step()
        print('for {} epoch, the loss is {}, the tensor is:'.format(epoch, total_loss))
        print(attr_weight.T)

    # analyze
    total_num = 0
    acc_num = 0

    for k in range(sum):
        item_agg = label_agg_matrix[k]
        item_label = train_list[k][0]
        tensor_label = torch.tensor(item_agg, dtype=torch.float, requires_grad=True, device=device)
        result = torch.mm(tensor_label, attr_weight)
        label_pre = int(torch.max(result, 0)[1])
        if label_pre == item_label:
            acc_num += 1
        total_num += 1

    print(
        'for the {} epoch, the total test num is {}, acc num is {}, the acc rate is {}.'.format(i, total_num,
                                                                                                acc_num,
                                                                                                acc_num / total_num))
    statistic_form.loc[statistic_form.shape[0]] = {'zoom_rate': float(zoom_rate), 'acc_rate': acc_num / total_num,
                                                   'best_params': attr_weight.cpu().detach().numpy().tolist()}

statistic_form.to_excel('/data/luckytiger/shengliOilWell/TSC/expand_rate_statistic.xlsx')
print('mission complete!')
