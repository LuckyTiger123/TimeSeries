import pandas as pd
import numpy as np
import torch
from dtw import dtw
from tqdm import tqdm

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# loc set
node_list = list()

for i in range(0, 6):
    attr_set = list()
    node_list.append(attr_set)

for i in tqdm(range(1, 9)):
    f = open('/data/luckytiger/shengliOilWell/TSC/train_data/level{}/loc_set'.format(i), 'r')
    file_name = f.readline()
    while file_name:
        file_name = file_name.replace('\n', '')
        item = pd.read_excel('/data/luckytiger/shengliOilWell/TSC/train_data/level{}/{}'.format(i, file_name))
        item = item.drop(['Perm', 'Por'], axis=1)
        np_item = np.array(item)
        np_item = np_item.T
        item_list = [i, int(file_name[:2])]
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

for i in range(0, 8):
    attr_set = list()
    train_list.append(attr_set)

for i in tqdm(range(1, 9)):
    f = open('/data/luckytiger/shengliOilWell/TSC/train_data/level{}/adjust_set'.format(i), 'r')
    file_name = f.readline()
    while file_name:
        file_name = file_name.replace('\n', '')
        item = pd.read_excel('/data/luckytiger/shengliOilWell/TSC/train_data/level{}/{}'.format(i, file_name))
        item = item.drop(['Perm', 'Por'], axis=1)
        np_item = np.array(item)
        np_item = np_item.T
        item_list = [int(file_name[:2]), np_item[1:]]
        train_list[i - 1].append(item_list)
        print('train file {}\'s level {} has been read in'.format(file_name, i))
        file_name = f.readline()
        # break
    f.close()

# parameters
zoom_rate = torch.tensor(0.35, device=device, requires_grad=True)
attr_weight = torch.full([6, 1], 1 / 6, device=device, requires_grad=True)

# optim = torch.optim.Adam([zoom_rate, attr_weight], lr=0.005)
optim = torch.optim.Adam([attr_weight], lr=0.01)

# train
manhattan_distance = lambda x, y: np.abs(x - y)

sum = 0
for i in range(8):
    sum += int(len(train_list[i]))

K = 20

cross_entropy = torch.nn.CrossEntropyLoss()

for epoch in range(0):
    total_loss = torch.zeros(1, device=device, dtype=torch.float)

    optim.zero_grad()

    for j in range(8):
        level_label = j
        for item in train_list[j]:
            well_region = item[0]
            level_data = item[1]

            label_list = list()

            for i in range(0, 6):
                attr_KNN = list()
                if np.sum(level_data[i] == -9999) != 0:
                    label_list.append([0, 0, 0, 0, 0, 0, 0, 0])
                    continue

                for loc_item in node_list[i]:
                    # d, cost_matrix, acc_cost_matrix, path = dtw(level_data[i], loc_item[2], dist=manhattan_distance)
                    d, _, _, _ = dtw(level_data[i], loc_item[2], dist=manhattan_distance)
                    d = d * (1 + abs(well_region - loc_item[1]) / 22 * zoom_rate)
                    if len(attr_KNN) == K:
                        max = attr_KNN[0][1]
                        index = 0
                        for k in range(1, K):
                            if attr_KNN[k][1] > max:
                                max = attr_KNN[k][1]
                                index = k
                        if d >= max:
                            continue
                        else:
                            attr_KNN[index][1] = d
                            attr_KNN[index][0] = loc_item[0]
                    else:
                        attr_KNN.append([loc_item[0], d])
                count = [0, 0, 0, 0, 0, 0, 0, 0]
                # count = torch.zeros(8)
                for item in attr_KNN:
                    count[item[0] - 1] += 1
                # label_list = torch.cat([label_list, count], dim=0)
                label_list.append(count)
            np_label = np.array(label_list)
            np_label = np_label.T
            tensor_label = torch.tensor(np_label, dtype=torch.float, requires_grad=True, device=device)
            result = torch.mm(tensor_label, attr_weight)
            result = result.T
            loss = cross_entropy(result, torch.tensor([level_label]).to(device=device))
            total_loss += loss
            print('level {} item loss is {}'.format(level_label, loss))

    total_loss = total_loss / sum
    total_loss.backward()
    optim.step()
    print('for {} epoch, the loss is {}, the tensor is:'.format(epoch, total_loss))
    print(attr_weight.T)

# analyze
total_num = 0
acc_num = 0

for j in range(8):
    level_label = j
    for item in train_list[j]:
        well_region = item[0]
        level_data = item[1]

        label_list = list()

        for i in range(0, 6):
            if np.sum(level_data[i] == -9999) != 0:
                label_list.append([0, 0, 0, 0, 0, 0, 0, 0])
                continue
            attr_KNN = list()
            for loc_item in node_list[i]:
                # d, cost_matrix, acc_cost_matrix, path = dtw(level_data[i], loc_item[2], dist=manhattan_distance)
                d, _, _, _ = dtw(level_data[i], loc_item[2], dist=manhattan_distance)
                d = d * (1 + abs(well_region - loc_item[1]) / 22 * zoom_rate)
                if len(attr_KNN) == K:
                    max = attr_KNN[0][1]
                    index = 0
                    for k in range(1, K):
                        if attr_KNN[k][1] > max:
                            max = attr_KNN[k][1]
                            index = k
                    if d >= max:
                        continue
                    else:
                        attr_KNN[index][1] = d
                        attr_KNN[index][0] = loc_item[0]
                else:
                    attr_KNN.append([loc_item[0], d])
            count = [0, 0, 0, 0, 0, 0, 0, 0]
            # count = torch.zeros(8)
            for item in attr_KNN:
                count[item[0] - 1] += 1
            # label_list = torch.cat([label_list, count], dim=0)
            label_list.append(count)
        np_label = np.array(label_list)
        np_label = np_label.T
        tensor_label = torch.tensor(np_label, dtype=torch.float, requires_grad=True, device=device)
        result = torch.mm(tensor_label, attr_weight)
        label_pre = int(torch.max(result, 0)[1])
        if label_pre == j:
            acc_num += 1
        total_num += 1
        print('test label {} with result {}'.format(j, label_pre))

print('the total test num is {}, acc num is {}, the acc rate is {}.'.format(total_num, acc_num, acc_num / total_num))
print('mission complete!')
