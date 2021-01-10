import pandas as pd
import numpy as np
import torch.nn.functional as F
import os
import torch

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

# read in train set
level_label_data = list()

for i in range(8):
    g = os.walk('/data/luckytiger/shengliOilWell/TSC/train_data/level{}'.format(i + 1))
    for _, _, file_list in g:
        for file in file_list:
            item = pd.read_excel('/data/luckytiger/shengliOilWell/TSC/train_data/level{}/{}'.format((i + 1), file))
            item.drop(['Por', 'Perm'], axis=1, inplace=True)

            if -9999 in item['AC'].values or -9999 in item['SP'].values or -9999 in item['COND'].values or -9999 in \
                    item['ML1'].values or -9999 in item['ML2'].values:
                print('a file is been thrown out.')
                continue

            # some handle option
            item['DEPTH'] = item['DEPTH'] / 1500
            item['AC'] = item['AC'] / 500
            item['SP'] = item['SP'] / 100
            item['COND'] = item['COND'] / 400
            item['ML1'] = item['ML1'] / 5
            item['ML2'] = item['ML2'] / 5

            # somehow to normalize the data.
            # item = (item - item.min()) / (item.max() - item.min())
            # item = item / item.mean()

            np_item = np.array(item)
            np_item = np.delete(np_item, 0, axis=1)
            level_label_data.append([np_item.tolist(), i, int(file[:2])])
            print('File {} of level {} has been read in.'.format(file, i))
            # break

sorted_data = sorted(level_label_data, key=lambda x: len(x[0]), reverse=True)
max_len = max(len(item[0]) for item in sorted_data)

len_list = list()
label_list = list()
region_list = list()

for item in sorted_data:
    len_list.append(int(len(item[0])))
    label_list.append(item[1])
    # label_list.append(int(item[1] / 4))

    region_list.append([item[2]])
    for i in range(max_len - len(item[0])):
        item[0].append([0, 0, 0, 0, 0, 0])

batch_seq = np.array([item[0] for item in sorted_data])

embed_input_x_packd = torch.nn.utils.rnn.pack_padded_sequence(torch.Tensor(batch_seq), torch.Tensor(len_list),
                                                              batch_first=True).to(device)

region_tensor = torch.unsqueeze(torch.Tensor(region_list), 0).to(device)

# read in test set
test_level_label_data = list()

for i in range(8):
    g = os.walk('/data/luckytiger/shengliOilWell/TSC/test_data/level{}'.format(i + 1))
    for _, _, file_list in g:
        for file in file_list:
            item = pd.read_excel('/data/luckytiger/shengliOilWell/TSC/test_data/level{}/{}'.format((i + 1), file))
            item.drop(['Por', 'Perm'], axis=1, inplace=True)

            if -9999 in item['AC'].values or -9999 in item['SP'].values or -9999 in item['COND'].values or -9999 in \
                    item['ML1'].values or -9999 in item['ML2'].values:
                print('a test file is been thrown out.')
                continue

            # some handle option
            item['DEPTH'] = item['DEPTH'] / 1500
            item['AC'] = item['AC'] / 500
            item['SP'] = item['SP'] / 100
            item['COND'] = item['COND'] / 400
            item['ML1'] = item['ML1'] / 5
            item['ML2'] = item['ML2'] / 5

            # somehow to normalize the data.
            # item = (item - item.min()) / (item.max() - item.min())
            # item = item / item.mean()

            np_item = np.array(item)
            np_item = np.delete(np_item, 0, axis=1)
            test_level_label_data.append([np_item.tolist(), i, int(file[:2])])
            print('test File {} of level {} has been read in.'.format(file, i))
            # break

test_sorted_data = sorted(test_level_label_data, key=lambda x: len(x[0]), reverse=True)
test_max_len = max(len(item[0]) for item in test_sorted_data)

test_len_list = list()
test_label_list = list()
test_region_list = list()

for item in test_sorted_data:
    test_len_list.append(int(len(item[0])))
    test_label_list.append(item[1])
    # label_list.append(int(item[1] / 4))

    test_region_list.append([item[2]])
    for i in range(test_max_len - len(item[0])):
        item[0].append([0, 0, 0, 0, 0, 0])

test_batch_seq = np.array([item[0] for item in test_sorted_data])

test_embed_input_x_packd = torch.nn.utils.rnn.pack_padded_sequence(torch.Tensor(test_batch_seq),
                                                                   torch.Tensor(test_len_list),
                                                                   batch_first=True).to(device)

test_region_tensor = torch.unsqueeze(torch.Tensor(test_region_list), 0).to(device)


# model layer
class LSTMLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_layer = torch.nn.LSTM(input_size=6, hidden_size=31, batch_first=True, num_layers=1)
        self.linear1 = torch.nn.Linear(32, 64)
        # self.linear2 = torch.nn.Linear(64, 2)
        self.linear2 = torch.nn.Linear(64, 8)

    def forward(self, x, region):
        encode_outputs_packed, (h_last, c_last) = self.lstm_layer(x)
        # encode_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(encode_outputs_packed, batch_first=True)
        last_level_output = h_last[-1:]
        x = torch.cat((last_level_output, region), 2)
        # x = h_last
        x = self.linear1(x)
        x = F.elu(x)
        x = self.linear2(x)

        return F.log_softmax(x[0], dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            m.weight.data.zero_()
            if m.bias is not None:
                m.bias.data.zero_()


model = LSTMLinear().to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
min_loss = 1
cur_loss = 0
model.train()
epoch = 0
while True:
    optim.zero_grad()
    out = model(embed_input_x_packd, region_tensor)
    loss = F.nll_loss(out, torch.LongTensor(label_list).to(device))
    cur_loss = float(loss)
    if cur_loss < min_loss:
        min_loss = cur_loss
    loss.backward()
    optim.step()

    # check train acc
    _, pre = model(embed_input_x_packd, region_tensor).max(dim=1)
    correct = int(pre.eq(torch.Tensor(label_list).to(device)).sum().item())
    total = int(len(label_list))

    # check test acc
    _, test_pre = model(test_embed_input_x_packd, test_region_tensor).max(dim=1)
    test_correct = int(test_pre.eq(torch.Tensor(test_label_list).to(device)).sum().item())
    test_total = int(len(test_label_list))

    print('In the {} epoch, the loss is {}.\nThe acc on train set is {}, the acc on test set is {}.'.format(epoch,
                                                                                                            float(loss),
                                                                                                            correct / total,
                                                                                                            test_correct / test_total))
    epoch += 1
    if epoch >= 5000 and cur_loss == min_loss:
        break

    # if float(loss) < 0.86:
    #     break
model.eval()

# test on tarin set
_, pre = model(embed_input_x_packd, region_tensor).max(dim=1)

cal_result = np.zeros([8, 8])

for i in range(len(pre)):
    cal_result[int(label_list[i])][int(pre[i])] += 1

print(cal_result)
correct = int(pre.eq(torch.Tensor(label_list).to(device)).sum().item())
total = int(len(label_list))
print('the correct rate on train set is {}.'.format(correct / total))

# test on test set
_, pre = model(test_embed_input_x_packd, test_region_tensor).max(dim=1)

cal_result = np.zeros([8, 8])

for i in range(len(pre)):
    cal_result[int(test_label_list[i])][int(pre[i])] += 1

print(cal_result)
correct = int(pre.eq(torch.Tensor(test_label_list).to(device)).sum().item())
total = int(len(test_label_list))
print('the correct rate on test set is {}.'.format(correct / total))
