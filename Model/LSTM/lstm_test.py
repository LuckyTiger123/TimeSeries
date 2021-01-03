import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch

device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

# train set
whole_table = pd.read_table('/data/luckytiger/synthetic_control/synthetic_control_TRAIN', sep=',', header=None)
label_list = np.array(whole_table.loc[:, 0])
time_series_data = np.array(whole_table.drop([0], axis=1))

# test set
test_whole_table = pd.read_table('/data/luckytiger/synthetic_control/synthetic_control_TEST', sep=',', header=None)
test_label_list = np.array(test_whole_table.loc[:, 0])
test_time_series_data = np.array(test_whole_table.drop([0], axis=1))


class LSTMLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_layer = torch.nn.LSTM(input_size=1, hidden_size=16, batch_first=True, num_layers=1)
        self.linear1 = torch.nn.Linear(16, 32)
        # self.linear2 = torch.nn.Linear(64, 2)
        self.linear2 = torch.nn.Linear(32, 6)

    def forward(self, x):
        encode_outputs_packed, (h_last, c_last) = self.lstm_layer(x)
        # encode_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(encode_outputs_packed, batch_first=True)
        # x = torch.cat((h_last, region), 2)
        x = h_last
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        return F.log_softmax(x[0], dim=1)


input_tensor = torch.unsqueeze(torch.Tensor(time_series_data), 2).to(device)
label_tensor = torch.LongTensor(label_list - 1).to(device)
model = LSTMLinear().to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.005)
model.train()
for epoch in range(400):
    optim.zero_grad()
    out = model(input_tensor)
    loss = F.nll_loss(out, label_tensor)
    loss.backward()
    optim.step()
    print('after {} epoch, the loss is {}.'.format(epoch, float(loss)))

model.eval()

# test on train set
_, pre = model(input_tensor).max(dim=1)
correct = int(pre.eq(label_tensor).sum().item())
total = int(len(label_list))
print('the correct rate is {} on train set.'.format(correct / total))

# test on the test set
test_tensor = torch.unsqueeze(torch.Tensor(test_time_series_data), 2).to(device)
test_label_tensor = torch.LongTensor(test_label_list - 1).to(device)
_, pre = model(test_tensor).max(dim=1)
correct = int(pre.eq(test_label_tensor).sum().item())
total = int(len(test_label_list))
print('the correct rate is {} on train set.'.format(correct / total))
