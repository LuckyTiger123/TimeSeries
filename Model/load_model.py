import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv1d(8, 16, 5, padding=2)
        self.conv2 = torch.nn.Conv1d(16, 9, 5, padding=2)

    def forward(self, data):
        x = self.conv1(data)
        x = F.relu(x)
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = '1607419350_387761'
model = torch.load('/data/luckytiger/shengliOilWell/train_result/model/{}.pkl'.format(model_name))
model = model.to(device)

test_file_name = '37-386.xlsx'
item = pd.read_excel('/data/luckytiger/shengliOilWell/test_data/{}'.format(test_file_name))
t_input = item[['DEPTH', 'Por', 'Perm', 'AC', 'SP', 'COND', 'ML1', 'ML2']]
tensor_in = torch.tensor(np.array(t_input))
tensor_in = tensor_in.unsqueeze(0).permute(0, 2, 1).to(device, dtype=torch.float)
_, pred = model(tensor_in).max(dim=1)
pred = pred.squeeze(0)
pred = pred + 60
a = pred.cpu().numpy()
d = pd.DataFrame({'judgement': a})
