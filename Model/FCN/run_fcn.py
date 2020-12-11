from Model.FCN import fcn_model
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

item = pd.read_excel('/data/luckytiger/shengliOilWell/train_data/{}'.format('21-226.xlsx'))
t_input = item[['DEPTH', 'Por', 'Perm', 'AC', 'SP', 'COND', 'ML1', 'ML2']]
t_input = t_input / t_input.mean()
train_input = np.array(t_input)

model = fcn_model.FCN8s().to(device)
tensor_in = torch.tensor(train_input)
tensor_in = tensor_in.unsqueeze(0).permute(0, 2, 1).to(device, dtype=torch.float)
out = model(tensor_in)
out = out.squeeze(0).permute(1, 0)
print(out[0])
