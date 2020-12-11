import torch
import torch.nn.functional as F

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
