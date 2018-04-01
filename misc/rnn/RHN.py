import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

class RHN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, dropout):
        super(RHN, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.i2t = nn.Linear(input_size, hidden_size)
        self.h2t = nn.Linear(hidden_size, hidden_size)
        self.i2c = nn.Linear(input_size, hidden_size)
        self.h2c = nn.Linear(hidden_size, hidden_size)
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(self.dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.xavier_uniform(m.weight, gain=np.sqrt(2))
                init.constant(m.bias, 0.1)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform(m.weight, gain=np.sqrt(2))
                init.constant(m.bias, 0.1)

    def forward(self, input, hidden):
        input = input.squeeze(0)
        hidden = hidden[0]
        hidden = self.tanh(torch.add(self.i2h(input), self.h2h(hidden)))
        transform = self.sigmoid(torch.add(self.i2t(input), self.h2t(hidden)))
        carry = self.sigmoid(torch.add(self.i2c(input), self.h2c(hidden)))
        hidden = torch.add(torch.mul(hidden,transform),torch.mul(carry,hidden))
        hidden = self.dropout(hidden)
        output = hidden
        return output.unsqueeze(0), hidden.unsqueeze(0)