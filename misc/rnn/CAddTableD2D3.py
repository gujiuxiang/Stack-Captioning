import numpy as np
import torch.nn.init as init
import torch.nn as nn
import torch

class CAddTableD2D3(nn.Module):
    def __init__(self, xDim, oDim):
        super(CAddTableD2D3, self).__init__()
        self.xDim = xDim
        self.oDim = oDim

    def forward(self, input, As):
        '''
        '''
        '''
        input[1]: bz * xDim
        input[2]: bz * oDim * xDim
        output: bz * oDim
        '''
        input = input.unsqueeze(1)
        input = torch.add(input.expand_as(As), As)
        output = input.sum(2)
        return output.squeeze(2)
'''
from torch.autograd import Variable
a=Variable(torch.Tensor(20, 512),requires_grad=True)
b=Variable(torch.Tensor(20, 12, 512),requires_grad=True)

l2=CAddTableD2D3(512, 12)
output = l2(a, b)
pass
'''