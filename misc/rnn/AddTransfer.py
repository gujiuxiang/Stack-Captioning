import numpy as np
import torch.nn.init as init
import torch.nn as nn
import torch

class AddTransfer(nn.Module):
    def __init__(self, addend, negate):
        super(AddTransfer, self).__init__()
        self.addend = addend
        self.negate = (negate == None) and False or negate

    def forward(self, input):
        output = torch.add(input, self.addend)
        if self.negate:
            output = torch.mul(output, -1)
        return output
'''
from torch.autograd import Variable
a=Variable(torch.Tensor(20, 512),requires_grad=True)
b=Variable(torch.Tensor(20, 12, 512),requires_grad=True)

l2=AddTransfer(-1, True)
output = l2(a)
output = l2(b)
pass
'''