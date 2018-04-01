import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
from torch.autograd import Variable

class CustomAlphaView(nn.Module):
    def __init__(self):
        super(CustomAlphaView, self).__init__()
        pass

    def forward(self, input):
        self.output = input.view(input.size(0), input.size(1), 1)
        return self.output