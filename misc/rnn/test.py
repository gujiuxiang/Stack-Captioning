"""
Simple example using the operations defined in :mod:`linear`.
"""

import numpy
import torch
from misc.rnn.LinearD2 import linear
from misc.rnn.LinearD2 import LinearNN
from misc.rnn.LinearD2 import LinearOp
import random


def simple_example():
    """
    Simple example illustrating forward and backward pass through the linear operation.
    """

    input = torch.autograd.Variable(torch.rand(1, 3), requires_grad=True)
    weights = torch.autograd.Variable(torch.rand(1, 3))
    bias = torch.autograd.Variable(torch.zeros(1))

    output = linear(input, weights, bias)
    output.backward()
    print(input.grad)


class Net(torch.nn.Module):
    """
    Network definition consisting of one fully connected (linear) layer followed by a softmax with two outputs.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = LinearNN(6, 2, True)

    def forward(self, x):
        x = self.fc1(x)
        return torch.nn.functional.softmax(x)


def nn_example():
    """
    Simple, linearly separable classification example.
    """

    N = 1000
    inputs = numpy.zeros((N, 6))
    outputs = numpy.zeros((N, 1))
    for n in range(N):
        outputs[n, 0] = random.randint(0, 1)
        if outputs[n, 0] > 0:
            inputs[n, 0:3] = 1
        else:
            inputs[n, 3:6] = 1

    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    for t in range(100):
        indices = numpy.arange(N)
        numpy.random.shuffle(indices)
        indices = indices[0:10]

        data = torch.autograd.Variable(torch.Tensor(inputs[indices]))
        target = torch.autograd.Variable(torch.Tensor(outputs[indices]))
        pred = model(data)
        # better pass long to loss: https://discuss.pytorch.org/t/problems-with-target-arrays-of-int-int32-types-in-loss-functions/140
        # also target has to be 1D - https://github.com/torch/cutorch/issues/227
        loss = torch.nn.functional.nll_loss(pred, target.resize(10).long())
        loss.backward()
        optimizer.step()

        if t % 10 == 0:
            pred = model(torch.autograd.Variable(torch.Tensor(inputs)))
            # use pred.data to get from a torch.autograd.Variable to the underlying Tensor
            accuracy = (pred.data.numpy().argmax(1).reshape((N)) == outputs.reshape((N))).astype(int).sum() / float(N)
            print(accuracy)


if __name__ == '__main__':
    # simple_example()
    nn_example()
