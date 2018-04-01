import torch
import torch.nn as nn
from torch.autograd import Variable
from misc.rnn.LSTM import *
from misc.rnn.RHN import *
from misc.rnn.GRU import *

class RNNs(nn.Module):
    def __init__(self, opt, input_size=512, hidden_size=512, num_layers=1, bias=False, dropout=0, bidirectional=False):
        super(RNNs, self).__init__()
        self.rnn_type = opt.rnn_type
        if self.rnn_type == 'LSTM_FC':
            self.rnn = LSTM_FC(opt, input_size, hidden_size, num_layers,
                                                           bias=False, dropout=dropout, bidirectional=False)
        elif self.rnn_type == 'LSTM_ATT2IN2':
            self.rnn = LSTM_ATT2IN2(opt)
        elif self.rnn_type in ['LSTM', 'RNN', 'GRU']:
            self.rnn = getattr(nn, self.rnn_type.upper())(input_size, hidden_size, num_layers,
                                                           bias=False, dropout=dropout, bidirectional=False)
        elif self.rnn_type in ['BiLSTM', 'BiRNN', 'BiGRU']:
            self.rnn = getattr(nn, self.rnn_type.strip('Bi').upper())(input_size, hidden_size, num_layers,
                                                           bias=False, dropout=dropout, bidirectional=True)
        elif self.rnn_type in ['RHN']:
            self.rnn = RHN(input_size, hidden_size, num_layers, bias, dropout)

    def weight_init(self):
        #pre_init_path = 'self-critical/log_fc_rl/model-best.pth'
        pre_init_path = ''
        if len(pre_init_path)>0 :
            pre_init = torch.load(pre_init_path)
            if self.rnn_type == 'LSTM_FC':
                del self.core.i2h.weight
                self.rnn.i2h.weight = nn.Parameter(pre_init.items()[2][1])
                del self.core.i2h.bias
                self.rnn.i2h.bias = nn.Parameter(pre_init.items()[3][1])
                del self.core.h2h.weight
                self.rnn.h2h.weight = nn.Parameter(pre_init.items()[4][1])
                del self.core.h2h.bias
                self.rnn.h2h.bias = nn.Parameter(pre_init.items()[5][1])
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    init.xavier_uniform(m.weight, gain=np.sqrt(2))
                    init.constant(m.bias, 0.1)
                elif isinstance(m, nn.Linear):
                    init.xavier_uniform(m.weight, gain=np.sqrt(2))
                    init.constant(m.bias, 0.1)

    def forward(self, input, hidden):
        output, hidden_t = self.rnn(input, hidden)
        return output, hidden_t