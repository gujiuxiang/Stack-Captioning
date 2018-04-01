from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
from misc.cnn.LanguageCNN import *
from misc.lm.Attention import *

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.attrs_in = getattr(opt, 'attrs_in', 0)
        self.attrs_out = getattr(opt, 'attrs_out', 0)
        self.rnn_size = opt.rnn_size
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.drop_prob_lm = opt.drop_prob_lm
        # Output mapping layer, Note: remove the mapping layer from RNN(LSTM, GRU, etc.)
        if self.attrs_out:
            self.logit = Sentence_out_attention(self.rnn_size, self.input_encoding_size, self.vocab_size + 1,
                                                drop_prob=self.drop_prob_lm)
        else:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.weight_init()

    def weight_init(self):
        if self.attrs_out==0:
            init.xavier_uniform(self.logit.weight, gain=np.sqrt(2))
            init.constant(self.logit.bias, 0.1)

    def forward(self, x, As):
        if self.attrs_out:
            output_vocab = self.logit(x.squeeze(0), As)
        else:
            output_vocab = self.logit(x.squeeze(0))
        output = F.log_softmax(output_vocab)
        return output