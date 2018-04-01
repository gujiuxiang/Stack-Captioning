from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import *
import misc.utils as utils
from misc.cnn.LanguageCNN import *
from misc.lm.Attention import *
from misc.rnn.RNNs import *

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.padd_idx = 0
        self.opt = opt
        self.stack = getattr(opt, 'stack', 0)
        self.attrs_in = getattr(opt, 'attrs_in', 0)
        self.attrs_out = getattr(opt, 'attrs_out', 0)
        self.lcnn_type = getattr(opt, 'lcnn_type', '')
        self.lcnn_core_width = getattr(opt, 'lcnn_core_width', 16)
        self.att_im = getattr(opt, 'att_im', 1)
        self.att_type = getattr(opt, 'att_type', 0)
        self.drop_prob_lm = getattr(opt, 'drop_prob_lm', 0.5)
        self.noise_action = getattr(opt, 'noise_action', 0)
        self.multi_rewards = getattr(opt, 'multi_rewards', 0)
        self.multi_rewards_type = getattr(opt, 'multi_rewards_type', 0)
        self.stack_pre_word = getattr(opt, 'stack_pre_word', 0)

        # If attrs_in == 1, then use ground-truth/predicted semantic words as input, and apply attention mechanism
        self.embed = nn.Sequential(nn.Embedding(opt.vocab_size + 1, opt.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(opt.drop_prob_lm))
        self.s_in_attention = Sentence_in_attention(opt.input_encoding_size, opt.rnn_size) if self.attrs_in else None
        self.i_in_attention = Image_In_Attention(opt) if self.att_im else None
        if self.stack==1:
            self.i_in_attention_1 = Image_In_Attention(opt) if self.att_im else None

        self.rnn_input_size = opt.input_encoding_size + opt.rnn_size
        '''
        self.core = RNNs(opt, input_size=self.rnn_input_size, hidden_size=self.rnn_size, num_layers=self.num_layers,
                         bias=False, dropout=self.drop_prob_lm)
        '''
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm_coarse = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)
        self.lang_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v

        self.weight_init()

    def weight_init(self):
        init.xavier_uniform(self.embed[0].weight, gain=np.sqrt(2))

        init.xavier_uniform(self.att_lstm.weight_ih, gain=np.sqrt(2))
        init.constant(self.att_lstm.bias_ih, 0.1)
        init.xavier_uniform(self.att_lstm.weight_hh, gain=np.sqrt(2))
        init.constant(self.att_lstm.bias_hh, 0.1)

        init.xavier_uniform(self.lang_lstm.weight_ih, gain=np.sqrt(2))
        init.constant(self.lang_lstm.bias_ih, 0.1)
        init.xavier_uniform(self.lang_lstm.weight_hh, gain=np.sqrt(2))
        init.constant(self.lang_lstm.bias_hh, 0.1)

    def init(self, ft_inputs=None):
        self.As = None
        self.word_idx = 0
        self.emb_attributes(ft_inputs)

    def emb_attributes(self, ft_inputs):
        [fc_feats, att_feats, p_att_feats, semantic_feats, semantic_feats_prob] = ft_inputs
        if self.attrs_in or self.attrs_out:
            self.As = self.embed(semantic_feats)
            if self.opt.attrs_prob_in:
                self.As = torch.mul(self.As,
                                    semantic_feats_prob.unsqueeze(2).expand(self.As.data.size()[0], self.As.data.size()[1],
                                                                            512))
        else:
            self.As = None

    def forward_visual(self, ft_inputs, hidden):
        [fc_feats, att_feats, p_att_feats, semantic_feats, semantic_feats_prob] = ft_inputs
        att_res = self.i_in_attention(ft_inputs, hidden) if self.att_im else fc_feats
        return att_res

    def forward_visual_1(self, ft_inputs, hidden, att_weights):
        [fc_feats, att_feats, p_att_feats, semantic_feats, semantic_feats_prob] = ft_inputs
        att_res = self.i_in_attention_1(ft_inputs, hidden, att_weights) if self.att_im else fc_feats
        return att_res

    def forward_language(self, ft_inputs=None, it=None):
        if self.word_idx == 0:
            xt = self.embed(it)
        else:
            xt = self.s_in_attention(self.embed(it), self.As) if self.attrs_in else self.embed(it)
        return xt

    def fixed_noize(self, batch_size=1, noize_dim=512):
        fixed_noise = torch.FloatTensor(batch_size, noize_dim).normal_(0, 1 / (self.word_idx+1))
        fixed_noise = Variable(fixed_noise.cuda(), requires_grad=False)
        return fixed_noise

    def distribution_sampler(self, mu=0.0, sigma=0.0):
        return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian

    def agent_noise(self, hidden):
        noise_hidden = torch.add(hidden, self.fixed_noize(hidden.size()[0], hidden.size()[1]))
        return noise_hidden

    def forward(self, ft_inputs=None, it=None, state=None, vis=None):
        [fc_feats, att_feats, p_att_feats, _, _] = ft_inputs
        prev_h = state[0][-1] if self.opt.num_layers == 1 else state[0][-1]
        xt = self.forward_language(ft_inputs, it=it)

        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))
        att, att_weights = self.forward_visual(ft_inputs, h_att)

        if self.multi_rewards:
            if self.stack_pre_word:
                lang_lstm_input_coarse = torch.cat([att, h_att, xt], 1)
            else:
                lang_lstm_input_coarse = torch.cat([att, h_att], 1)
            h_lang_coarse, c_lang_coarse = self.lang_lstm_coarse(lang_lstm_input_coarse, (state[0][1], state[1][1]))
            if self.stack:
                if self.att_type==1:
                    h_lang_coarse_ = torch.add(h_lang_coarse, att)
                    att_coarse, att_coarse_weights = self.forward_visual_1(ft_inputs, h_lang_coarse_, None)
                elif self.att_type==2:
                    att_coarse, att_coarse_weights = self.forward_visual_1(ft_inputs, h_lang_coarse, att_weights)
                elif self.att_type==3:
                    h_lang_coarse_ = torch.add(h_lang_coarse, att)
                    att_coarse, att_coarse_weights = self.forward_visual_1(ft_inputs, h_lang_coarse_, att_weights)
                else:
                    att_coarse, att_coarse_weights = self.forward_visual_1(ft_inputs, h_lang_coarse, att_weights)
            if self.stack_pre_word:
                lang_lstm_input = torch.cat([att_coarse, h_lang_coarse, xt], 1)
            else:
                lang_lstm_input = torch.cat([att_coarse, h_lang_coarse], 1)
        else:
            lang_lstm_input = torch.cat([att, h_att], 1)

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][self.multi_rewards+1], state[1][self.multi_rewards+1]))

        o_att = F.dropout(h_att, self.drop_prob_lm, self.training) if self.stack else None
        o_lang_coarse = F.dropout(h_lang_coarse, self.drop_prob_lm, self.training) if self.stack else None
        o_lang = F.dropout(h_lang, self.drop_prob_lm, self.training)
        #encode_output = torch.cat([xt, att], 1).unsqueeze(0)

        self.word_idx = self.word_idx + 1
        if self.multi_rewards:
            state = (torch.stack([h_att, h_lang_coarse, h_lang]), torch.stack([c_att, c_lang_coarse, c_lang]))
        else:
            state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        if self.multi_rewards_type<=1:
            return [o_lang_coarse, o_lang], state
        else:
            if vis:
                return [o_att, o_lang_coarse, o_lang], [att_weights, att_coarse_weights], state
            else:
                return [o_att, o_lang_coarse, o_lang], state
