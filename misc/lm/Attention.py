import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import misc.utils as utils
########################################################################################################################
# Image Attention
########################################################################################################################

class Image_In_Attention(nn.Module):
    def __init__(self, opt):
        super(Image_In_Attention, self).__init__()
        self.opt = opt
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.weight_init()

    def weight_init(self):
        init.xavier_uniform(self.h2att.weight, gain=np.sqrt(2))
        init.constant(self.h2att.bias, 0.1)
        init.xavier_uniform(self.alpha_net.weight, gain=np.sqrt(2))
        init.constant(self.alpha_net.bias, 0.1)

    def forward(self, ft_inputs, hidden, in_weights=None):
        #[fc_feats, att_feats, p_att_feats, semantic_feats, semantic_feats_prob] = ft_inputs
        att_feats = ft_inputs[1]
        p_att_feats = ft_inputs[2]
        #
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        if in_weights is not None:
            att = torch.mul(in_weights.unsqueeze(2).expand_as(att), att)
        att_h = self.h2att(hidden)                  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)   # batch * att_size * att_hid_size
        dot = att + att_h                           # batch * att_size * att_hid_size
        dot = F.tanh(dot)                           # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)       # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                   # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                # batch * att_size

        weight = F.softmax(dot)                     # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return att_res, weight

# Original two stage attention model, without connection path
class StackAttention(nn.Module):
    def __init__(self, opt):
        super(StackAttention, self).__init__()
        self.input_size = opt.input_size or 512
        self.att_size = opt.att_size or 512
        self.image_seq_size = opt.image_size or 14

        self.cap_em = nn.Linear(512, 512)
        self.im_em = nn.Linear(512, 512)
        self.h1_emb = nn.Linear(512, 1)
        self.att_softmax1 = nn.Softmax()

        self.cap_em2 = nn.Linear(512, 512)
        self.im_em2 = nn.Linear(512, 512)
        self.h1_emb2 = nn.Linear(512, 1)
        self.att_softmax2 = nn.Softmax()

    def forward(self, capft, convft):
        '''
        :param capft: batch_size, 512
        :param convft: batch_size, 14*14, 512
        :return: batch_size, 512
        '''

        caption_emb_1 = self.cap_em(capft)
        caption_emb_expand_1 = nn.ReplicationPad2d(196, 2)(caption_emb_1)
        img_emb_dim_1 = self.im_em(convft.view(-1,512))
        img_emb_1 = img_emb_dim_1.view(-1, 196, 512)
        h1 = nn.Tanh()(torch.add(caption_emb_expand_1, img_emb_1))
        h1_drop = nn.Dropout(0.5)(h1)
        h1_emb = self.h1_emb(h1_drop.view(-1, 512))
        p1 = self.att_softmax1(h1_emb.view(-1, 196))
        p1_att = p1.view(1, -1)
        img_att1 = torch.mul(p1_att, convft)
        img_att_feat_1 = img_att1.view(-1, 512)
        u1 = torch.add(capft, img_att_feat_1)

        caption_emb_2 = self.cap_em(u1)
        caption_emb_expand_2 = nn.ReplicationPad2d(196, 2)(caption_emb_2)
        img_emb_dim_2 = self.im_em2(convft.view(-1,512))
        img_emb_2 = img_emb_dim_2.view(-1, 196, 512)
        h2 = nn.Tanh()(torch.add(caption_emb_expand_2, img_emb_2))
        h2_drop = nn.Dropout(0.5)(h2)
        h2_emb = self.h2_emb(h2_drop.view(-1, 512))
        p2 = self.att_softmax2(h2_emb.view(-1, 196))
        p2_att = p2.view(1, -1)
        img_att2 = torch.mul(p2_att, convft)
        img_att_feat_2 = img_att2.view(-1, 512)
        u2 = torch.add(capft, img_att_feat_2)

        return u2

########################################################################################################################
# Sentence Attention
########################################################################################################################
'''
    input[1]: bz * xDim;
    input[2]: bz * oDim * yDim; output: bz * oDim;
    output: compute output scores: bz * L 
'''
class BilinearD3(nn.Module):
    def __init__(self, xDim, yDim, bias=True):
        super(BilinearD3, self).__init__()
        self.xDim = xDim
        self.yDim = yDim
        self.bias = bias or False
        self.x2y = nn.Linear(self.xDim, self.yDim, bias = self.bias)
        self.weight_init()

    def weight_init(self):
        init.xavier_uniform(self.x2y.weight, gain=np.sqrt(2))
        if self.bias:
            init.constant(self.x2y.bias, 0.1)

    '''
        # As = As.transpose(1, 2)  # input[2]: bz * oDim *yDim -->  bz * yDim *oDim
        #output = torch.bmm(map.view(-1, 1, self.yDim), As)  ## temp: (bz * 1 * xDim) * (bz * xDim * yDim) = bz * yDim
    '''
    def forward_bak(self, input, As):
        map = self.x2y(input) # bz * xDim --> bz * yDim
        output = Variable(torch.Tensor(As.size(0), As.size(1)).cuda(), requires_grad=True)
        for i in range(As.size(1)):
            output[:, i] = torch.sum(torch.bmm(map, As[:, i, :]), 1)

        return output

    def forward(self, input, As):
        map = self.x2y(input).unsqueeze(1) # bz * xDim --> bz * 1 * yDim
        As = As.transpose(1, 2)  # input[2]: bz * oDim *yDim -->  bz * yDim *oDim
        output = torch.bmm(map, As)  ## temp: (bz * 1 * xDim) * (bz * xDim * yDim) = bz * yDim

        return output.squeeze(1)

class Sentence_in_attention(nn.Module):
    def __init__(self, word_embed_size, m):
        super(Sentence_in_attention, self).__init__()
        self.m = m
        self.word_embed_size = word_embed_size

        self.BilinearD3 = BilinearD3(self.word_embed_size, self.word_embed_size, bias=False)
        self.l1 = nn.Linear(self.word_embed_size, self.m)
        self.weight_init()

    def weight_init(self):
        init.xavier_uniform(self.l1.weight, gain=np.sqrt(2))
        init.constant(self.l1.bias, 0.1)

    def makeWeightedSumUnit(self, x, alpha):
        g = torch.bmm(x.transpose(1,2), alpha.unsqueeze(2)) .squeeze(2)
        return g

        # prev_word_embedding: bz*512, As: bz*10*512
    def forward(self, prev_word_embed, As):
        attention_output = self.BilinearD3(prev_word_embed, As)  # bz*512
        beta = nn.Softmax()(attention_output)
        g_in = self.makeWeightedSumUnit(As, beta)  # bz*512
        temp = torch.add(g_in, prev_word_embed)
        output = self.l1(temp)
        return output


class Sentence_out_attention(nn.Module):
    def __init__(self, hDim, word_embed_size, output_size, drop_prob):
        super(Sentence_out_attention, self).__init__()
        self.drop_prob = drop_prob
        self.hDim = hDim
        self.word_embed_size = word_embed_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.rand(word_embed_size))
        self.BilinearD3 = BilinearD3(self.hDim, self.word_embed_size, bias=False)
        self.l1 = nn.Linear(self.word_embed_size, self.hDim)
        self.l2 = nn.Linear(self.hDim, self.output_size)
        self.weight_init()

    def weight_init(self):
        #init.xavier_uniform(self.l2.weight, gain=np.sqrt(2))
        init.xavier_uniform(self.l1.weight, gain=np.sqrt(2))
        init.constant(self.l1.bias, 0.1)
        init.xavier_uniform(self.l2.weight, gain=np.sqrt(2))
        init.constant(self.l2.bias, 0.1)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def makeWeightedSumUnit(self, x, alpha):
        g = torch.bmm(x.transpose(1,2), alpha.unsqueeze(2)) .squeeze(2)
        return g

        # h_t: bz*512, As: bz*10*512

    def forward(self, h_t, As):
        attention_output = self.BilinearD3(h_t, F.tanh(As.float()))
        beta = nn.Softmax()(attention_output)
        g_out = self.makeWeightedSumUnit(F.tanh(As), beta)  # bz*512
        output = torch.add(self.l1(torch.mul(self.weight.expand(g_out.size(0), g_out.size(1)), g_out)), h_t)
        output = nn.Dropout(self.drop_prob)(output) if self.drop_prob > 0 else output
        proj = self.l2(output)
        logsoft = F.log_softmax(proj)
        return logsoft
