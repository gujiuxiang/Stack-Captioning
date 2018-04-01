from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import *
from misc.lm.Encoder import Encoder
from misc.lm.Decoder import Decoder
import misc.utils as utils

class LanguageModel(nn.Module):
    def __init__(self, opt):
        super(LanguageModel, self).__init__()
        self.batch_size = opt.batch_size
        self.seq_per_img = opt.seq_per_img

        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm

        self.seq_length = opt.seq_length
        self.ss_prob = 0.0  # Schedule sampling probability

        self.att_im = getattr(opt, 'att_im', 1)
        self.attrs_in = getattr(opt, 'attrs_in', 0)
        self.attrs_out = getattr(opt, 'attrs_out', 0)
        self.hidden_type = getattr(opt, 'hidden_type', 0)
        self.noise_action = getattr(opt, 'noise_action', 0)
        self.greedy_decoding = getattr(opt, 'greedy_decoding', 0)
        self.multi_rewards = getattr(opt, 'multi_rewards', 0)
        self.split = getattr(opt, 'split', 'test')
        self.multi_rewards_type = getattr(opt, 'multi_rewards_type', 0)
        self.stack_pre_word = getattr(opt, 'stack_pre_word', 0)

        # modules with parameters
        self.fc_embed = nn.Sequential(nn.Linear(opt.fc_feat_size, opt.rnn_size),
                                       nn.ReLU(),
                                       nn.Dropout(opt.drop_prob_lm))
        self.att_embed = nn.Sequential(nn.Linear(opt.att_feat_size, opt.rnn_size),
                                       nn.ReLU(),
                                       nn.Dropout(opt.drop_prob_lm))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.encoder = Encoder(opt)
        self.decoder_bleu = Decoder(opt)
        self.decoder = Decoder(opt)

        self.rand_init_weights()

    def rand_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.xavier_uniform(m.weight, gain=np.sqrt(2))
                init.constant(m.bias, 0.1)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform(m.weight, gain=np.sqrt(2))
                init.constant(m.bias, 0.1)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if 'LSTM' in self.rnn_type:
            return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                    Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_())

    def project_image_ft(self, ft_inputs):
        [fc_feats, att_feats, p_att_feats, semantic_feats, semantic_feats_prob] = ft_inputs
        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(*(att_feats.size()[:-1] + (self.rnn_size,)))

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.rnn_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))
        return [fc_feats, att_feats, p_att_feats, semantic_feats, semantic_feats_prob]

    def train_greedy_decoding(self, logprobs):
        sampleLogprobs, it = torch.max(logprobs.data, 1)
        it = it.view(-1).long()
        it = Variable(it, requires_grad=False)
        return it

    def train_scheduled_sampling(self, i, ft_inputs, seq, batch_size):
        sample_prob = ft_inputs[0].data.new(batch_size).uniform_(0, 1)
        sample_mask = sample_prob < self.ss_prob
        if sample_mask.sum() == 0:
            it = seq[:, i].clone()
        else:
            sample_ind = sample_mask.nonzero().view(-1)
            it = seq[:, i].data.clone()
            # prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
            # it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
            prob_prev = torch.exp(self.outputs[-1].data)  # fetch prev distribution: shape Nx(M+1)
            it.index_copy_(0, sample_ind,
                           torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            it = Variable(it, requires_grad=False)
        return it


    def sample_greedy_decoding(self, logprobs):
        sampleLogprobs, it = torch.max(logprobs.data, 1)
        it = it.view(-1).long()
        sampleLogprobs = Variable(sampleLogprobs, requires_grad=False)
        #it = Variable(it, requires_grad=False)
        return sampleLogprobs, it

    def sample_scheduled_sampling(self, temperature, logprobs):
        if temperature == 1.0:
            prob_prev = torch.exp(logprobs.data).cpu()  # fetch prev distribution: shape Nx(M+1)
        else:
            # scale logprobs by temperature
            prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
        it = torch.multinomial(prob_prev, 1).cuda()
        sampleLogprobs = logprobs.gather(1, Variable(it,
                                                     requires_grad=False))  # gather the logprobs at sampled positions
        it = it.view(-1).long()  # and flatten indices for downstream processing
        return sampleLogprobs, it

    def sample(self, ft_inputs, opt={}, vis=None):
        return self.sample_beam(ft_inputs, opt)

    def beam_search(self, k, t, logprobs, state, beam_size):
        """perform a beam merge. that is,
                            for every previous beam we now many new possibilities to branch out
                            we need to resort our beams to maintain the loop invariant of keeping
                            the top beam_size most likely sequences."""
        logprobsf = logprobs.float()  # lets go to CPU for more efficiency in indexing operations
        ys, ix = torch.sort(logprobsf, 1,
                            True)  # sorted array of logprobs along each previous beam (last true = descending)
        candidates = []
        cols = min(beam_size, ys.size(1))
        rows = beam_size
        if t == 1:  # at first time step only the first beam is active
            rows = 1
        for c in range(cols):
            for q in range(rows):
                # compute logprob of expanding beam q with word in (sorted) position c
                local_logprob = ys[q, c]
                candidate_logprob = self.beam_logprobs_sum[q] + local_logprob
                candidates.append(
                    {'c': ix.data[q, c], 'q': q, 'p': candidate_logprob.data[0], 'r': local_logprob.data[0]})
        candidates = sorted(candidates, key=lambda x: -x['p'])

        # construct new beams
        if 'LSTM' in self.rnn_type:
            new_state = [_.clone() for _ in state]
        else:
            new_state = state.clone()
        if t > 1:
            # well need these as reference when we fork beams around
            beam_seq_prev = self.beam_seq[:t - 1].clone()
            beam_seq_logprobs_prev = self.beam_seq_logprobs[:t - 1].clone()
        for vix in range(beam_size):
            v = candidates[vix]
            # fork beam index q into index vix
            if t > 1:
                self.beam_seq[:t - 1, vix] = beam_seq_prev[:, v['q']]
                self.beam_seq_logprobs[:t - 1, vix] = beam_seq_logprobs_prev[:, v['q']]

            # rearrange recurrent states
            if 'LSTM' in self.rnn_type:
                for state_ix in range(len(new_state)):
                    # copy over state in previous beam q to new beam at vix
                    new_state[state_ix][0, vix] = state[state_ix][0, v['q']]  # dimension one is time step
            else:
                new_state[0, vix] = state[0, v['q']]

            # append new end terminal at the end of this beam
            self.beam_seq[t - 1, vix] = v['c']  # c'th word is the continuation
            self.beam_seq_logprobs[t - 1, vix] = v['r']  # the raw logprob here
            self.beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam

            if v['c'] == 0 or t == self.seq_length:
                # END token special case here, or we reached the end.
                # add the beam to a set of done beams
                self.done_beams[k].append({'seq': self.beam_seq[:, vix].clone(),
                                           'logps': self.beam_seq_logprobs[:, vix].clone(),
                                           'p': self.beam_logprobs_sum[vix]
                                           })

        # encode as vectors
        it = self.beam_seq[t - 1]
        # t>=1
        state = new_state
        return it, state

    def sample_beam(self, ft_inputs, opt={}, vis=None):
        ft_inputs = self.project_image_ft(ft_inputs)
        [fc_feats, att_feats, p_att_feats, _, __] = ft_inputs
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)
        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            #state = self.init_hidden(beam_size, fc_feats[k:k + 1].expand(beam_size, fc_feats.size()[1]))
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1)).contiguous()
            tmp_att_feats = att_feats[k:k+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous() if self.att_im else None
            tmp_p_att_feats = p_att_feats[k:k + 1].expand(*((beam_size,) + p_att_feats.size()[1:])).contiguous() if self.att_im else None

            bs_ft_inputs = [tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, None, None]
            self.encoder.init(ft_inputs=bs_ft_inputs)

            self.beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            self.beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            self.beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam

            for t in range(self.seq_length + 1):
                if t == 0:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                else:
                    it, state = self.beam_search(k, t, logprobs, state, beam_size)

                it = Variable(it.cuda(), requires_grad=False)
                if vis:
                    encode_output, state, weights = self.encoder(ft_inputs=bs_ft_inputs, it=it, state=state)
                else:
                    encode_output, state = self.encoder(ft_inputs=bs_ft_inputs, it=it, state=state)
                logprobs = self.decoder(encode_output[-1], self.encoder.As)

            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)