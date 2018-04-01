from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
import torch
from torchvision import transforms as trn
from multiprocessing.dummy import Pool
from misc.utils import *
import math
import gc

preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

preprocess_vgg16 = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([123.680, 103.939, 116.779], [1.000, 1.000, 1.000])
])
'''
# batch: Bx3xHxW BGR [0,255] Variable
def preprocess_vgg16(batch):
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 123.680
    mean[:, 2, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    batch -= mean
'''
def get_npy_data(ix, fc_file, att_file):
    return (np.load(fc_file),
        np.load(att_file)['feat'],
        ix)

'''
Load data from h5 files
'''
class DataLoader():
    def reset_iterator(self, split):
        # if load files from directory, then reset the prefetch process
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        vocabulary = self.ix_to_word
        return vocabulary

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.type = 'h5'
        self.opt = opt
        self.cnn_model = getattr(opt, 'cnn_model', 'resnet101')
        self.attrs_in = getattr(opt, 'attrs_in', 0)
        self.attrs_out = getattr(opt, 'attrs_out', 0)
        self.att_im = getattr(opt, 'att_im', 1)
        self.pre_ft = getattr(opt, 'pre_ft', 1)

        self.top_attrs = 10
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.batch_size = opt.batch_size
        self.seq_per_img = opt.seq_per_img
        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))

        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        # open the hdf5 file
        self.h5_fc_file = h5py.File(self.opt.input_fc_h5)
        if self.att_im: self.h5_att_file = h5py.File(self.opt.input_att_h5)

        # load det semantic words
        if len(self.opt.input_attrs_h5)>0 and self.attrs_in or self.attrs_out:
            self.h5_attrs_file = h5py.File(self.opt.input_attrs_h5, 'r', driver='core')

        # load in the sequence data
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]
        semantic_attrs_size = self.h5_label_file['semantic_words'].shape if self.attrs_in or self.attrs_out else 0
        self.semantic_attrs_length = semantic_attrs_size[1] if self.attrs_out or self.attrs_in else 0
        print('max sequence length in data is', self.seq_length)
        print('max semantic words length in data is', self.semantic_attrs_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0]
        print('read %d image / features' %(self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        split_ix = self.split_ix[split]
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        fc_batch = np.ndarray((batch_size, self.fc_feat_size), dtype = 'float32') if self.pre_ft else None
        att_batch = np.ndarray((batch_size, 14, 14, self.att_feat_size), dtype = 'float32') if self.pre_ft and self.att_im else None
        img_batch = np.ndarray([batch_size, 3, 224, 224], dtype='float32') if not(self.pre_ft) else None
        label_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype = 'int')
        mask_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype = 'float32')
        attrs_batch = np.zeros([batch_size, self.top_attrs], dtype = 'int') if self.attrs_in or self.attrs_out else None
        attrs_prob_batch = np.zeros([batch_size, self.top_attrs], dtype = 'float32') if self.attrs_in or self.attrs_out else None

        max_index = len(split_ix)
        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            import time
            t_start = time.time()

            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            ix = split_ix[ri]

            # fetch image
            if self.pre_ft: fc_batch[i] = self.h5_fc_file['fc'][ix, :]
            if self.pre_ft and self.att_im: att_batch[i] = self.h5_att_file['att'][ix, :, :, :]

            if not(self.pre_ft):
                #img = self.load_image(self.image_info[ix]['filename'])
                img = self.h5_im_file['images'][ix, :, :, :]
                if self.cnn_model == 'resnet101':
                    img_batch[i] = preprocess(torch.from_numpy(img[:, 16:-16, 16:-16].astype('float32')/255.0)).numpy()
                else:
                    img_batch[i] = preprocess_vgg16(torch.from_numpy(img[:, 16:-16, 16:-16].astype('float32'))).numpy()

            # fetch the semantic_attributes
            if self.attrs_in or self.attrs_out:
                if len(self.opt.input_attrs_h5)>0:
                    attrs_batch[i] = self.h5_attrs_file['pred_semantic_words'][ix, : self.top_attrs]
                    attrs_prob_batch[i] = self.h5_attrs_file['pred_semantic_words_prob'][ix, : self.top_attrs]
                else:
                    attrs_batch[i] = self.h5_label_file['semantic_words'][ix, : self.top_attrs]

            # fetch the sequence labels
            ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
            ix2 = self.label_end_ix[ix] - 1
            ncap = ix2 - ix1 + 1 # number of captions available for this image
            assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

            if ncap < self.seq_per_img:
                # we need to subsample (with replacement)
                seq = np.zeros([self.seq_per_img, self.seq_length], dtype = 'int')
                for q in range(self.seq_per_img):
                    ixl = random.randint(ix1,ix2)
                    seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]

            else:
                ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
                seq = self.h5_label_file['labels'][ixl: ixl + self.seq_per_img, :self.seq_length]

            label_batch[i * self.seq_per_img : (i + 1) * self.seq_per_img, 1 : self.seq_length + 1] = seq

            # Used for reward evaluation
            gts_labels = self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]]
            gts.append(gts_labels)

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        # generate mask
        t_start = time.time()
        nonzeros = np.array(map(lambda x: (x != 0).sum()+2, label_batch))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        data = {}

        data['fc_feats'] = fc_batch # if pre_ft is 0, then it equals None
        data['att_feats'] = att_batch # if pre_ft is 0, then it equals None
        data['images'] = img_batch # if pre_ft is 1, then it equals None
        data['semantic_words'] = attrs_batch # if attributes is 1, then it equals None
        data['semantic_words_prob'] = attrs_prob_batch  # if attributes is 1, then it equals None
        data['labels'] = label_batch
        data['gts'] = gts
        data['masks'] = mask_batch
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(split_ix), 'wrapped': wrapped}
        data['infos'] = infos

        gc.collect()

        return data

class DataLoader_pool():
    def reset_iterator(self, split):
        self._prefetch_process[split].terminate()
        self._prefetch_process[split].join()
        self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.type = 'pool'
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.top_attrs = 10
        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')

        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        semantic_attrs_size = self.h5_label_file['semantic_words'].shape
        self.semantic_attrs_length = semantic_attrs_size[1]
        print('max semantic words length in data is', self.semantic_attrs_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0]
        print('read %d image features' % (self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0:  # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' % len(self.split_ix['train']))
        print('assigned %d images to split val' % len(self.split_ix['val']))
        print('assigned %d images to split test' % len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')
            # Terminate the child process when the parent exists

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                self._prefetch_process[split].terminate()
                self._prefetch_process[split].join()

        import atexit
        atexit.register(cleanup)

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        fc_batch = []  # np.ndarray((batch_size * seq_per_img, self.opt.fc_feat_size), dtype = 'float32')
        att_batch = []  # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')
        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='int')
        attrs_batch = np.zeros([batch_size * seq_per_img, self.top_attrs], dtype='int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='float32')

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            import time
            t_start = time.time()
            # fetch image
            tmp_fc, tmp_att, \
            ix, tmp_wrapped = self._prefetch_process[split].get()
            fc_batch += [tmp_fc] * seq_per_img
            att_batch += [tmp_att] * seq_per_img

            # fetch the sequence labels
            ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
            ix2 = self.label_end_ix[ix] - 1
            ncap = ix2 - ix1 + 1  # number of captions available for this image
            assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

            # fetch the semantic_attributes
            attrs_batch[i] = self.h5_label_file['semantic_words'][ix, : self.top_attrs]

            if ncap < seq_per_img:
                # we need to subsample (with replacement)
                seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
                for q in range(seq_per_img):
                    ixl = random.randint(ix1, ix2)
                    seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
            else:
                ixl = random.randint(ix1, ix2 - seq_per_img + 1)
                seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]

            label_batch[i * seq_per_img: (i + 1) * seq_per_img, 1: self.seq_length + 1] = seq

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)
            # print(i, time.time() - t_start)

        # generate mask
        t_start = time.time()
        nonzeros = np.array(map(lambda x: (x != 0).sum() + 2, label_batch))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        # print('mask', time.time() - t_start)

        data = {}
        data['fc_feats'] = np.stack(fc_batch)
        data['att_feats'] = np.stack(att_batch)
        data['labels'] = label_batch
        data['semantic_words'] = attrs_batch
        data['gts'] = gts
        data['masks'] = mask_batch
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]),
                          'wrapped': wrapped}
        data['infos'] = infos

        return data

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

        self.pool = Pool()
        self.fifo = []

    # Add more in the queue
    def reset(self):
        if len(self.fifo) == 0:
            self.cur_idx = self.dataloader.iterators[self.split]
            self.cur_split_ix = self.dataloader.split_ix[self.split][:]  # copy
        for i in xrange(512 - len(self.fifo)):
            ix = self.cur_split_ix[self.cur_idx]
            if self.cur_idx + 1 >= len(self.cur_split_ix):
                self.cur_idx = 0
                if self.if_shuffle:
                    random.shuffle(self.cur_split_ix)
            else:
                self.cur_idx += 1
            self.fifo.append(self.pool.apply_async(get_npy_data, \
                                                   (ix, \
                                                    os.path.join(self.dataloader.input_fc_dir, str(
                                                        self.dataloader.info['images'][ix]['id']) + '.npy'),
                                                    os.path.join(self.dataloader.input_att_dir,
                                                                 str(self.dataloader.info['images'][ix][
                                                                         'id']) + '.npz')
                                                    )))

    def terminate(self):
        while len(self.fifo) > 0:
            self.fifo.pop(0).get()
        self.pool.terminate()
        print(self.split, 'terminated')

    def join(self):
        self.pool.join()
        print(self.split, 'joined')

    def _get_next_minibatch_inds(self):
        max_index = len(self.cur_split_ix)
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            self.dataloader.split_ix[self.split] = self.cur_split_ix[:]  # copy
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if len(self.fifo) < 400:
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.fifo.pop(0).get()

        assert tmp[2] == ix, "ix not equal"

        return tmp + (wrapped,)
