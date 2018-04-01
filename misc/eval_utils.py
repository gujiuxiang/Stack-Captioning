from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import gc
from tqdm import tqdm
import skimage
import matplotlib.pyplot as plt
import skimage.io
from scipy.misc import imread, imresize
from caption_eval.run_evaluations import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Dump_Outputs(nn.Module):
    def __init__(self, opt):
        super(Dump_Outputs, self).__init__()
        self.opt = opt
        self.html=''
        self.head = '<html>'\
                    '<body>'\
                    '<h1>Image Captions and Reconstructed Images</h1>'\
                    '<table border="1px solid gray" style="width=100%">'\
                    '<tr>'\
                    '<td><b>Image</b></td>'\
                    '<td><b>ID</b></td>'\
                    '<td><b>Image ID</b></td>'\
                    '<td><b>Generated Caption coarse</b></td>' \
                    '<td><b>Generated Caption fine</b></td>'\
                    '<td><b>Generated Caption final</b></td>'\
                    '<td><b>Ground truth</b></td>'\
                    '</tr>'
        self.fname_html = 'eval_results/caption_' + self.opt.model_name + '.html'
    def create_html(self):
        os.system('rm -rf ' + format(self.fname_html))
        os.system('echo ' + '"' + self.head + '"' + ' >> ' + self.fname_html)

    def append_html(self, str):
        #self.html = self.html+str
        os.system('echo ' + '"' + str + '"' + ' >> ' + self.fname_html)

    def dump_all(self):
        self.html = self.html + '</html>'
        os.system('echo ' + '"' + self.html + '"' + ' >> ' + self.fname_html)

def language_eval_chinese(dataset, preds, model_id, split):
    from caption_eval.coco_caption.pycxtools.coco import COCO
    from caption_eval.coco_caption.pycxevalcap.eval import COCOEvalCap
    m1_score = {}
    m1_score['error'] = 0

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    reference_file = 'coco-caption/annotations/coco_val_caption_validation_annotations_20170910.json'
    coco = COCO(reference_file)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print(preds_filt)
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    coco_res = coco.loadRes(cache_path)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params['image_id'] = coco_res.getImgIds()
    print(coco_res.getImgIds())
    # evaluate results
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print('%s: %.3f' % (metric, score))
        m1_score[metric] = score

    return m1_score

def language_eval(dataset, preds, model_id, split):
    import sys
    if 'chinese' in dataset:
        sys.path.append("coco-caption")
        annFile = 'coco-caption/annotations/coco_caption_validation_annotations_20170910.json'
    elif 'coco' in dataset:
        sys.path.append("coco-caption")
        annFile = 'coco-caption/annotations/captions_val2014.json'
    elif 'flickr30k' in dataset:
        sys.path.append("f30k-caption")
        annFile = 'f30k-caption/annotations/dataset_flickr30k.json'
    print(annFile)

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(opt, infos, cnn_model, model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    lang_eval = getattr(opt, 'lang_eval', 1)
    pre_ft = getattr(opt, 'pre_ft', 1)
    server = getattr(opt, 'server', 0)
    stack = getattr(opt, 'stack', 0)
    dump_output = getattr(opt, 'dump_output', 0)
    disp_coarse_to_fine = getattr(opt, 'disp_coarse_to_fine', 0)

    if dump_output:
        dump_html = Dump_Outputs(opt)
        dump_html.create_html()

    if not pre_ft: cnn_model.eval()
    model.eval()

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {}".format(split))
    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    predictions_fine0 = []
    predictions_mle = []
    # If testing the images from "vis" file loader, then only set the bz=1
    with tqdm(total=val_images_use) as pbar:
        while True:
            '''
            collect memory
            '''
            gc.collect()
            pbar.update(loader.batch_size)

            data = loader.get_batch(split)
            n = n + loader.batch_size

            # forward the model to get loss
            images = Variable(torch.from_numpy(data['images']), volatile=True).cuda() if not opt.pre_ft else None
            _fc_feats = fc_feats = Variable(torch.from_numpy(data['fc_feats']), volatile=True).cuda() if opt.pre_ft else None
            _att_feats = att_feats = Variable(torch.from_numpy(data['att_feats']), volatile=True).cuda() if opt.pre_ft and opt.att_im else None
            _semantic_feats = semantic_feats = Variable(torch.from_numpy(data['semantic_words']), volatile=True).cuda() if opt.attrs_in or opt.attrs_out else None
            _semantic_feats_prob = semantic_feats_prob = Variable(torch.from_numpy(data['semantic_words_prob']), volatile=True).cuda() if opt.attrs_in or opt.attrs_out else None
            if server==0:
                labels = Variable(torch.from_numpy(data['labels']), volatile=True).cuda()
                masks = Variable(torch.from_numpy(data['masks']), volatile=True).cuda()

            if not pre_ft:
                # forward cnn model to get image features
                if opt.cnn_model == 'resnet101':
                    att_feats = _att_feats = cnn_model(images)
                    fc_feats = _fc_feats = att_feats.mean(2).mean(3).squeeze(2).squeeze(2)
                elif opt.cnn_model == 'resnet152':
                    att_feats = _att_feats = cnn_model(images)
                    fc_feats = _fc_feats = att_feats.mean(2).mean(3).squeeze(2).squeeze(2)
                elif opt.cnn_model == 'vgg16':
                    fc_feats = _fc_feats = cnn_model(images)

            if not opt.pool_loader:
                fc_feats = fc_feats.unsqueeze(1).expand(
                    *((fc_feats.size(0), loader.seq_per_img,) + fc_feats.size()[1:])).contiguous().view(
                    *((fc_feats.size(0) * loader.seq_per_img,) + fc_feats.size()[1:]))
                if opt.att_im:
                    att_feats = att_feats.unsqueeze(1).expand(
                        *((att_feats.size(0), loader.seq_per_img,) + att_feats.size()[1:])).contiguous().view(
                        *((att_feats.size(0) * loader.seq_per_img,) + att_feats.size()[1:]))

            '''
            Generate sequences
            '''
            if opt.pool_loader:
                tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                       data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
                fc_feats, att_feats = tmp
            else:
                att_feats = _att_feats if opt.att_im else None
                fc_feats = _fc_feats
                semantic_feats = _semantic_feats if opt.attrs_in or opt.attrs_out else None
                semantic_feats_prob = _semantic_feats_prob if opt.attrs_in or opt.attrs_out else None
            sample_inputs = [fc_feats, att_feats, None, semantic_feats, semantic_feats_prob]

            # Used for testing
            seq, seq_ = model.sample(sample_inputs, {'beam_size': beam_size})
            sents = utils.decode_sequence_chinese(loader.get_vocab(), seq) if 'chinese' in opt.input_json else utils.decode_sequence(
                loader.get_vocab(), seq)
            for k, sent in enumerate(sents):
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                predictions.append(entry)
                if verbose: print(
                    'Beam size: %d, image %s: %s' % (beam_size, entry['image_id'], entry['caption']))

            ix0 = data['bounds']['it_pos_now']
            ix1 = data['bounds']['it_max']
            if val_images_use != -1:
                ix1 = min(ix1, val_images_use)
            for i in range(n - ix1):
                predictions.pop()
            if server==0 and verbose:
                print('evaluating validation preformance... %d/%d (%f, with coarse_loss %f)' % (ix0 - 1, ix1, loss, 0.0))

            if data['bounds']['wrapped']:
                break
            if n >= val_images_use:
                break

    if dump_output:
        print('Dunmp html files')
        dump_html.dump_all()

    if lang_eval == 1 and server==0:
        if 'chinese' in opt.input_json:
            print('Language evaluation')
            lang_stats = language_eval_chinese(dataset, predictions, eval_kwargs['model_id'], split)
        else:
            lang_stats = language_eval(dataset, predictions, eval_kwargs['model_id'], split)

    gc.collect()

    # Switch back to training mode
    model.train()
    if server==0:
        return loss_sum / loss_evals, predictions, lang_stats
    else:
        return predictions