from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import json
import numpy as np

import torch
import time
import os
from six.moves import cPickle
import opts
import misc.utils as utils
from misc import eval_utils
from misc.dataloader import *
from misc.dataloader.dataloader import *
from misc.dataloader.dataloader_pool import *
from models import *

NUM_THREADS = 2  # int(os.environ['OMP_NUM_THREADS'])

# Input arguments and options
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='debug',help='')
parser.add_argument('--caption_model', type=str, default='debug',help='')
parser.add_argument('--start_from', type=str, default='save/09021100.lm_stack_cap_2finestages', help='None')
parser.add_argument('--data_type', type=int, default=1)  # 'use pool to load pre-extracted features from dir'
# Model settings
parser.add_argument('--exchange_vocab', type=int, default=0)
# Input paths
parser.add_argument('--disp_coarse_to_fine', type=int, default=0)
parser.add_argument('--dump_output', type=int, default=0)
parser.add_argument('--shuffle', type=int, default=0)#load shuffled data from h5 or dir
parser.add_argument('--pool_loader', type=int, default=0)#use pool to load pre-extracted features from dir
parser.add_argument('--pre_ft', type=int, default=1)#pre extracted feature input
# Basic options
parser.add_argument('--batch_size', type=int, default=50)#if > 0 then overrule, otherwise load from checkpoint.
parser.add_argument('--val_images_use', type=int, default=5000)#how many images to use when periodically evaluating the loss? (-1 = all)
parser.add_argument('--language_eval', type=int, default=0)
parser.add_argument('--dump_images', type=int, default=0) #Dump images into vis/imgs folder for vis? (1=yes,0=no)
parser.add_argument('--dump_json', type=int, default=0) #Dump json with predictions into vis folder? (1=yes,0=no)
parser.add_argument('--dump_path', type=int, default=0) #Write image paths along with predictions into vis json? (1=yes,0=no)
# Sampling options
parser.add_argument('--sample_max', type=int, default=1) #1 = sample argmax words. 0 = sample from distributions.
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--temperature', type=float, default=1.0)
# For evaluation on a folder of images:
parser.add_argument('--server', type=int, default=0)#if running on MSCOCO challenge
parser.add_argument('--server_best', type=int, default=1)#if running on MSCOCO challenge
parser.add_argument('--split', type=str, default='test')#if running on MSCOCO images, which split to use: val|test|train
parser.add_argument('--coco_json', type=str, default='data/mscoco/annotations/image_info_test2014.json')#if nonempty then use this file in DataLoaderRaw (see docs there)
parser.add_argument('--image_folder', type=str, default='data/mscoco/test2014')#If this is nonempty then will predict on the images in this folder path
parser.add_argument('--image_root', type=str, default='') #In case the image paths have to be preprended with a root path to an image folder
# For evaluation on a folder of h5 file:
parser.add_argument('--input_json', type=str, default='data/mscoco/cocotalk_karpathy_0816.json')
parser.add_argument('--input_im_h5', type=str, default='data/mscoco/cocotalk_karpathy_images.h5')
parser.add_argument('--input_fc_h5', type=str, default='data/mscoco/cocotalk_karpathy_fc_0816.h5')
parser.add_argument('--input_att_h5', type=str, default='data/mscoco/cocotalk_karpathy_att_0816.h5')
parser.add_argument('--input_label_h5', type=str, default='data/mscoco/cocotalk_karpathy_label_0816.h5')
parser.add_argument('--input_attrs_h5', type=str, default='data/mscoco/cocotalk_karpathy_label_semantic_words_mil_det.h5')
parser.add_argument('--input_fc_dir', type=str, default='data/mscoco/cocotalk_karpathy_resnet101_fc')
parser.add_argument('--input_att_dir', type=str, default='data/mscoco/cocotalk_karpathy_resnet101_att')

# misc
parser.add_argument('--model_id', type=str, default='')#an id identifying this run/job. used in cross-val and appended when writing progress files
parser.add_argument('--id', type=str, default='evalscript')#an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files

args = parser.parse_args()
last_name = os.path.basename(args.start_from)
last_time = last_name[0:8]

# Load infos
def load_infos(args):
    model_name = os.path.basename(args.start_from)
    with open(os.path.join(args.start_from, 'infos.pkl')) as f:
        infos = cPickle.load(f)
    # override and collect parameters
    opt = infos['opt']

    for k in vars(args).keys():
        if k in vars(opt):
            vars(opt).update({k: vars(args)[k]})  # copy over options from model
        else:
            vars(opt)[k] = vars(args)[k]
    if opt.data_type ==1:
        opt.input_json = 'data/mscoco/cocotalk_karpathy_0816.json'
        opt.input_label_h5 = 'data/mscoco/cocotalk_karpathy_label_0816.h5'

    if opt.server:
        opt.dump_path = 0
        opt.dump_json = 1
        opt.disp_coarse_to_fine = 0
        opt.dump_output = 0
        opt.batch_size = 10
        if opt.split=='test':
            opt.coco_json = 'data/mscoco/annotations/image_info_test2014.json'
            opt.image_folder = 'data/mscoco/test2014'
            opt.val_images_use = 40775
        else:
            opt.coco_json = 'data/mscoco/annotations/captions_val2014.json'
            opt.image_folder = 'data/mscoco/val2014'
            opt.val_images_use = 40504
    else:
        opt.image_folder = ''

    model, cnn_model, crit = load_models(opt, infos)

    return cnn_model, model, crit, opt, infos

def eval(cnn_model, model, crit, opt, infos, args):
    visualization = False
    if visualization:
        eval_utils.visualize_embeddings(20, model.embed, infos['vocab'])

    # Create the Data Loader instance
    if len(opt.image_folder) == 0:
        loader = DataLoader_pool(opt) if args.pool_loader else DataLoader(opt)
        loader.ix_to_word = infos['vocab']
        loss, split_predictions, lang_stats = eval_utils.eval_split(opt, infos, cnn_model, model, crit, loader, vars(opt))
        print('loss: ', loss)
        if lang_stats:
            print(lang_stats)
    else:
        from misc.dataloader.dataloaderraw import *
        loader = DataLoaderRaw(opt)
        loader.ix_to_word = infos['vocab']
        split_predictions = eval_utils.eval_split(opt, infos, cnn_model, model, crit, loader, vars(opt))

    dump_json = getattr(args, 'dump_json', 0)
    if dump_json == 1:
        # dump the json
        if opt.server:
            json.dump(split_predictions, open('captions_' + opt.split + '2014_' + last_time + '_results.json', 'w'))
        else:
            json.dump(split_predictions, open('vis/vis.json', 'w'))


if __name__ == "__main__":
    cnn_model, model, crit, opt, infos = load_infos(args)
    eval(cnn_model, model, crit, opt, infos, args)