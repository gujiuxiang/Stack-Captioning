import os
import copy

import numpy as np
import misc.utils as utils
import torch
import misc.cnn.resnet as resnet
from .ShowTellModel import ShowTellModel
from .FCModel import FCModel
from .CaptionModel import ShowAttendTellModel, AllImgModel
from .Att2inModel import Att2inModel
from .AttModel import *
from .LanguageModel import *

def setup(opt):
    
    if opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    elif opt.caption_model == 'show_attend_tell':
        model = ShowAttendTellModel(opt)
    # img is concatenated with word embedding at every time step as the input of lstm
    elif opt.caption_model == 'all_img':
        model = AllImgModel(opt)
    # FC model in self-critical
    elif opt.caption_model == 'fc':
        model = FCModel(opt)
    # Att2in model in self-critical
    elif opt.caption_model == 'att2in':
        model = Att2inModel(opt)
    # Att2in model with two-layer MLP img embedding and word embedding
    elif opt.caption_model == 'att2in2':
        model = Att2in2Model(opt)
    # Adaptive Attention model from Knowing when to look
    elif opt.caption_model == 'adaatt':
        model = AdaAttModel(opt)
    # Adaptive Attention with maxout lstm
    elif opt.caption_model == 'adaattmo':
        model = AdaAttMOModel(opt)
    # Top-down attention model
    elif opt.caption_model == 'topdown':
        model = TopDownModel(opt)
    elif opt.caption_model == 'debug':
        # I only privde the testing code now
        model = LanguageModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos.pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    return model

def build_cnn(opt):
    opt.pre_ft = getattr(opt, 'pre_ft', 1)
    if opt.pre_ft == 0:
        if opt.cnn_model == 'resnet101':
            net = getattr(resnet, opt.cnn_model)()
            # if vars(opt).get('start_from', None) is None and vars(opt).get('cnn_weight', '') != '':
            if len(opt.start_from) == 0 and len(opt.cnn_weight) != 0:
                net.load_state_dict(torch.load(opt.cnn_weight))
            net = nn.Sequential( \
                net.conv1,
                net.bn1,
                net.relu,
                net.maxpool,
                net.layer1,
                net.layer2,
                net.layer3,
                net.layer4)
            if len(opt.start_from) != 0:
                net.load_state_dict(torch.load(os.path.join(opt.start_from, opt.model_id + '.model-cnn-best.pth')))
        elif opt.cnn_model == 'resnet152':
            net = getattr(resnet, opt.cnn_model)()
            # if vars(opt).get('start_from', None) is None and vars(opt).get('cnn_weight', '') != '':
            if len(opt.start_from) == 0 and len(opt.cnn_weight) != 0:
                print(opt.start_from)
                net.load_state_dict(torch.load(opt.cnn_weight))
            net = nn.Sequential( \
                net.conv1,
                net.bn1,
                net.relu,
                net.maxpool,
                net.layer1,
                net.layer2,
                net.layer3,
                net.layer4)
            if len(opt.start_from) != 0:
                net.load_state_dict(torch.load(os.path.join(opt.start_from, opt.model_id + '.model-cnn-best.pth')))
        elif opt.cnn_model == 'vgg16':
            net = getattr(models, opt.cnn_model)()

            if len(opt.start_from) == 0 and len(opt.cnn_weight) != 0:
                net.load_state_dict(torch.load(opt.cnn_weight))
                print("Load pretrained CNN model from " + opt.cnn_weight)

            new_classifier = nn.Sequential(*list(net.classifier.children())[:6])
            net.classifier = new_classifier

            #if vars(opt).get('start_from', None) is None and vars(opt).get('cnn_weight', '') != '':
            if len(opt.start_from) != 0:
                print("Load pretrained CNN model (from start folder) : " + opt.start_from)
                net.load_state_dict(torch.load(os.path.join(opt.start_from, opt.model_id + '.model-cnn-best.pth')))
        net.cuda()
    else:
        net = None

    return net

def load_models(opt, infos):
    opt.model_name = getattr(opt, 'model_name', 'topdown')
    cnn_model = build_cnn(opt)
    model = setup(opt)
    crit = None
    return model.cuda(), cnn_model, crit