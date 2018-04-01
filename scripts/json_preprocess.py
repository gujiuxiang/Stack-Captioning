#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: github.com/duinodu

from __future__ import print_function
import os
import argparse
import json
from PIL import Image
import jieba

def convert2coco(caption_json, img_dir):
    dataset = json.load(open(caption_json, 'r'))
    imgdir = img_dir

    coco = dict()
    coco[u'info'] = { u'desciption':u'AI challenger image caption in mscoco format'}
    coco[u'licenses'] = ['Unknown', 'Unknown']
    coco[u'images'] = list()
    coco[u'annotations'] = list()

    for ind, sample in enumerate(dataset):
        img = Image.open(os.path.join(imgdir, sample['image_id']))
        width, height = img.size

        coco_img = {}
        coco_img[u'license'] = 0
        coco_img[u'file_name'] = os.path.split(img_dir)[-1]+'/'+sample['image_id']
        coco_img[u'width'] = width
        coco_img[u'height'] = height
        coco_img[u'date_captured'] = 0
        coco_img[u'coco_url'] = sample['url']
        coco_img[u'flickr_url'] = sample['url']
        coco_img['id'] = os.path.splitext(os.path.basename(sample['image_id']))[0]

        coco_anno = {}
        coco_anno[u'image_id'] = os.path.splitext(os.path.basename(sample['image_id']))[0]
        coco_anno[u'id'] = os.path.splitext(os.path.basename(sample['image_id']))[0]
        coco_anno[u'caption'] = sample['caption']

        coco[u'images'].append(coco_img)
        coco[u'annotations'].append(coco_anno)

        print('{}/{}'.format(ind, len(dataset)))

    output_file = os.path.join(os.path.dirname(caption_json), 'coco_'+os.path.basename(caption_json))
    with open(output_file, 'w') as fid:
        json.dump(coco, fid)
    print('Saved to {}'.format(output_file))

def convert2coco_val(caption_json, img_dir):
    dataset = json.load(open(caption_json, 'r'))
    imgdir = img_dir

    coco = dict()
    coco[u'info'] = { u'desciption':u'AI challenger image caption in mscoco format'}
    coco[u'licenses'] = ['Unknown', 'Unknown']
    coco[u'images'] = list()
    coco[u'annotations'] = list()

    for ind, sample in enumerate(dataset):
        img = Image.open(os.path.join(imgdir, sample['image_id']))
        width, height = img.size

        coco_img = {}
        coco_img[u'license'] = 0
        coco_img[u'file_name'] = os.path.split(img_dir)[-1]+'/'+sample['image_id']
        coco_img[u'width'] = width
        coco_img[u'height'] = height
        coco_img[u'date_captured'] = 0
        coco_img[u'coco_url'] = sample['url']
        coco_img[u'flickr_url'] = sample['url']
        coco_img['id'] = os.path.splitext(os.path.basename(sample['image_id']))[0]

        coco_anno = {}
        coco_anno[u'image_id'] = os.path.splitext(os.path.basename(sample['image_id']))[0]
        coco_anno[u'id'] = os.path.splitext(os.path.basename(sample['image_id']))[0]
        coco_anno[u'caption'] = sample['caption']
        idx = 0
        for s in sample['caption']:
            if len(s)==0:
                print('error: some caption had no words?')
                print(coco_img[u'file_name'])
                sample['caption'][idx] = sample['caption'][idx-1]
                print(sample['caption'])
                #break
            idx = idx+1
        coco[u'images'].append(coco_img)
        coco[u'annotations'].append(coco_anno)

        #print('{}/{}'.format(ind, len(dataset)))

    output_file = os.path.join(os.path.dirname(caption_json), 'coco_'+os.path.basename(caption_json))
    with open(output_file, 'w') as fid:
        json.dump(coco, fid)
    print('Saved to {}'.format(output_file))

def convert2coco_eval(caption_json, img_dir):
    dataset = json.load(open(caption_json, 'r'))
    imgdir = img_dir

    coco = dict()
    coco[u'info'] = { u'desciption':u'AI challenger image caption in mscoco format'}
    coco[u'licenses'] = ['Unknown', 'Unknown']
    coco[u'images'] = list()
    coco[u'annotations'] = list()
    coco[u'type'] = u'captions'
    for ind, sample in enumerate(dataset):
        #img = Image.open(os.path.join(imgdir, sample['image_id']))
        #width, height = img.size
        width, height = 224, 224

        coco_img = {}
        coco_img[u'license'] = 0
        coco_img[u'file_name'] = os.path.split(img_dir)[-1]+'/'+sample['image_id']
        coco_img[u'width'] = width
        coco_img[u'height'] = height
        coco_img[u'date_captured'] = 0
        coco_img[u'coco_url'] = sample['url']
        coco_img[u'flickr_url'] = sample['url']
        coco_img['id'] = os.path.splitext(os.path.basename(sample['image_id']))[0]

        coco_anno = {}
        coco_anno[u'image_id'] = os.path.splitext(os.path.basename(sample['image_id']))[0]
        coco_anno[u'id'] = os.path.splitext(os.path.basename(sample['image_id']))[0]
        coco_anno[u'caption'] = sample['caption']

        coco[u'images'].append(coco_img)
        for coco_anno_ in coco_anno['caption']:
            coco_anno_s = {}
            coco_anno_s[u'image_id'] = coco_anno[u'image_id']
            coco_anno_s[u'id'] = coco_anno[u'id']
            w = jieba.cut(coco_anno_.strip(), cut_all=False)
            p = ' '.join(w)
            coco_anno_ = p
            coco_anno_s[u'caption'] = coco_anno_
            coco[u'annotations'].append(coco_anno_s)

        print('{}/{}'.format(ind, len(dataset)))

    output_file = os.path.join(os.path.dirname(caption_json), 'coco_val_'+os.path.basename(caption_json))
    with open(output_file, 'w') as fid:
        json.dump(coco, fid)
    print('Saved to {}'.format(output_file))

def convert2coco_test(img_dir):
    coco = dict()
    coco[u'info'] = { u'desciption':u'AI challenger image caption in mscoco format'}
    coco[u'licenses'] = ['Unknown', 'Unknown']
    coco[u'images'] = list()
    ind = 0
    for im_name in enumerate(os.listdir(img_dir)):
        width, height = 224, 224

        coco_img = {}
        coco_img[u'license'] = 0
        coco_img[u'file_name'] = im_name[1]
        coco_img[u'width'] = width
        coco_img[u'height'] = height
        coco_img[u'date_captured'] = 0
        coco_img[u'id'] = os.path.splitext(os.path.basename(im_name[1]))[0]
        ind = ind + 1
        coco[u'images'].append(coco_img)

        print('{}/{}'.format(ind, len(os.listdir(img_dir))))

    output_file = os.path.join('data/ai_challenger', 'ai_challenger_test1.json')
    with open(output_file, 'w') as fid:
        json.dump(coco, fid)
    print('Saved to {}'.format(output_file))

def ai_challenger_preprocess():
    import os
    import json
    val = json.load(open('data/ai_challenger/ai_challenger_caption_validation_20170910/coco_caption_validation_annotations_20170910.json', 'r'))
    train = json.load(open('data/ai_challenger/ai_challenger_caption_train_20170902/coco_caption_train_annotations_20170902.json', 'r'))

    print(val.keys())
    print(val['info'])
    print(len(val['images']))
    print(len(val['annotations']))
    print(val['images'][0])
    print(val['annotations'][0])

    import json
    import os

    # combine all images and annotations together
    imgs = train['images']+val['images']
    annots = train['annotations']+val['annotations']

    # for efficiency lets group annotations by image
    itoa = {}
    for a in annots:
        imgid = a['image_id']
        if not imgid in itoa: itoa[imgid] = []
        itoa[imgid].append(a)

    # create the json blob
    out_json={}
    out=[]
    for i, img in enumerate(imgs):
        out_im = {}
        # coco specific here, they store train/val images separately
        split = 'train' if 'train' in img['file_name'] else 'val'
        annotsi = itoa[img['id']]
        sentid = 0
        out_im['cocoid'] = img['id']
        out_im['filename'] = os.path.basename(img['file_name'])
        if 'val' in img['file_name']:
            out_im['filepath'] = 'ai_challenger_caption_validation_20170910/caption_validation_images_20170910'
        else:
            out_im['filepath'] = 'ai_challenger_caption_train_20170902/caption_train_images_20170902'
        out_s = []
        for a in annotsi:
            for s in a['caption']:
                jimg = {}
                jimg['imgid'] = img['id']
                jimg['raw'] = s
                jimg['sentid'] = img['id']+'_'+str(sentid)
                txt = []
                for sentence in s:
                    txt.append("".join(jieba.cut(sentence)))
                jimg['tokens'] = txt
                jimg['sentids'] = []
                out_s.append(jimg)
        out_im['sentences']=out_s
        out_im['split'] = split
        out.append(out_im)
    out_json['images']=out
    out_json['dataset']='ai_challenger'
    output_file = os.path.join('data/ai_challenger', 'coco_ai_challenger.json')
    json.dump(out_json, open(output_file, 'w'))

if __name__ == "__main__":
    train_caption_json = '/media/jxgu/d2ta/dataset/ai_challenger/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json'
    train_img_dir = '/media/jxgu/d2ta/dataset/ai_challenger/ai_challenger_caption_train_20170902/caption_train_images_20170902'
    val_caption_json = '/media/jxgu/d2ta/dataset/ai_challenger/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json'
    val_img_dir = '/media/jxgu/d2ta/dataset/ai_challenger/ai_challenger_caption_validation_20170910/caption_validation_images_20170910'
    test_img_dir = '/media/jxgu/d2ta/dataset/ai_challenger/ai_challenger_caption_test1_20170923/caption_test1_images_20170923'
    # Convert json (ai challenger) to coco format
    convert2coco(train_caption_json, train_img_dir)
    convert2coco_val(val_caption_json, val_img_dir)
    # Create json file for testing
    convert2coco_eval(test_img_dir)
    # Create json file for evaluation
    convert2coco_eval(val_caption_json, val_img_dir)
    # Create json file for sentence label and image feature extraction
    ai_challenger_preprocess()
