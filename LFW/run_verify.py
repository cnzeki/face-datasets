# -*- coding:utf-8 -*-
from __future__ import print_function
import math
import numpy as np
import logging
import time
import os
import pickle

from verification import verification
from plot import draw_chart

import sys
sys.path.insert(0, '../facealign')
sys.path.insert(0, '../util') 
  
from caffe_extractor import CaffeExtractor
from distance import get_distance


def extract_feature(extractor, img_list):
    feat_list = []
    n = len(img_list)
    idx = 1
    for pair in img_list:
        img1 = pair[0]
        img2 = pair[1]
        feat1 = extractor.extract_feature(img1)
        feat2 = extractor.extract_feature(img2)
        feat_list.append([feat1, feat2])
        if idx > 1:
            print('{}'.format('\b'*10))
        print('{}/{}'.format(idx, n), end='')
        idx += 1
    return feat_list

def crop_image_list(img_list, imsize):
    out_list = []
    w, h = 112, 112
    x1 = (w - imsize[0])/2
    y1 = (h - imsize[1])/2
    for pair in img_list:
        img1 = pair[0]
        img2 = pair[1]
        img1 = img1[y1:(y1+imsize[1]),x1:(x1+imsize[0]),:]
        img2 = img2[y1:(y1+imsize[1]),x1:(x1+imsize[0]),:]
        out_list.append([img1, img2])
    #print(img1.shape)
    return out_list
    
    
def load_bin(path, image_size):
  import mxnet as mx
  bins, issame_list = pickle.load(open(path, 'rb'))
  pos_img = []
  neg_img = []
  for i in xrange(len(issame_list)):
    _bin = bins[i*2]
    img1 = mx.image.imdecode(_bin)
    _bin = bins[i*2+1]
    img2 = mx.image.imdecode(_bin)
    if issame_list[i]:
      pos_img.append([img1, img2])
    else:
      neg_img.append([img2, img2])
  return pos_img, neg_img


def model_centerface(do_mirror):
    model_dir = './models/centerface/'
    model_proto = model_dir + 'face_deploy.prototxt'
    model_path = model_dir + 'face_model.caffemodel'
    image_size = (96, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc5')
    return extractor, image_size
    
def model_sphereface(do_mirror):
    model_dir = './models/sphereface/'
    model_proto = model_dir + 'sphereface_deploy.prototxt'
    model_path = model_dir + 'sphereface_model.caffemodel'
    image_size = (96, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc5')
    return extractor, image_size
    
def model_AMSoftmax(do_mirror):
    model_dir = './models/AMSoftmax/'
    if do_mirror:
        model_proto = model_dir + 'face_deploy_mirror_normalize.prototxt'
    else:
        model_proto = model_dir + 'deploy.prototxt'
    model_path = model_dir + 'face_train_test_iter_30000.caffemodel'
    image_size = (96, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = False, featLayer='norm1')
    return extractor, image_size
    
    
def model_arcface(do_mirror):
    model_dir = './models/arcface/'
    model_proto = model_dir + 'model.prototxt'
    model_path = model_dir + 'model-r50-am.caffemodel'
    image_size = (112, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc1')
    return extractor, image_size
    
    
    
def model_yours(do_mirror):
    model_dir = '/path/to/your/model/'
    model_proto = model_dir + 'deploy.prototxt'
    model_path = model_dir + 'weights.caffemodel'
    image_size = (112, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc5')
    return extractor, image_size

        
def model_factory(name, do_mirror):
    model_dict = {
        'centerface':model_centerface, 
        'sphereface':model_sphereface, 
        'AMSoftmax':model_AMSoftmax, 
        'arcface':model_arcface,
        'model_yours':model_yours, 
    }
    model_func = model_dict[name]
    return model_func(do_mirror)
    
if __name__ == '__main__':
    model_name = ''
    lfw_path = './lfw.np'
    output_dir = '.'
    dist_type = 'cosine'
    do_mirror = False
    # parse args
    if len(sys.argv) < 2:
        print('run_verify.py modelname [dist_type(`cosine` | `L2`)] [do_mirror(`0`| `1`)]'
              'specify which model to test \n'
              ' centerface\n'
              ' sphereface\n'
              ' AMSoftmax\n'
              ' arcface\n'
              ' yours \n'
              )
        exit()
    # get model name
    model_name = sys.argv[1]
    # dist type
    if len(sys.argv) > 2:
        dist_type = sys.argv[2]
    # do mirror
    if len(sys.argv) > 3 and sys.argv[3] == '1':
        do_mirror = True
        
    print('Testing  \t: %s' % model_name)
    print('Distance \t: %s' % dist_type)
    print('Do mirror\t: {}'.format(do_mirror))
    # model
    extractor, image_size = model_factory(model_name, do_mirror)
    print('Testing model\t: %s' % (extractor.weight))
    print('Image size\t: {}'.format(image_size))
    # extract feature
    pos_img, neg_img = pickle.load(open(lfw_path, 'rb'))
    print('Lfw pairs\t: {}'.format(len(pos_img)))
    # crop image
    pos_img = crop_image_list(pos_img, image_size)
    neg_img = crop_image_list(neg_img, image_size)
    #pos_img, neg_img = pickle.load(open(lfw_path, 'rb'), encoding='iso-8859-1')
    # compute feature
    print('Extracting features ...')
    pos_list = extract_feature(extractor, pos_img)
    print('  Done positive pairs')
    neg_list = extract_feature(extractor, neg_img)
    print('  Done negative pairs')
    print('Extracting features ...')
    # evaluate
    print('Evaluating ...')
    precision, std, threshold, pos, neg = verification(pos_list, neg_list, dist_type = dist_type)    
    _, title = os.path.split(extractor.weight)
    #draw_chart(title, output_dir, {'pos': pos, 'neg': neg}, precision, threshold)
    print('------------------------------------------------------------')
    print('Precision on LFW : %1.5f+-%1.5f \nBest threshold   : %f' % (precision, std, threshold))
   
   

