# -*- coding:utf-8 -*-
from __future__ import print_function
import math
import numpy as np
import logging
import time
import os
import pickle
import argparse
import cv2

from verification import verification
from plot import draw_chart

import sys
sys.path.insert(0, '../facealign')
sys.path.insert(0, '../util') 
  
from caffe_extractor import CaffeExtractor
from distance import get_distance

def parse_line(line):
    splits = line.split()
    # skip line
    if len(splits) < 3:
        return None
    # name id1 id2
    if len(splits) == 3:
        return True, splits[0], splits[1], splits[0], splits[2]
    # name1 id1 name2 id2
    return False, splits[0], splits[1], splits[2], splits[3]
    
    
def load_image_paris(pair_path, prefix):
    pair_list = []
    # parse pairs
    with open(pair_path, 'r') as f:
        for line in f.readlines():
            pair = parse_line(line)
            if pair is not None:
                pair_list.append(pair)
                # print(pair)
    #print('#pairs:%d' % len(pair_list))
    # compute feature
    pos_img = []
    neg_img = []
    count = 0
    for pair in pair_list:
        count += 1
        
        img_path1 = '%s/%s/%s_%04d.jpg' % (prefix, pair[1], pair[1], int(pair[2]))
        img_path2 = '%s/%s/%s_%04d.jpg' % (prefix, pair[3], pair[3], int(pair[4]))
        rel_path1 = '%s/%s_%04d.jpg' % (pair[1], pair[1], int(pair[2]))
        rel_path2 = '%s/%s_%04d.jpg' % (pair[3], pair[3], int(pair[4]))
        # skip invalid pairs
        if not os.path.exists(img_path1) or not os.path.exists(img_path2):
            continue
        img1 = cv2.imread(img_path1)
        #img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.imread(img_path2)
        #img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        #print(img1.shape)
        if pair[0]:
            pos_img.append([img1, img2, rel_path1, rel_path2])
        else:
            neg_img.append([img1, img2, rel_path1, rel_path2])
    return pos_img, neg_img
        
        
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
    h, w, c = img_list[0][0].shape
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
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = False, featLayer='fc5')
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

def model_tzk(do_mirror):
    model_dir = '/home/ysten/tzk/fr/insightface/caffe/AMSoftmax/'
    model_proto = model_dir + 'deploy.prototxt'
    model_path = model_dir + 'tune_iter_20000.caffemodel'
    image_size = (112, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc5')
    return extractor, image_size

    
def model_factory(name, do_mirror):
    model_dict = {
        'centerface':model_centerface, 
        'sphereface':model_sphereface, 
        'AMSoftmax' :model_AMSoftmax, 
        'arcface'   :model_arcface,
        'yours'     :model_yours, 
        'tzk'       :model_tzk, 
    }
    model_func = model_dict[name]
    return model_func(do_mirror)

def check_lfw_data():
    pos_1, _ = load_image_paris('./pairs.txt', '/home/ysten/denghui/fr/LFW_TEST/data/lfw-112X96')
    pos_2, _ = load_image_paris('./pairs.txt', 'lfw-96x112')
    count = 0
    for idx in range(len(pos_1)):
        diff = np.float32(pos_1[idx][0]) - np.float32(pos_2[idx][0])
        diff = np.abs(diff)
        abs_sum = np.sum(diff) / (96*112*3)
        if abs_sum > 20:
            count += 1
            print('%4d %6.3f %s -- %s' % (idx, abs_sum, pos_1[idx][2], pos_2[idx][2]) )
    print('%d'%count)   
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lfw_data", help=" lfw.np or lfw-112x112")
    parser.add_argument("--model_name", help= 'specify which model to test \n'
                                              ' centerface\n'
                                              ' sphereface\n'
                                              ' AMSoftmax\n'
                                              ' arcface\n'
                                              ' yours \n')
    parser.add_argument("--dist_type", default='cosine', help="distance measure ['cosine', 'L2', 'SSD']")
    parser.add_argument("--do_mirror", default=False, help="mirror image and concatinate features")

    args = parser.parse_args()
    output_dir = '.'
    # parse args   
    model_name = args.model_name
    lfw_data = args.lfw_data
    dist_type = args.dist_type
    do_mirror = args.do_mirror
    
    print('Testing  \t: %s' % model_name)
    print('Distance \t: %s' % dist_type)
    print('Do mirror\t: {}'.format(do_mirror))
    # model
    extractor, image_size = model_factory(model_name, do_mirror)
    print('Testing model\t: %s' % (extractor.weight))
    print('Image size\t: {}'.format(image_size))
    # load images
    if lfw_data.find('.np') > 0:
        pos_img, neg_img = pickle.load(open(lfw_data, 'rb'))
        #pos_img, neg_img = pickle.load(open(lfw_data, 'rb'), encoding='iso-8859-1')
        
    else:
        pos_img, neg_img = load_image_paris('./pairs.txt', lfw_data)
        
    # crop image
    pos_img = crop_image_list(pos_img, image_size)
    neg_img = crop_image_list(neg_img, image_size)
    #print(type(pos_img[0][0]))
    #exit()  
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
   
   

