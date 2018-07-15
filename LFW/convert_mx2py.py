# -*- coding:utf-8 -*- 
import pickle
import sys
import cv2
import numpy as np
import mxnet as mx

binf = open('./lfw.bin', 'rb')
bins, issame_list = pickle.load(binf)
#binf = open('lfw.bin', 'rb')
#bins, issame_list = pickle.load(binf, encoding='iso-8859-1')
npairs = len(issame_list)
pos_list = []
neg_list = []

for i in range(npairs):
    img1 = mx.image.imdecode(bins[i*2])
    npimg1 = img1.asnumpy()
    img2 = mx.image.imdecode(bins[i*2+1])
    npimg2 = img2.asnumpy()
    if issame_list[i]:
        pos_list.append([npimg1, npimg2])
    else:
        neg_list.append([npimg1, npimg2])

with open('lfw.np', 'wb') as f:
    pickle.dump((pos_list, neg_list), f, protocol=pickle.HIGHEST_PROTOCOL)    

