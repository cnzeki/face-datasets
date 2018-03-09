# coding: utf-8

import cv2
import os,sys
import time
import os.path
sys.path.insert(0, '../facealign')
sys.path.insert(0, '../util') 

from fileutil import *
import json
import shutil
 
if __name__=='__main__':
    image_dir = '/home/ysten/tzk/fr/MegaFace/facescrub'
    target_dir = '/home/ysten/tzk/fr/MegaFace/facescrub_80'
    # load json  
    json_path = os.path.join('/home/ysten/tzk/fr/MegaFace/devkit/templatelists', 'facescrub_uncropped_features_list.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    rel_list = data['path']
    # 
    total_size = len(rel_list)  
    for i in range(total_size):
        # make fullpath
        path = os.path.join(image_dir, rel_list[i])
        #print(path)
        #print(img.shape)
        feat_path = translate_path(image_dir, target_dir, path)
        makedirs(feat_path)
        shutil.copyfile(path, feat_path)  
        # copy
        if i % 100 == 0:
            print('%d/%d %s' % (i, total_size, path))