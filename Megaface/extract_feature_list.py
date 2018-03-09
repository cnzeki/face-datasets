# coding: utf-8
import cv2
import os,sys
import time
import os.path

sys.path.insert(0, '../facealign')
sys.path.insert(0, '../util') 

from fileutil import *
from caffe_extractor import CaffeExtractor
from alignment import cv2_imread
from matio import save_mat, load_mat
from config import Config
import json
import argparse


def load_extractor( model, weights, gpu_id = 0 ):
    # init extractor
    extractor = CaffeExtractor(model, weights, gpu_id)
    return extractor
    

def extract_feature_list(rel_list, image_dir, feature_dir, suffix, extractor):
    total_size = len(rel_list)  
    for i in range(total_size):
        # make fullpath
        path = os.path.join(image_dir, rel_list[i])
        img = cv2_imread(path)
        #print(path)
        #print(img.shape)
        feat = extractor.extract_feature(img)
        feat_path = translate_path(image_dir, feature_dir, path)
        feat_path = feat_path + suffix
        makedirs(feat_path)
        save_mat(feat_path, feat)
        if i % 100 == 0:
            print('%6d/%6d %s' % (i, total_size, rel_list[i]))
            

def load_feature_list(rel_list, image_dir, feature_dir, suffix):
    total_size = len(rel_list) 
    featlist = []
    for i in range(total_size):
        # make fullpath
        feat_path = os.path.join(feature_dir, rel_list[i])
        feat_path = feat_path + suffix
        feat = load_mat(feat_path)
        featlist.append(feat)
        if i % 100 == 0:
            print('\t%d/%d\t%s' % (i, total_size, rel_list[i]))
    return featlist

            
def extract_facescrub_uncropped(facescrub_dir, templatelists_dir, feature_dir, suffix, extractor):
    # load json  
    json_path = os.path.join(templatelists_dir, 'facescrub_uncropped_features_list.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    rel_list = data['path']

    extract_feature_list(rel_list, facescrub_dir, feature_dir, suffix, extractor)
    

def extract_megaface_features_list(mega_dir, templatelist, feature_dir, suffix, extractor):
    # load json  
    with open(templatelist, 'r') as f:
        data = json.load(f)
    rel_list = data['path']

    extract_feature_list(rel_list, mega_dir, feature_dir, suffix, extractor)
    
    
def extract_megaface_features(mega_dir, templatelists_dir, feature_dir, suffix, extractor):
    for i in range(1,6):
        set_size = 10**i
        json_path = 'megaface_features_list.json_%d_1' % (set_size)
        json_path = os.path.join(templatelists_dir,json_path)
        print(json_path)
        extract_megaface_features_list(mega_dir, json_path, feature_dir, suffix, extractor)
        
        
if __name__=='__main__': 
    # config_file model_name
    if len(sys.argv) < 2:
        print('config_file [model_name]')
        exit(0)
    # read config
    config = Config(sys.argv[1])
    # config
    model_name = config.get('model').name
    if len(sys.argv) >= 3:
        model_name = sys.argv[2]
        
    # load model
    model = config.get(model_name).model
    weights = config.get(model_name).weights
    suffix = config.get(model_name).suffix
    extractor = load_extractor(model, weights, config.get('model').gpu_id)

    # devkit/templatelists
    templatelists_dir = config.get('devkit').templatelists_dir
    
    # Extract facescrub features
    facescrub_dir = config.get('facescrub').aligned_dir
    facescrub_feature_dir = config.get('facescrub').feature_dir
    extract_facescrub_uncropped(facescrub_dir, templatelists_dir, facescrub_feature_dir, suffix,  extractor)
    
    # Extract MegaFace features
    mega_dir = config.get('megaface').aligned_dir
    mega_feature_dir = config.get('megaface').feature_dir
    extract_megaface_features(mega_dir, templatelists_dir, mega_feature_dir, suffix, extractor)