# coding: utf-8
import cv2
import os,sys
import time
import os.path
from fileutil import *
from alignment import cv2_imread
from matio import save_mat, load_mat
import json
import argparse
import numpy as np

def load_feature_list(rel_list, feature_dir, suffix):
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

def load_probset(templatelist, feature_dir, suffix):
    # load json  
    with open(templatelist, 'r') as f:
        data = json.load(f)
    probe_path = data['path']
    prob_feat = load_feature_list(probe_path, feature_dir, suffix)
    data['feat'] = prob_feat
    return data

def L2_distance(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))  
    
def test_set(distractor_path, probe, feature_dir, suffix):
    # load distractor  
    with open(distractor_path, 'r') as f:
        data = json.load(f)
    distractor_list = data['path']
    # probe set
    T = len(probe['id'])
    probe_set = set()
    for id in probe['id']:
        probe_set.add(id)
    ids = list(probe_set)
    N = len(ids)
    pclass = {}
    for id in ids:
        pclass[id] = []
    for i in range(len(probe['id'])):
        id = probe['id'][i]
        pclass[id].append(i)
    probe_id_list = probe['id']
    probe_feat_list = probe['feat']
    correct = 0
    wrong = 0
    # class 1,..., N
    for id in pclass:
        # i,...,M
        S = pclass[id]
        M = len(S)
        print('test id:%s' % id)
        for i in range(M):
            min_dist = 100000
            min_idx = 0
            probe_idx = S[i]
            probe_feat = probe_feat_list[probe_idx]
            probe_id = probe_id_list[probe_idx]
            # match distractors
            for j in range(len(distractor_list)):
                # make fullpath
                feat_path = os.path.join(feature_dir, distractor_list[j])
                feat_path = feat_path + suffix
                feat = load_mat(feat_path)
                dist = L2_distance(feat,probe_feat)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
            p2_dist = 100000
            p2_idx = 0
            # match probe
            '''
            for j in range(T):
                if j == probe_idx:
                    continue
                dist = L2_distance(probe_feat_list[j],probe_feat)
                if dist < p2_dist:
                    p2_dist = dist
                    p2_idx = j
            '''
            for c in ids:
                if c != id:
                    SS = pclass[c]
                    MM = len(SS)
                    II = np.random.randint(MM)
                    j = SS[II]
                else:
                    II = np.random.randint(M-1)
                    II = II if II < i else (II+1)
                    j = S[II]
                dist = L2_distance(probe_feat_list[j],probe_feat)
                if dist < p2_dist:
                    p2_dist = dist
                    p2_idx = j
                    
            if p2_dist < min_dist:
                if probe['id'][p2_idx] == probe_id:
                    print("TTT:%s\t %s %f" % (probe['path'][probe_idx], probe['path'][p2_idx], p2_dist))
                else:
                    print("FFF:%s\t %s %f" % (probe['path'][probe_idx], probe['path'][p2_idx], p2_dist))
            else:
                print("FFF:%s\t %s %f" % (probe['path'][probe_idx], distractor_list[min_idx], min_dist))
            
            # compare
            if p2_dist < min_dist and probe_id_list[p2_idx] == probe_id:
                correct += 1
            else:
                wrong += 1
    rank1_ratio = float(correct)/(correct+wrong)
    print('RandK-1:%f %d/%d' % (rank1_ratio, correct, (correct+wrong)))           
        
if __name__=='__main__': 
    # config_file model_name
    if len(sys.argv) < 3:
        print('config_file distractors')
        exit(0)
    # read config
    config = Config(sys.argv[1])
    distractors = sys.argv[2]
    # config
    model_name = config.get('model').name
    # load model
    model = config.get(model_name).model
    weights = config.get(model_name).weights
    suffix = config.get(model_name).suffix
    extractor = load_extractor(model, weights, config.get('model').gpu_id)

    # devkit/templatelists
    templatelists_dir = config.get('devkit').templatelists_dir
    
    facescrub_feature_dir = config.get('facescrub').feature_dir
    mega_feature_dir = config.get('megaface').feature_dir
    
    # devkit/templatelists
    probe = load_probset(templatelists_dir+'/facescrub_uncropped_features_list.json',
      facescrub_feature_dir, suffix)
    list_name = 'megaface_features_list.json_%s_1' % distractors
    test_set(templatelists_dir+'/' + list_name, probe, mega_feature_dir, suffix)
    