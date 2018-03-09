# -*- coding:utf-8 -*-
import os,sys
import math

sys.path.insert(0, '../util')
  
from caffe_extractor import CaffeExtractor
from distance import get_distance

import numpy as np
import argparse


LFW_PAIRS = None
LFW_IMG_DIR = None

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


def find_threshold_sort(pos_list,neg_list):
    pos_list = sorted(pos_list, key=lambda x: x[0])
    neg_list = sorted(neg_list, key=lambda x: x[0], reverse=True)
    pos_count = len(pos_list)
    neg_count = len(neg_list)
    correct = 0
    threshold = 0
    for i in range(min(pos_count, neg_count)):
        if pos_list[i][0] > neg_list[i][0]:
            correct = i
            threshold = (pos_list[i][0] + neg_list[i][0])/2
            break
    precision = (correct * 2.0) / (pos_count + neg_count)
    return precision, threshold


def get_accuracy(pos_list,neg_list,threshold):
    pos_count = len(pos_list)
    neg_count = len(neg_list)
    correct = 0
    for i in range(pos_count):
        if pos_list[i][0] < threshold:
            correct += 1

    for i in range(neg_count):
        if neg_list[i][0] > threshold:
            correct += 1
    precision = float(correct) / (pos_count + neg_count)
    return precision


def best_threshold(pos_list, neg_list, thrNum = 10000):
    ts = np.linspace(-1, 1, thrNum*2+1)
    best_acc = 0
    best_t = 0
    for t in ts:
        acc = get_accuracy(pos_list, neg_list, t)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_acc, best_t


def test_kfold(pos_list, neg_list, k = 10):
    fold_size = len(pos_list)/k
    sum_acc = 0
    sum_thresh = 0
    sum_n = 0
    for i in range(k):
        val_pos = []
        val_neg = []
        test_pos = []
        test_neg = []
        for j in range(len(pos_list)):
            fi = j/fold_size
            if fi != i:
                val_pos.append(pos_list[j])
                val_neg.append(neg_list[j])
            else:
                test_pos.append(pos_list[j])
                test_neg.append(neg_list[j])
        precision, threshold = find_threshold_sort(val_pos, val_neg)
        accuracy = get_accuracy(test_pos, test_neg, threshold)
        sum_acc += accuracy
        sum_thresh += threshold
        sum_n += 1
        # verbose
        print('precision:%.4f threshold:%f' % (accuracy, threshold))
    return sum_acc/sum_n, sum_thresh/sum_n


def parse_pair_file(pair_path, prefix, feat_extractor,dist_func):
    pair_list = []
    # parse pairs
    with open(pair_path, 'r') as f:
        for line in f.readlines():
            pair = parse_line(line)
            if pair is not None:
                pair_list.append(pair)
                # print(pair)
    print('#pairs:%d' % len(pair_list))
    # compute feature
    pos_list = []
    neg_list = []
    count = 0
    features = []
    for pair in pair_list:
        count += 1
        img_path1 = '%s/%s/%s_%04d.jpg' % (prefix, pair[1], pair[1], int(pair[2]))
        img_path2 = '%s/%s/%s_%04d.jpg' % (prefix, pair[3], pair[3], int(pair[4]))
        # skip invalid pairs
        if not os.path.exists(img_path1) or not os.path.exists(img_path2):
            continue
        feat1 = feat_extractor.extract(img_path1)
        feat2 = feat_extractor.extract(img_path2)
        dist = dist_func(feat1, feat2)
        if count % 100 == 1:
            print('%4d dist:%.4f %s |1|:%.4f |2|:%.4f' % (count, dist, pair[0],
              np.sqrt(np.sum(np.square(feat1))),  np.sqrt(np.sum(np.square(feat2)))))
            
        if pair[0]:
            pos_list.append([dist, feat1, feat2, img_path1, img_path2])
        else:
            neg_list.append([dist, feat1, feat2, img_path1, img_path2])
        features.append(feat1)
        features.append(feat2)

    # find best threshold()
    #precision, threshold = best_threshold(pos_list, neg_list, 10000)
    #return precision, threshold, pos_list, neg_list
    precision, threshold = test_kfold(pos_list, neg_list)
    return precision, threshold, pos_list, neg_list

def test_loss(extractor, weight, dist_type):
    dist_func = get_distance(dist_type)
    global LFW_PAIRS
    global LFW_IMG_DIR
    dir, path = os.path.split(weight)
    fnames = os.listdir(dir)
    fpattern = '%s.%s' % (path,dist_type)
    existed = False
    for fname in fnames:
        if fname.startswith(fpattern):
            existed = True
            print('skip:%s ' % (weight))
            return 
            
    print('test:%s ' % (weight))
    # test
    precision, threshold, pos_list, neg_list = parse_pair_file(LFW_PAIRS, LFW_IMG_DIR, extractor,dist_func)
    print('precision on lfw:%.4f threshold:%f ' % (precision, threshold))
    filename = '.%s.%.2f.txt' % (dist_type, precision*100)
    # write result
    with open(weight+filename,'w') as f:
        f.write( 'precision on lfw:%.4f threshold:%f ' % (precision, threshold) )

        
def test_model(model, weight,dist_type='cosine',do_mirror=False):
    extractor = CaffeExtractor(model, weight,do_mirror=do_mirror )
    test_loss(extractor, weight, dist_type)
    #test_loss(extractor, weight, 'SSD')

def test_dir(model_dir,dist_type='cosine',do_mirror=False): 
    filenames = os.listdir(model_dir)
    for filename in filenames:
        ext = os.path.splitext(filename)[1]
        if ext != '.caffemodel':
            continue
        # test the model
        test_model(model_dir+'/deploy.prototxt',filename,dist_type,do_mirror)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lfw_dir", help="lfw image dir")
    parser.add_argument("--lfw_pair", help="lfw pair file")
    parser.add_argument("--model", help="model prototxt OR dir")
    parser.add_argument("--weights", help="model weights")
    parser.add_argument("-t", "--dist_type", default='cosine', help="distance measure ['cosine', 'L2', 'SSD']")
    parser.add_argument("-f", "--do_mirror", default=False,help="mirror image and concatinate features")

    args = parser.parse_args()
    print(args)
    LFW_PAIRS = args.lfw_pair
    LFW_IMG_DIR = args.lfw_dir
    
    if args.weights:
        test_model(args.model,args.weights,args.dist_type,args.do_mirror)
    else:
        test_dir(args.model_dir,args.dist_type,args.do_mirror)

