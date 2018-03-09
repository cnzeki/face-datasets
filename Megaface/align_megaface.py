# coding: utf-8

import cv2
import os
import sys
import time
import os.path
sys.path.insert(0, '../facealign')
sys.path.insert(0, '../util') 

from fileutil import *
from MtcnnPycaffe import MtcnnDetector, draw_and_show
from alignment import FaceAlignVisitor, align_to_96x112
from logfile import *

import json
import argparse

def parse_Megaface_json(data):
    #print(data)
    count = 0
    # bound, points
    rect = [0,0,0,0]
    points = [None for i in range(10)]
    # bounding_box
    if 'bounding_box' in data:
        rect[0] = data['bounding_box']['x']
        rect[1] = data['bounding_box']['y']
        rect[2] = data['bounding_box']['width']
        rect[3] = data['bounding_box']['height']
        # to int
        rect = [int(i) for i in rect]
    # points
    landmarks = data['landmarks'] if 'landmarks' in data else None
    if landmarks is None:
        return rect, points, count
        
    for i in range(5):
        key = str(i)
        if key in landmarks:
            points[i] = landmarks[key]['x']
            points[i+5] = landmarks[key]['y']
            count += 1
    return rect, points, count

    
def align_Megaface_fail_image(src_dir, dst_dir, path):
    # load image
    img = cv2.imread(path)
    dst_path = translate_path(src_dir, dst_dir, path)
    # has json
    json_path = path + ".json"
    if not os.path.exists(json_path):
        return False
    # load json    
    with open(json_path, 'r') as f:
        data = json.load(f)
    # parse json
    rect, points, count = parse_Megaface_json(data)
    if count >= 2: # has key points
        aligned = align_to_96x112(img, points)
        #draw_and_show(img, [rect], [points])
        cv2.waitKey(0)
    elif rect[2] > 0: # has bbox
        croped = img[rect[1]:(rect[1]+rect[3]),rect[0]:(rect[0]+rect[2])]
        aligned = cv2.resize(croped, dsize=(96,112))
    else: # nothing, just crop
        aligned = cv2.resize(img, dsize=(96,112))
        
    # save result
    makedirs(dst_path)
    cv2.imwrite(dst_path, aligned)
    return True
    
    
class MegafaceFailVisitor(object):
    """
        Megaface alignment
    """
    def __init__(self,
                 src_prefix,
                 dst_prefix):
        
        self.src_prefix = src_prefix
        self.dst_prefix = dst_prefix
        
    def process(self, path):
        # JPG file
        title, ext = os.path.splitext(path)
        if ext.upper() != '.JPG':
            return True
        ret = align_Megaface_fail_image(self.src_prefix, self.dst_prefix, path)
        if not ret:
            print("Can't find json for :%s" % (path))
            log_write(path)
        return ret

         
def align_Megaface_fail(src_dir, dst_dir, todo_list, fail_list):
    visitor = MegafaceFailVisitor(src_dir,dst_dir)
    log_open(fail_list)    
    list_walker(todo_list,visitor)
    log_close()
    
    
def align_Megaface_detect(src_dir, dst_dir, fail_list, skip_exist=False, transform='similarity'):
    detector = MtcnnDetector( minsize=36 )
    log_open(fail_list)    
    visitor = FaceAlignVisitor(src_dir,dst_dir,detector, skip_exist=skip_exist, transform=transform)
    file_walker(src_dir,visitor)
    log_close()
    
    
def load_Megaface_features_list(src_dir, json_path):
    # load json    
    with open(json_path, 'r') as f:
        data = json.load(f)
    rel_list = data['path']
    # to fullpath
    path_list = [ os.path.join(src_dir,p) for p in rel_list ]
    #print(path_list)
    return path_list

def align_Megaface_features_list(src_dir, dst_dir, json_path):
    # load file list
    path_list = load_Megaface_features_list(src_dir,json_path)
    # init detector
    detector = MtcnnDetector( minsize=36 )
    # align by detection
    visitor = FaceAlignVisitor(src_dir,dst_dir,detector)
    detect_fail_list = '~temp.txt'
    log_open(detect_fail_list)    
    for i in range(len(path_list)):
        path = path_list[i]
        visitor.process(path)
    log_close()
    # align by meta
    align_Megaface_fail(src_dir, dst_dir, detect_fail_list, json_path+".fail.txt")
    
    
def align_Megaface(src_dir, dst_dir, templatelists_dir):
    for i in range(1,7):
        set_size = 10**i
        json_path = 'megaface_features_list.json_%d_1' % (set_size)
        json_path = os.path.join(templatelists_dir,json_path)
        print(json_path)
        align_Megaface_features_list(src_dir, dst_dir, json_path)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_dir",help="Megaface image dir")
    parser.add_argument("-d", "--dst_dir",help="Aligned image dir")
    parser.add_argument("-l", "--fail_list", default='megafail.txt',help="Failed image list")
    parser.add_argument("-t", "--transform", default='similarity',help="similarity OR affine")
    parser.add_argument("-f", "--skip_exist", default=True,help="skip if aliged file exists")
    args = parser.parse_args()
    # get config
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    fail_list = args.fail_list
    skip_exist = args.skip_exist
    transform = args.transform
    # detect 
    detect_fail_list = '~detect-fail.txt'
    align_Megaface_detect(src_dir, dst_dir, detect_fail_list, skip_exist, transform)
    align_Megaface_fail(src_dir, dst_dir, detect_fail_list, fail_list)
    #os.remove(detect_fail_list)
    
if __name__=='__main__': 
    if len(sys.argv) < 3:
        print('megaface_image_dir  aligned_dir  distractor_templatelists_dir')
        exit()
        
    src_dir = sys.argv[1]  
    dst_dir = sys.argv[2]
    templatelists_dir = sys.argv[3]    
    align_Megaface(src_dir, dst_dir, templatelists_dir)


