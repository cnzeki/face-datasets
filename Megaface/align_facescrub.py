# coding: utf-8

import cv2
import os,sys
import time
import os.path
import math
sys.path.insert(0, '../facealign')
sys.path.insert(0, '../util')  

from fileutil import *
from MtcnnPycaffe import MtcnnDetector, draw_and_show
from alignment import *
from logfile import *

import json
import argparse

def IoU(bbox1, bbox2): 
    intersect_bbox = [max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]),
      min(bbox1[0]+bbox1[2], bbox2[0]+bbox2[2]), min(bbox1[1]+bbox1[3], bbox2[1]+bbox2[3])]
    overlap = (intersect_bbox[2] - intersect_bbox[0]) * (intersect_bbox[3] - intersect_bbox[1])
    overlap_rate = overlap / (bbox1[2]*bbox1[3] + bbox2[2]*bbox2[3] - overlap)
    return overlap_rate
    
    
def load_bbox_file(path, dict):
    lines = read_lines(path)
    # skip first
    for i in range(1,len(lines)):
        line = lines[i]
        segs = line.split('\t')
        name = segs[0]
        face_id = segs[2]
        bbox = segs[4]
        vals = bbox.split(',')
        x0 = int(vals[0])
        y0 = int(vals[1])
        x1 = int(vals[2])
        y1 = int(vals[3])
        rect = [x0,y0,x1 - x0, y1 - y0]
        # name_faceid
        key = name + '_' + face_id
        dict[key] = rect
        
    return dict
        
class FacescrubAlignVisitor(object):
    """
        Megaface alignment
    """
    def __init__(self,
                 src_prefix,
                 dst_prefix,
                 detector,
                 bbox,
                 skip_exist = False,
                 transform = 'sililarity',
                 pading = 0):
        
        self.src_prefix = src_prefix
        self.dst_prefix = dst_prefix
        self.skip_exist = skip_exist
        self.detector = detector
        self.bbox = bbox
        self.transform = transform
        self.pading = pading
        # statistic
        self.done_count = 0
        self.fail_count = 0
                
    def process(self, path):
        if not is_image_file(path):
            return True
            
        dst_path = translate_path(self.src_prefix, self.dst_prefix, path)
        
        if self.skip_exist and os.path.exists(dst_path):
            # print('skip:%s' % path)
            return True
        #print('%s -> %s' % (path, dst_path))    
        img = cv2_imread(path)
        if img is None:
            print('load error:%s'%(path))
            log_write(path)
            self.fail_count += 1
            return False
            
        #print('run:%s/%s'%(subdir,filename))
        try:
            boxes, points = self.detector.detect_face(img)
        except:
            print('detect error:%s'%(path))
            log_write(path)
            self.fail_count += 1
            return False
        if points is None or len(points) == 0:
            log_write(path)
            self.fail_count += 1
            return False
              
        # find the one largest IoU
        dir, fname = os.path.split(path)
        key, _ = os.path.splitext(fname)
        target_box = self.bbox[key]
        max_idx = 0
        max_iou = 0
        for i, box in enumerate(boxes):
            box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            iou = IoU(box, target_box)
            if iou > max_iou:
                max_iou = iou
                max_idx = i
        # check iou
        if max_iou < 0.3:
            #cv2.rectangle(img, (target_box[0],target_box[1]), 
            #  (target_box[0] + target_box[2], target_box[1] + target_box[3]), (0,255,0), 2)
            #draw_and_show(img, boxes, points )
            #ch = cv2.waitKey(0)
            ch = 0
            if ch == 27:
                log_write(path)
                self.fail_count += 1
                return False
            
        max_chip = align_to_96x112(img, points[max_idx], self.pading, trans_type = self.transform)
        #draw_and_show(img,boxes, points )
        #cv2.imshow('chip', max_chip)
        #cv2.waitKey(0)
        makedirs(dst_path)
        ret = cv2_imwrite(dst_path, max_chip)
        if ret == False:
            print('imwrite error:%s'%(path))
            log_write(path)
            self.fail_count += 1
            return False
            
        # report
        if self.done_count % 100 == 0:
            print('done:%05d, fail:%05d img:%s'%(self.done_count, self.fail_count, path))
            
        self.done_count += 1
        return True
        
def align_facescrub_uncropped(src_dir, dst_dir, templatelists_path, dict, gpu_id = 0):
    # load json  
    with open(templatelists_path, 'r') as f:
        data = json.load(f)
    rel_list = data['path']

    # to fullpath
    path_list = [ os.path.join(src_dir,p) for p in rel_list ]
    # init detector
    detector = MtcnnDetector( minsize=36, gpu_id = gpu_id )
    # align by detection
    visitor = FacescrubAlignVisitor(src_dir,dst_dir,detector, dict)
    detect_fail_list = templatelists_path + '.detect-fail.txt'
    log_open(detect_fail_list)  
    total_size = len(path_list)  
    for i in range(total_size):
        path = path_list[i]
        #print('%d/%d %s' % (i,total_size,path))
        visitor.process(path)
    log_close() 
    
    
def align_facescrub_fail(src_dir, dst_dir, templatelists_path, dict, gpu_id = 0):
    # init detector
    detector = MtcnnDetector( minsize=36, gpu_id = gpu_id )
    # align by detection
    visitor = FacescrubAlignVisitor(src_dir,dst_dir,detector, dict)
    detect_fail_list = templatelists_path + '.detect-fail.txt'
    log_open(templatelists_path + '.final-fail.txt') 
    list_walker(detect_fail_list,visitor)
    log_close()

def align_facescrub_fail_json(src_dir, dst_dir, templatelists_path, dict, json_path):
    # load json  
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(data)
    list = read_lines(templatelists_path + '.final-fail.txt')
    for path in list:
        dst_path = translate_path(src_dir, dst_dir, path)
        dir, fname = os.path.split(path)
        key, _ = os.path.splitext(fname)
        print(key)
        target_box = dict[key]
        img = cv2_imread(path)
        point = data[key]
        xxyy = []
        for i in range(5):
            xxyy.append(point[i*2])
        for i in range(5):
            xxyy.append(point[i*2+1])
        print(xxyy)    
        max_chip = align_to_96x112(img, xxyy)
        makedirs(dst_path)
        cv2_imwrite(dst_path, max_chip)
        #draw_and_show(img, [target_box], [xxyy] )
        #ch = cv2.waitKey(0)
        
        
def detect_facescrub_landmarks(src_dir, templatelists_path, bbox, detector):
    # load json  
    with open(templatelists_path, 'r') as f:
        data = json.load(f)
    rel_list = data['path']

    landmarks = {}
    for rel_path in rel_list:
        # to fullpath
        path = os.path.join(src_dir, rel_path)
        img = cv2_imread(path)
        try:
            boxes, points = detector.detect_face(img)
        except:
            print('detect error:%s'%(path))

        if points is None or len(points) == 0:
            continue
              
        # find the one largest IoU
        dir, fname = os.path.split(path)
        key, _ = os.path.splitext(fname)
        target_box = bbox[key]
        max_idx = 0
        max_iou = 0
        for i, box in enumerate(boxes):
            box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            iou = IoU(box, target_box)
            if iou > max_iou:
                max_iou = iou
                max_idx = i
        landmarks[key] = points[max_idx].tolist()
        
    return landmarks
        
    
def correct_facescrub_json(src_dir, dst_dir, dict, json_path):
    # load json  
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(data)
    for key, value in data.items():
        name, image_id = key.split('_')
        path = os.path.join(src_dir,name+'/'+key+'.jpg')
        dst_path = translate_path(src_dir, dst_dir, path)
        target_box = dict[key]
        img = cv2_imread(path)
        point = data[key]
        xxyy = []
        for i in range(5):
            xxyy.append(point[i*2])
        for i in range(5):
            xxyy.append(point[i*2+1])
        print(xxyy)
        print(key)
        max_chip = align_to_96x112(img, xxyy)
        makedirs(dst_path)
        #cv2_imwrite(dst_path, max_chip)
        draw_and_show(img, [target_box], [xxyy] )
        cv2.imshow('chip', max_chip)
        ch = cv2.waitKey(0)

def merge_landmarks(labeled_json, detect_json, dst_json):
    # load json  
    with open(labeled_json, 'r') as f:
        data = json.load(f)
    # load detect
    with open(detect_json, 'r') as f:
        landmarks = json.load(f)
    # merge
    for key, value in data.items():
        point = value
        xxyy = []
        for i in range(5):
            xxyy.append(point[i*2])
        for i in range(5):
            xxyy.append(point[i*2+1])
        landmarks[key] = xxyy
    # output
    with open(dst_json, 'w') as f:
        f.write(json.dumps(landmarks))
    print(len(landmarks))
    
def align_facescrub_by_landmark(src_dir, dst_dir, templatelists_path, landmarks_path):
    # path list  
    with open(templatelists_path, 'r') as f:
        data = json.load(f)
    rel_list = data['path']
    # landmarks
    with open(landmarks_path, 'r') as f:
        landmarks = json.load(f)
        
    for rel_path in rel_list:
        # to fullpath
        path = os.path.join(src_dir, rel_path)
        img = cv2_imread(path)
        dst_path = translate_path(src_dir, dst_dir, path)
        dir, fname = os.path.split(path)
        key, _ = os.path.splitext(fname)
        points = landmarks[key]  
        max_chip = align_to_96x112(img, points)
        makedirs(dst_path)
        cv2_imwrite(dst_path, max_chip)
        #cv2.imshow('face', max_chip)
        #ch = cv2.waitKey(1)
        
'''
wrong label:Richard Madden_48806
'''           
if __name__=='__main__': 
    if len(sys.argv) < 3:
        print('facescrub_image_dir  aligned_dir  features_list_json_path')
        exit()
    #    
    src_dir = sys.argv[1]  
    dst_dir = sys.argv[2]
    templatelists_path = sys.argv[3]
    merged_json = './facescrub_80_landmark5.json' 
    align_facescrub_by_landmark(src_dir, dst_dir, templatelists_path, merged_json)
    

