# coding: utf-8

import cv2
import sys
sys.path.insert(0, '../util')
sys.path.insert(0, '../facealign')

from fileutil import *
from MtcnnPycaffe import MtcnnDetector, draw_and_show
from alignment import FaceAlignVisitor, align_to_96x112
from logfile import *

        
def align_CASIA(src_dir, dst_dir):
    log_open('fail-list.txt')
    detector = MtcnnDetector(minsize=36)
    visitor = FaceAlignVisitor(src_dir,dst_dir,detector, skip_exist = False)
    file_walker(src_dir,visitor)
    log_close()
      
if __name__=='__main__': 
    if len(sys.argv) < 3:
        print('CASIA_image_dir  aligned_dir ')
        exit()
    #    
    src_dir = sys.argv[1]  
    dst_dir = sys.argv[2]
    align_CASIA(src_dir, dst_dir)
