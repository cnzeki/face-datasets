# coding: utf-8
import os
import numpy as np
import math
import cv2
from fileutil import *
from logfile import *

def compute_affine_transform(points, refpoints,  w = None):
    '''
    compute the affine tranform matrix
    '''
    if w == None:
        w = [1] * (len(points) * 2)
    assert(len(w) == 2*len(points))
    y = []
    for n, p in enumerate(refpoints):
        y += [p[0]/w[n*2], p[1]/w[n*2+1]]
    A = []
    for n, p in enumerate(points):
        A.extend([ [p[0]/w[n*2], p[1]/w[n*2], 0, 0, 1/w[n*2], 0], [0, 0, p[0]/w[n*2+1], p[1]/w[n*2+1], 0, 1/w[n*2+1]] ])
    
    lstsq = cv2.solve(np.array(A), np.array(y), flags=cv2.DECOMP_SVD)
    h11, h12, h21, h22, dx, dy = lstsq[1]

    #R = np.array([[h11, h12, dx], [h21, h22, dy]])
    # The row above works too - but creates a redundant dimension
    R = np.array([[h11[0], h12[0], dx[0]], [h21[0], h22[0], dy[0]]])
    return R

    
def compute_similarity_transform(src, dst):
    '''
    compute the similarity tranform matrix
    '''
    assert len(src) == len(dst)
    N = len(src)
    A = np.zeros((N*2, 4), dtype=np.float)
    B = np.zeros((N*2, 1), dtype=np.float)
    for i in range(N):
        # x'
        row = i * 2
        A[row][0] = src[i][0]
        A[row][1] = -src[i][1]
        A[row][2] = 1
        A[row][3] = 0
        B[row][0] = dst[i][0]
        # y'
        row += 1
        A[row][0] = src[i][1]
        A[row][1] = src[i][0]
        A[row][2] = 0
        A[row][3] = 1
        B[row][0] = dst[i][1]
    AT = np.transpose(A)
    invAA = np.linalg.inv(np.dot(AT,A))
    AAT = np.dot(invAA,AT)
    X = np.dot(AAT,B)  
    
    R = np.array([[X[0], -X[1], X[2]], [X[1], X[0], X[3]]])
    return R

def cv2_imread(path):
    img = cv2.imread(path)
    if img is not None:
        return img
    # try .png
    print('Not find:%s try:%s' % (path, path+'.png'))
    img = cv2.imread(path+'.png')
    return img
    

def cv2_imwrite(path,img):
    ret = True
    title, ext = os.path.splitext(path)
    ext = ext.lower()
    makedirs(path)
    # append gif with .png
    if ext == '.gif':
        ext = '.png'
        path = path+'.png'
    elif ext == '':
        path = path+'.png'
        
    try:
        cv2.imwrite(path, img)
    except:
        ret = False
        
    return ret

    
def alignface_96x112(img, points, pading=0, trans_type = 'similarity'):
    """
        crop and align face
    Parameters:
    ----------
        img: numpy array, bgr order of shape (1, 3, n, m)
            input image
        points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
        padding: default 0
        trans_type: similarity OR affine, default similarity
    Return:
    -------
        crop_imgs: list, n
            cropped and aligned faces 
    """
    # average positions of face points
    mean_face_shape_x = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299]
    mean_face_shape_y = [51.6963, 51.5014, 71.7366, 92.3655, 92.2041]
    # tranform
    tranform = compute_similarity_transform
    if trans_type == 'affine' :
        tranform = compute_affine_transform
    # do the job
    crop_imgs = []
    for p in points:
        shape  =[]
        for k in range(len(p)/2):
            shape.append(p[k])
            shape.append(p[k+5])

        from_points = []
        to_points = []

        for i in range(len(shape)/2):
            x = mean_face_shape_x[i] + pading
            y = mean_face_shape_y[i] + pading
            to_points.append([x, y])
            from_points.append([shape[2*i], shape[2*i+1]])
            
        N = tranform(from_points,to_points)
        chips = cv2.warpAffine(img, N, (96+2*pading, 112+2*pading) )
        crop_imgs.append(chips)

    return crop_imgs

   
def align_to_96x112(img, points, pading=0, trans_type = 'similarity'):
    """
        crop and align face
    Parameters:
    ----------
        img: numpy array, bgr order of shape (1, 3, n, m)
            input image
        points: list, 1 x 10 (x1, x2 ... x5, y1, y2 ..y5)
        padding: default 0
        trans_type: similarity OR affine, default similarity
    Return:
    -------
        cropped and aligned face
    """
    # average positions of face points
    mean_face_shape_x = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299]
    mean_face_shape_y = [51.6963, 51.5014, 71.7366, 92.3655, 92.2041]
    # tranform
    tranform = compute_similarity_transform
    if trans_type == 'affine' :
        tranform = compute_affine_transform
    # do the job
    from_points = []
    to_points = []

    for i in range(len(points)/2):
        if points[i] == None:
            continue
        x = mean_face_shape_x[i] + pading
        y = mean_face_shape_y[i] + pading
        to_points.append([x, y])
        from_points.append([points[i], points[i + 5]])
        
    N = tranform(from_points,to_points)
    chip = cv2.warpAffine(img, N, (96+2*pading, 112+2*pading) )
    return chip


class FaceAlignVisitor(object):
    """
        Megaface alignment
    """
    def __init__(self,
                 src_prefix,
                 dst_prefix,
                 detector,
                 skip_exist = True,
                 transform = 'sililarity',
                 pading = 0):
        
        self.src_prefix = src_prefix
        self.dst_prefix = dst_prefix
        self.skip_exist = skip_exist
        self.detector = detector
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
              
        # find biggest face    
        max_idx = 0
        max_width = 0
        for i, box in enumerate(boxes):
            if box[2] > max_width:
                max_width = box[2]
                max_idx = i
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