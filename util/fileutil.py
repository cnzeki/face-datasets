import os
import time
import os.path

def translate_path(src_prefix, dst_prefix, path):
    sp = src_prefix.replace('\\','/')
    dp = dst_prefix.replace('\\','/')
    p = path.replace('\\','/')
    # src -> dst
    if path.startswith(sp):
        rel_p = p[len(sp):len(p)]
        #print(rel_p)
        return dp + rel_p
    # dst -> src
    if path.startswith(dp):
        rel_p = p[len(dp):len(p)]
        return sp + rel_p
    return dp + p

def is_image_file(path):
    img_exts = ['.jpg','.jpeg','.gif','.png','.bmp']
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    return ext in img_exts or ext == ''
        
        
def makedirs(path):
    dir,fname = os.path.split(path)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            pass
            
            
def file_walker(dir,visitor):
    '''
        Recursive walk through a dir
    '''
    filenames = os.listdir(dir)
    for filename in filenames:
        fullpath = os.path.join(dir,filename)
        if os.path.isdir(fullpath):
            file_walker(fullpath, visitor)
        elif os.path.isfile(fullpath):
            visitor.process(fullpath)
            
            
def read_lines(path):
    list = []
    if not os.path.exists(path):
        return list
        
    with open(path, 'r') as f:
        for line in f.readlines():
            list.append(line.strip())
    print('read:%d lines from %s' % (len(list), path))
    return list
    
def list_walker(list_path,visitor):
    list = read_lines(list_path)
    for i in range(len(list)):
        path = list[i]
        visitor.process(path)
        