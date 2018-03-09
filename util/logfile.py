# coding: utf-8
error_log = None
def log_open(path):
    global error_log
    if error_log:
        return
    error_log = open(path,'w')
    return error_log

def log_write(img_path):
    global error_log
    if error_log:
        error_log.write(img_path+'\n')
        error_log.flush()
    
def log_close():
    global error_log
    error_log.close()
    error_log = None