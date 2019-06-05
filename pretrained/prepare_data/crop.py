import os
import cv2
import time
import numpy as np
from cropAlign import imageAlignCrop

def line2kv(line):
    line = line.strip().split(' ')
    filename = line[0]
    coords   = list(map(float, line[1:]))
    return [filename, coords]

def parseCoord(coords):
    box, score, landmark = coords[:4], coords[4], coords[5:]
    return np.array(box), score, np.array(landmark)

def crop_casia(datapath='../../data/CASIA-WebFace', detected='../../data/CASIA_detect.txt', 
        aligned='../../data/CASIA-WebFace-Aligned', dsize=(112, 96)):
    """
    Notes:
        根据检测出的结果，进行截取，结果保存在文件夹
        - 已对齐： `../../data/CASIA-WebFace-Aligned`
        目录结构一致
    """
    if not os.path.exists(aligned): os.mkdir(aligned)
    
    ## 载入已检测的结果
    start_time = time.time()
    print('\033[2J\033[1;1H')
    print('\033[1;1HLoading detections...\033[s')
    with open(detected, 'r') as f:
        detect = f.readlines()
    detect = list(map(lambda x: line2kv(x), detect))
    detect = {k: v for k, v in detect}
    print('\033[uOK! >>> {:.2f}s'.format(time.time() - start_time))

    i = 0; n = len(detect)
    elapsed_time = 0
    start_time = time.time()

    for subdir in os.listdir(datapath):
        subdirSrc      = os.path.join(datapath, subdir)
        subdirDst   = os.path.join(aligned, subdir)
        if not os.path.exists(subdirDst): os.mkdir(subdirDst)
        
        for imidx in os.listdir(subdirSrc):
            key = '/'.join([subdir, imidx])
            if key not in detect.keys(): continue

            i += 1
            duration = time.time() - start_time
            elapsed_time += duration
            start_time = time.time()
            
            print('\033[2;1H[{:6d}]/[{:6d}] FPS: {:.4f}  Elapsed: {:.4f}h Left: {:.4f}h'.\
                            format(i, n, 1./duration, elapsed_time/3600, (duration*n - elapsed_time)/3600))
            
            ## 读取原图
            srcImg = cv2.imread(os.path.join(subdirSrc, imidx), cv2.IMREAD_COLOR)
            ## 剪裁
            box, score, landmark = parseCoord(detect[key])
            dstImage = imageAlignCrop(srcImg, landmark.reshape(-1, 2), dsize)

            ## 保存结果
            if (dstImage is None) or (dstImage is None): continue
            cv2.imwrite(os.path.join(subdirDst, imidx), dstImage)


def crop_lfw(datapath='../../data/lfw', detected='../../data/lfw_detect.txt', 
        aligned='../../data/lfw-Aligned', dsize=(112, 96)):
    """
    Notes:
        根据检测出的结果，进行截取，结果保存在文件夹
        - 未对齐： `../../data/lfw-Unaligned`
        - 已对齐： `../../data/lfw-Aligned`
        目录结构一致
    """
    if not os.path.exists(aligned): os.mkdir(aligned)
    
    ## 载入已检测的结果
    start_time = time.time()
    print('\033[2J\033[1;1H')
    print('\033[1;1HLoading detections...\033[s')
    with open(detected, 'r') as f:
        detect = f.readlines()
    detect = list(map(lambda x: line2kv(x), detect))
    detect = {k: v for k, v in detect}
    print('\033[uOK! >>> {:.2f}s'.format(time.time() - start_time))

    i = 0; n = len(detect)
    elapsed_time = 0
    start_time = time.time()

    for subdir in os.listdir(datapath):
        subdirSrc = os.path.join(datapath, subdir)
        subdirDst = os.path.join(aligned, subdir)
        if not os.path.exists(subdirDst): os.mkdir(subdirDst)
        
        for imidx in os.listdir(subdirSrc):
            key = '/'.join([subdir, imidx])
            if key not in detect.keys(): continue

            i += 1
            duration = time.time() - start_time
            elapsed_time += duration
            start_time = time.time()
            
            print('\033[2;1H[{:6d}]/[{:6d}] FPS: {:.4f}  Elapsed: {:.4f}h Left: {:.4f}h'.\
                            format(i, n, 1./duration, elapsed_time/3600, (duration*n - elapsed_time)/3600))
            
            ## 读取原图
            srcImg = cv2.imread(os.path.join(subdirSrc, imidx), cv2.IMREAD_COLOR)
            ## 剪裁
            box, score, landmark = parseCoord(detect[key])
            dstImg = imageAlignCrop(srcImg, landmark.reshape(-1, 2), dsize)

            ## 保存结果
            if (dstImg is None) or (dstImg is None): continue
            cv2.imwrite(os.path.join(subdirDst, imidx), dstImg)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="detect dataset")
    parser.add_argument('--data', '-d', required=True, choices=['casia', 'lfw'])
    args = parser.parse_args()
    
    if args.data == 'casia':
        crop_casia()
    elif args.data == 'lfw':
        crop_lfw()