import cv2
import numpy as np
from cp2tform import cp2tform, warpImage, warpCoordinate

# ALIGNED = [ 1./3, 1./3,
#             2./3, 1./3,
#             1./2, 1./2,
#             1./3, 2./3,
#             2./3, 2./3]

# ALIGNED = [ 30.2946/96, 51.6963/112,
#             65.5318/96, 51.5014/112,
#             48.0252/96, 71.7366/112,
#             33.5493/96, 92.3655/112,
#             62.7299/96, 92.2041/112]

ALIGNED = [ 30.2946, 51.6963,
            65.5318, 51.5014,
            48.0252, 71.7366,
            33.5493, 92.3655,
            62.7299, 92.2041]

def drawCoordinate(im, coord):
    """
    Params:
        im:  {ndarray(H, W, 3)}
        coord: {ndarray(n, 2)}
    Returns:
        im:  {ndarray(H, W, 3)}
    """
    coord = coord.astype('int')
    for i in range(coord.shape[0]):
        cv2.circle(im, tuple(coord[i]), 1, (255, 255, 255), 3)
    return im
    

def imageAlignCrop(im, landmark, dsize=(112, 96)):
    """
    Params:
        im:         {ndarray(H, W, 3)}
        landmark:   {ndarray(5, 2)}
        dsize:      {tuple/list(H, W)}
    Returns:
        dstImage:   {ndarray(h, w, 3)}
    Notes:
        对齐后裁剪
    """
    ## 变换矩阵
    M = cp2tform(landmark, np.array(ALIGNED).reshape(-1, 2))
    ## 用矩阵变换图像
    warpedImage = warpImage(im, M)
    cv2.imshow("a", warpedImage)
    ## 裁剪固定大小的图片尺寸
    h, w = dsize
    dstImage = warpedImage[:h, :w]
    cv2.imshow("b", dstImage)
    cv2.waitKey(0)
    return dstImage