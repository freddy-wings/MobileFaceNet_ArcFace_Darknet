import cv2
import numpy as np
import torch
from extract_weights_cfg import _read_ckpt

im  = cv2.imread('images/patch_112x96.jpg')
im  = im.transpose(2, 0, 1)
im  = (torch.from_numpy(im).float() - 127.5) / 128.
im  = im.unsqueeze(0)

net = _read_ckpt('pretrained/MobileFacenet_best.pkl'); net.eval()

feat = net.conv1.conv(im)
feat = net.conv1.bn(feat)
feat = net.conv1.prelu(feat)
with open('images/patch_112x96_py.txt', 'w') as f:
    feat = '\n'.join(map(str, list(feat.view(-1).detach().numpy())))
    f.write(feat)
...
