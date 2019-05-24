import cv2
import numpy as np
import torch
from extract_weights_cfg import _read_ckpt

im  = cv2.imread('images/patch_112x96.jpg')
im  = im.transpose(2, 0, 1)
im  = (torch.from_numpy(im).float() - 127.5) / 128.
im  = im.unsqueeze(0)

net = _read_ckpt('pretrained/MobileFacenet_best.pkl'); net.eval()
f = open('images/patch_112x96_py.txt', 'w')

x = net.conv1.conv(im)
f.write('conv1.conv =================\n' + '\n'.join(map(str, list(x.view(-1).detach().numpy()))))
x = net.conv1.bn(x)
f.write('\nconv1.bn   =================\n' + '\n'.join(map(str, list(x.view(-1).detach().numpy()))))
x = net.conv1.prelu(x)
f.write('\nconv1.prelu=================\n' + '\n'.join(map(str, list(x.view(-1).detach().numpy()))))
x = net.dw_conv1(x)
f.write('\ndw_conv1   =================\n' + '\n'.join(map(str, list(x.view(-1).detach().numpy()))))
for i in range(15):
    x = net.blocks[i](x)
    f.write('\nblocks %2d =================\n'%i + '\n'.join(map(str, list(x.view(-1).detach().numpy()))))
x = net.conv2(x)
f.write('\nconv2      =================\n' + '\n'.join(map(str, list(x.view(-1).detach().numpy()))))
x = net.linear7.conv(x)
f.write('\nlinear7.conv =================\n' + '\n'.join(map(str, list(x.view(-1).detach().numpy()))))
x = net.linear7.bn(x)
f.write('\nlinear7.bn =================\n' + '\n'.join(map(str, list(x.view(-1).detach().numpy()))))
x = net.linear1.conv(x)
f.write('\nlinear1.conv =================\n' + '\n'.join(map(str, list(x.view(-1).detach().numpy()))))
x = net.linear1.bn(x)
f.write('\nlinear1.bn =================\n' + '\n'.join(map(str, list(x.view(-1).detach().numpy()))))

f.close()