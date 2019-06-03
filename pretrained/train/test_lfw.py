import os
import numpy as np

import torch
from torch import cuda
from torch import functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import precision_score, recall_score, accuracy_score

from datasets import LFWPairs
from models import MobileFacenet
from utils import flip, distCosine

def load_net(path):
    netstate = torch.load(path, map_location='cuda:0' if cuda.is_available() else 'cpu')['net']
    net = MobileFacenet(num_classes=netstate['weight'].shape[0])
    net.load_state_dict(netstate)
    return net

def main(thresh=0.3):
    
    dataset = LFWPairs(datapath='../../data/lfw-Aligned')
    dataloader = DataLoader(dataset)
    net = load_net('./ckpt/MobileFacenet_{}.pkl')   # TODO
    if cuda.is_available(): net.cuda()

    f = open("../../cosine_score_py.txt", 'w')
    gt = []; pred = []    
    print('\033[2J'); print('\033[1;1H')
    
    net.eval()
    for i, (X1, X2, y_true) in enumerate(dataloader):
        feat1 = torch.cat([net.get_feature(X1), net.get_feature(flip(X1))], dim=1).view(-1)
        feat2 = torch.cat([net.get_feature(X2), net.get_feature(flip(X2))], dim=1).view(-1)
        cosine = distCosine(feat1, feat2)

        gt += [y_true.cpu().numpy()]
        pred += [cosine.detach().cpu().numpy()]

        line = "{:d} {:d} {:f}\n".format(i, int(gt[i]), float(pred[i]))
        f.write(line)

        print('\033[0;0H\033[K')
        print('[{:4d}]/[{:4d}] {:s}'.format(i, len(dataset), line))

    gt = np.array(gt); pred = np.array(pred)
    pd = np.zeros(len(pred)); pd[pred > thresh] = 1

    gt = gt.reshape(-1); pd = pd.reshape(-1)
    acc = accuracy_score(gt, pd)
    p = precision_score(gt, pd)
    r = recall_score(gt, pd)

    print('accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(acc, p, r))

    f.close()

if __name__ == "__main__":
    main()

"""
lfw-Aligned
../MobileFacenet_best.pkl
accuracy: 0.8163, precision: 0.7854, recall: 0.8689

lfw-112X96
../MobileFacenet_best.pkl
accuracy: 0.9883, precision: 0.9923, recall: 0.9843

lfw-Aligned
./ckpt/MobileFacenet_{}.pkl

"""