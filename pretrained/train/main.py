import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from config import configer
from datasets import CasiaWebFace, LFWPairs
from metrics import MobileFacenetLoss, MobileFacenetUnsupervisedLoss
from models import MobileFacenet, MobileFacenetUnsupervised
from trainer import Trainer, MobileFacenetTrainer, MobileFacenetUnsupervisedTrainer

import sys
sys.path.append('../prepare_data/')
from label import gen_casia_label

def main(classifypath, verifypath):

    classifyData = CasiaWebFace(datapath=classifypath)
    verifyData = LFWPairs(datapath=verifypath)

    net = MobileFacenet(classifyData.n_class)
    criterion = MobileFacenetLoss()

    ignored_params = list(map(id, net.linear1.parameters()))
    ignored_params += [id(net.weight)]
    prelu_params = []
    for m in net.modules():
        if isinstance(m, nn.PReLU):
            ignored_params += list(map(id, m.parameters()))
            prelu_params += m.parameters()
    base_params = filter(lambda p: id(p) not in ignored_params,
                         net.parameters())
    params = [
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.linear1.parameters(), 'weight_decay': 4e-4},
        {'params': net.weight, 'weight_decay': 4e-4},
        {'params': prelu_params, 'weight_decay': 0.0}
    ]
    
    trainer = MobileFacenetTrainer(
        configer, net, params,
        classifyData, verifyData, 
        criterion, 
        SGD, MultiStepLR
    )
    trainer.train()


def mainUnsupervised(classifypath, verifypath):

    classifyData = CasiaWebFace(datapath=classifypath)
    verifyData = LFWPairs(datapath=verifypath)

    net = MobileFacenetUnsupervised(classifyData.n_class)
    criterion = MobileFacenetUnsupervisedLoss(classifyData.n_class)

    ignored_params = list(map(id, net.linear1.parameters()))
    ignored_params += list(map(id, criterion.parameters()))
    prelu_params = []
    for m in net.modules():
        if isinstance(m, nn.PReLU):
            ignored_params += list(map(id, m.parameters()))
            prelu_params += m.parameters()
    base_params = filter(lambda p: id(p) not in ignored_params,
                         net.parameters())
    params = [
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.linear1.parameters(), 'weight_decay': 4e-4},
        {'params': criterion.parameters(), 'weight_decay': 4e-4},
        {'params': prelu_params, 'weight_decay': 0.0}
    ]
    
    trainer = MobileFacenetUnsupervisedTrainer(
        configer, net, params,
        classifyData, verifyData, 
        criterion, 
        SGD, MultiStepLR
    )
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="detect dataset")
    parser.add_argument('--classify', '-c', 
            choices=['../../data/CASIA-WebFace-Aligned', 
                     '../../data/CASIA/CASIA-WebFace-112X96'])
    parser.add_argument('--verify', '-v', 
            choices=['../../data/lfw-Aligned', 
                     '../../data/lfw-112X96'])
    parser.add_argument('--unsupervised', '-un', action='store_true')
    args = parser.parse_args()

    if not args.unsupervised:
        main(datapath=args.dir)
    else:
        mainUnsupervised(datapath=args.dir)
