import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from config import configer
from datasets import CasiaWebFace
from metrics import MobileFacenetLoss, MobileFacenetUnsupervisedLoss
from models import MobileFacenet, MobileFacenetUnsupervised
from trainer import Trainer, TrainerUnsupervised

import sys
sys.path.append('../prepare_data/')
from label import gen_casia_label

def main(datapath):

    gen_casia_label(prefix=datapath)
    
    trainset = CasiaWebFace(mode='train', datapath=datapath)
    validset = CasiaWebFace(mode='valid', datapath=datapath)
    assert trainset.n_class == validset.n_class

    net = MobileFacenet(trainset.n_class)
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

    trainer = Trainer(
        configer,
        net, params,
        trainset, validset,
        criterion, 
        SGD, MultiStepLR,
        num_to_keep=1, resume=False
    )
    trainer.train()


def mainUnsupervised(datapath):

    gen_casia_label(prefix=datapath)
    
    trainset = CasiaWebFace(mode='train', datapath=datapath)
    validset = CasiaWebFace(mode='valid', datapath=datapath)
    assert trainset.n_class == validset.n_class

    net = MobileFacenetUnsupervised(trainset.n_class)
    criterion = MobileFacenetUnsupervisedLoss()

    ignored_params = list(map(id, net.linear1.parameters()))
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

    trainer = TrainerUnsupervised(
        configer,
        net, params,
        trainset, validset,
        criterion, 
        SGD, MultiStepLR,
        num_to_keep=1, resume=False
    )
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="detect dataset")
    parser.add_argument('--dir', '-d', default='../../data/CASIA-WebFace-Aligned', 
            choices=['../../data/CASIA-WebFace-Aligned', '../../data/CASIA/CASIA-WebFace-112X96'])
    parser.add_argument('--unsupervised', '-un', action='store_true')
    args = parser.parse_args()

    if not args.unsupervised:
        main(datapath=args.dir)
    else:
        mainUnsupervised(datapath=args.dir)
