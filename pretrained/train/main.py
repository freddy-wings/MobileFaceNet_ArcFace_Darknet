import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from config import configer
from datasets import CasiaWebFace
from metrics import MobileFacenetLoss
from models import MobileFacenet
from trainer import Trainer

def main():
    
    trainset = CasiaWebFace(mode='train', datapath='../data/CASIA/CASIA-WebFace-112X96')
    validset = CasiaWebFace(mode='valid', datapath='../data/CASIA/CASIA-WebFace-112X96')
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


if __name__ == "__main__":
    main()
