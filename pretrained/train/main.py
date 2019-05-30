import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from config import configer
from datasets import CasiaWebFace
from metrics import MobileFacenetLoss
from models import MobileFacenet
from trainer import Trainer

def main():
    
    trainset = CasiaWebFace(mode='train')
    validset = CasiaWebFace(mode='valid')
    assert trainset.n_class == validset.n_class

    net = MobileFacenet(trainset.n_class)
    params = net.parameters()
    criterion = MobileFacenetLoss()

    trainer = Trainer(
        configer,
        net, params,
        trainset, validset,
        criterion, 
        SGD, MultiStepLR,
        num_to_keep=5, resume=False
    )
    trainer.train()


if __name__ == "__main__":
    main()