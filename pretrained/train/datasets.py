import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class CasiaWebFace(Dataset):

    LABEL_TXT = '../../data/CASIA_label_{}.txt'

    def __init__(self, mode='train', datapath='../../data/CASIA-WebFace-Aligned', dsize=None):
        self.dsize = dsize

        labelpath = self.LABEL_TXT.format(mode)
        with open(labelpath, 'r') as f:
            listFileLabel = f.readlines()
        self.fileList = []; self.labelList = []
        for fileLabel in listFileLabel:
            file, label = fileLabel.strip().split(' ')
            self.fileList += [os.path.join(datapath, file)]
            self.labelList += [int(label)]
        self.n_class = len(list(set(self.labelList)))

    def __getitem__(self, index):

        file = self.fileList[index]
        label = self.labelList[index]

        image = cv2.imread(file, cv2.IMREAD_COLOR)
        flip = np.random.choice(2) * 2 - 1
        image = image[:, ::flip, :]

        if self.dsize is not None:
            image = cv2.resize(image, self.dsize[::-1])
        
        image = (image - 127.5) / 128.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        return image, label

    def __len__(self):

        return len(self.labelList)
