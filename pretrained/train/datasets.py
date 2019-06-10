import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class CasiaWebFace(Dataset):

    def __init__(self, labelpath='../../data/CASIA_label.txt', datapath='../../data/CASIA-WebFace-Aligned', dsize=None):
        self.dsize = dsize

        with open(labelpath, 'r') as f:
            listFileLabel = f.readlines()

        self.fileList = []; self.labelList = []
        for fileLabel in listFileLabel:
            file, label = fileLabel.strip().split(' ')
            filepath = os.path.join(datapath, file)

            if os.path.exists(filepath):
                self.fileList += [filepath]
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

class LFWPairs(Dataset):

    def __init__(self, datapath='../../data/lfw-Aligned', dsize=None):
        self.dsize = dsize

        if datapath == '../../data/lfw-Aligned':
            pairstxt = '../../data/lfw_pairs.txt'
        else:
            pairstxt = '../../data/pairs.txt'
        with open(pairstxt, 'r') as f:
            pairs = f.readlines()[1:]

        self.pairList = []; self.labelList = []
        for pair in pairs:
            pair = pair.strip().split('\t')
            if len(pair) == 3:
                name, index1, index2 = pair
                name1 = name; name2 = name
            else:
                name1, index1, name2, index2 = pair
            
            path1 = '{:s}/{:s}/{:s}_{:04d}.jpg'.format(datapath, name1, name1, int(index1))
            path2 = '{:s}/{:s}/{:s}_{:04d}.jpg'.format(datapath, name2, name2, int(index2))

            if os.path.exists(path1) and os.path.exists(path2):
                self.pairList += [' '.join([path1, path2])]
                self.labelList += [1 if name1 == name2 else 0]

    def __getitem__(self, index):

        pair = self.pairList[index].split(' ')
        label = self.labelList[index]

        img1 = cv2.imread(pair[0], cv2.IMREAD_COLOR)
        img2 = cv2.imread(pair[1], cv2.IMREAD_COLOR)

        if self.dsize is not None:
            img1 = cv2.resize(img1, self.dsize[::-1])
            img2 = cv2.resize(img2, self.dsize[::-1])

        img1 = (img1 - 127.5) / 128.0
        img2 = (img2 - 127.5) / 128.0
        img1 = img1.transpose(2, 0, 1)
        img2 = img2.transpose(2, 0, 1)
        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()

        return img1, img2, label

    def __len__(self):

        return len(self.pairList)

