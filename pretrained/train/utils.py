import os
import time
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

getTime  = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def prob2label(y_pred_prob):
    """
    Params:
        y_pred_prob:{tensor(N, n_classes)
    Returns:
        y_pred:     {tensor(N)}
    """
    return torch.argmax(y_pred_prob, dim=1)

def accuracy(y_pred, y_true):
    """
    Params:
        y_pred: {tensor(N)}
        y_true: {tensor(N)}
    Returns:
        acc:    {tensor(1)}
    """
    acc = torch.mean((y_pred==y_true).float())
    return acc

def precision(y_pred, y_true):
    """
    Params:
        y_pred: {tensor(N)}
        y_true: {tensor(N)}
    Returns:
        p:      {tensor(1)}
    Notes: 
        precision = \frac{tp}{tp + fp}
    """
    

def recall(y_pred, y_true):
    """
    Params:
        y_pred: {tensor(N)}
        y_true: {tensor(N)}
    Returns:
        r:      {tensor(1)}
    Notes: 
        recall = \frac{tp}{tp + fn}
    """



def flip(X):
    """
    Params:
        X: {tensor(N, C, H, W)}
    Returns:
        Xf: {tensor(N, C, H, W)}
    """
    Xf = torch.from_numpy(X.cpu().numpy()[:, :, :, ::-1].copy())
    if torch.cuda.is_available(): Xf = Xf.cuda()
    return Xf

def distCosine(x1, x2):
    """
    Params:
        x1: {tensor(n)}
        x2: {tensor(n)}
    Returns:
        cos: {tensor(1)}
    """
    x1n = x1 / torch.norm(x1)
    x2n = x2 / torch.norm(x2)
    cos = torch.dot(x1n, x2n)
    return cos

def cvSelectThreshold(X, y, folds=10, steps=1000):
    """
    Params:
        X:    {ndarray(N)}
        y:    {ndarray(N)}
        fold: {int}
        steps:{int}
    """
    thresholds = np.linspace(np.min(X), np.max(X), steps)
    kAcc = np.zeros(folds); kF1 = np.zeros(folds)

    cv = KFold(n_splits=folds, shuffle=True)
    for k, (trainIdx, validIdx) in enumerate(cv.split(y)):
        X_train = X[trainIdx]; y_train = y[trainIdx]
        X_valid = X[validIdx]; y_valid = y[validIdx]

        threshBest = -1; accBest = float('-inf'); f1Best = float('-inf')
        for t in thresholds:
            y_train_p = np.zeros_like(y_train)
            y_train_p[X_train > t] = 1.
            acc = accuracy_score(y_train_p, y_train)
            f1  = f1_score(y_train_p, y_train)
            if f1 > f1Best:
                threshBest = t; accBest = acc; f1Best = f1
        
        y_valid_p = np.zeros_like(y_valid)
        y_valid_p[X_valid > threshBest] = 1.
        kAcc[k] = accuracy_score(y_valid_p, y_valid)
        kF1 [k] = f1_score(y_valid_p, y_valid)
    
    thresholds = np.mean(thresholds); kAcc = np.mean(kAcc); kF1 = np.mean(kF1)
    return thresholds, kAcc, kF1
