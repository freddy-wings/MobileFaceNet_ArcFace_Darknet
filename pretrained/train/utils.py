import os
import time
import torch
import numpy as np

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

def kFold(gt, pred, folds=10, steps=100):
    """
    Params:
        gt:   {ndarray(N)}
        pred: {ndarray(N)}
        fold: {int}
        steps:{int}
    """
