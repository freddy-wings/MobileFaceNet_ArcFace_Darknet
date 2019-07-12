import math
import torch
import torch.nn as nn
from torch.nn import Parameter

"""
关于`ArcMarginProduct`中单调性的处理

import numpy as np
import matplotlib.pyplot as plt

def phi(x, m=0.5):
    """
    Params:
        x: [0, pi]
    """

    th = np.cos(np.pi - m)
    mm = np.sin(np.pi - m) * m

    cos_x = np.cos(x)

    # y = np.where(cos_x > 0, np.cos(x + m), np.cos(x))         # 不单调
    y = np.where(cos_x > th, np.cos(x + m), np.cos(x) - mm)     # 单调

    return y


if __name__ == '__main__':

    x = np.linspace(0, np.pi, 200)
    y = phi(x)

    plt.figure(0)
    plt.plot(x, y)
    plt.show()
"""

class ArcMarginProduct(nn.Module):

    def __init__(self, s=32.0, m=0.50, easy_margin=False):

        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)     # self.th = - math.cos(m)
        self.mm = math.sin(math.pi - m) * m # self.mm =   math.sin(m) * m

    def forward(self, cosine, label):
        """
        Params:
            cosine: {tensor(N, n_classes)} 每个样本(N)，到各类别(n_classes)矢量的余弦值
            label:  {tensor(N)}
        Returns:
            output: {tensor(N, n_classes)}
        Notes:
            - L^{(i)} = - \log \frac{e^{s \cos(\theta_{y_i} + m)}}
                                    {e^{s \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cos \theta_j}}
            - \cos(\theta_{y_i} + m)} = \cos \theta_{y_i} \cos m - \sin \theta_{y_i} \sin m
        """
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m                           # phi(theta) = cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)                          # cos(theta) = cos(theta + m)      if cos(theta) > 0
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)    # cos(theta) = cos(theta + m)      if cos(theta) > cos(m) else
                                                                                # cos(theta) = cos(theta) - m * sin(m)

        one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)                   # cos(theta)[y_i, i] = phi
        output = self.s * output
        return output


class MobileFacenetLoss(nn.Module):

    def __init__(self):

        super(MobileFacenetLoss, self).__init__()
        self.ArcMargin = ArcMarginProduct()
        self.classifier = nn.CrossEntropyLoss()

    def forward(self, pred, gt):

        output = self.ArcMargin(pred, gt)
        loss = self.classifier(output, gt)
        return loss


## TODO
class MobileFacenetUnsupervisedLoss(nn.Module):
    """
    Notes:
    -   
    """
    def __init__(self, num_classes, use_entropy=True):
        super(MobileFacenetUnsupervisedLoss, self).__init__()

        self.use_entropy = use_entropy

        self.m = Parameter(torch.Tensor(num_classes, 128))
        self.s1 = Parameter(torch.ones(num_classes))
        nn.init.xavier_uniform_(self.m)

        if use_entropy:
            self.s2 = Parameter(torch.ones(num_classes))
    
    def _entropy(self, x):
        """
        Params:
            x: tensor{(n)}
        """
        x = torch.where(x<=0, 1e-16*torch.ones_like(x), x)
        x = torch.sum(- x * torch.log(x))
        return x

    def _softmax(self, x):
        """
        Params:
            x: {tensor(n)}
        """
        x = torch.exp(x - torch.max(x))
        return x / torch.sum(x)

    def _f(self, x, m, s=None):
        """
        Params:
            x: {tensor(n_features(n))}
            m: {tensor(n_features(C, n))}
            s: {tensor(n_features(C))}
        Returns:
            y: {tensor(n_features(128))}
        Notes:
            p_{ik} = \frac{\exp( - \frac{||x^{(i)} - m_k||^2}{s_k^2})}{\sum_j \exp( - \frac{||x^{(i)} - m_j||^2}{s_j^2})}
        """
        y = torch.norm(x - m, dim=1)
        if s is not None:
            y = y / s
        y = - y**2
        y = self._softmax(y)
        return y

    def forward(self, x):
        """
        Params:
            x:    {tensor(N, n_features(128))}
        Returns:
            loss: {tensor(1)}
        Notes:
        -   p_{ik} = \frac{\exp( - \frac{||x^{(i)} - m_k||^2}{s_k^2})}{\sum_j \exp( - \frac{||x^{(i)} - m_j||^2}{s_j^2})}
        -   entropy^{(i)}  = - \sum_k p_{ik} \log p_{ik}
        -   inter = \frac{1}{N} \sum_i entropy^{(i)}
        """
        ## 类内，属于各类别的概率的熵，求极小
        intra = map(lambda x: self._f(x, self.m, self.s1).unsqueeze(0), x) # [tensor(n_classes), ..., tensor(n_classes)]
        intra = torch.cat(list(intra), dim=0)                                               # P_{N × n_classes} = [p_{ik}]
        intra = torch.cat(list(map(lambda x: self._entropy(x).unsqueeze(0), intra)), dim=0) # ent_i = \sum_k p_{ik} \log p_{ik}
        intra = torch.mean(intra)                                                           # ent   = \frac{1}{N} \sum_i ent_i

        ## 类间
        if self.use_entropy:
            ## 各类别中心到中心均值，计算为熵，求极大
            inter = self._f(torch.mean(self.m, dim=0), self.m, self.s2)
            inter = self._entropy(inter)
        else:
            ## 类间离差阵的迹，求极大
            inter = self.m - torch.mean(self.m, dim=0)                                      # M_{n_classes × n_features}
            inter = torch.trace(torch.mm(inter.transpose(1, 0), inter) / inter.shape[0])    # \text{Trace} M = \text{trace} \frac{M^T M}{K}

        ## 优化目标，最小化
        total = intra / inter
        return total
