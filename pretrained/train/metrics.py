import math
import torch
import torch.nn as nn


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
        phi = cosine * self.cos_m - sine * self.sin_m                           #    phi = cos(cosine + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)                          # cosine = cos(cosine + m)      if cosine > 0
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)    # cosine = cos(cosine + m)      if cosine > cos(m) else
                                                                                # cosine = cosine - m * sin(m)

        one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)                   # cosine[y_i, i] = phi
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


class MobileFacenetClusterLoss(nn.Module):

    def __init__(self):
        ...
    
    def forward(self, pred, gt):
        """
        Params:
            pred: {tensor(N, n_features(128))}
            gt:   {tensor(N)}
        Returns:
            loss: {tensor(1)}
        """
        ...