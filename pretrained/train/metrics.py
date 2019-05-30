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
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine, label):

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
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

        