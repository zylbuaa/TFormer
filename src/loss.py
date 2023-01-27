# -*- coding： utf-8 -*-
'''
@Time: 2022/5/12 16:05
@Author:YilanZhang
@Filename:loss.py
@Software:PyCharm
@Email:zhangyilan@buaa.edu.cn
'''
import torch
import numpy as np
import math
import torch.nn.functional as F
import torch.nn as nn

'''Multi Weighted New loss'''
def get_config():
    config = {
        "num_class_list":[1038,854,211,785,266,223],
        "device": torch.device("cuda:0"),
        "WEIGHT_POWER":1.2,
        "EXTRA_WEIGHT":[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "LOSS_TYPE":'MWNLoss',
        "SCHEDULER":'cls',
        "DRW_EPOCH": 50,
        "CLS_EPOCH_MIN": 20,
        "CLS_EPOCH_MAX": 60,
        "BETA":0.1,
        "TYPE":"fix",
        "SIGMOID":'enlarge',
        "GAMMA":2.0

    }

    return config

class BaseLoss(nn.Module):
    def __init__(self, para_dict=None):
        super(BaseLoss, self).__init__()
        self.num_class_list = np.array(para_dict["num_class_list"])
        self.no_of_class = len(self.num_class_list)
        self.device = para_dict["device"]

        self.class_weight_power = para_dict["WEIGHT_POWER"]
        self.class_extra_weight = np.array(para_dict["EXTRA_WEIGHT"])
        self.scheduler = para_dict["SCHEDULER"]
        self.drw_epoch = para_dict["DRW_EPOCH"]
        self.cls_epoch_min = para_dict["CLS_EPOCH_MIN"]
        self.cls_epoch_max = para_dict["CLS_EPOCH_MAX"]
        self.weight = None

    def reset_epoch(self, epoch):
        if self.scheduler == "default":     # the weights of all classes are "1.0"
            per_cls_weights = np.array([1.0] * self.no_of_class)
        elif self.scheduler == "re_weight":
            per_cls_weights = 1.0 / (self.num_class_list.astype(np.float64))
            per_cls_weights = per_cls_weights * self.class_extra_weight
            per_cls_weights = [math.pow(num, self.class_weight_power) for num in per_cls_weights]
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * self.no_of_class
        elif self.scheduler == "drw":       # two-stage strategy using re-weighting at the second stage
            if epoch < self.drw_epoch:
                per_cls_weights = np.array([1.0] * self.no_of_class)
            else:
                per_cls_weights = 1.0 / (self.num_class_list.astype(np.float64))
                per_cls_weights = per_cls_weights * self.class_extra_weight
                per_cls_weights = [math.pow(num, self.class_weight_power) for num in per_cls_weights]
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * self.no_of_class
        elif self.scheduler == "cls":       # cumulative learning strategy
            if epoch <= self.cls_epoch_min:
                now_power = 0
            elif epoch < self.cls_epoch_max:
                now_power = ((epoch - self.cls_epoch_min) / (self.cls_epoch_max - self.cls_epoch_min)) ** 2
                now_power = now_power * self.class_weight_power
            else:
                now_power = self.class_weight_power

            per_cls_weights = 1.0 / (self.num_class_list.astype(np.float64))
            per_cls_weights = per_cls_weights * self.class_extra_weight
            per_cls_weights = [math.pow(num, now_power) for num in per_cls_weights]
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * self.no_of_class
        else:
            raise AttributeError(
                "loss scheduler can only be 'default', 're_weight', 'drw' and 'cls'.")

        # print("class weight of loss: {}".format(per_cls_weights))
        self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

class MWNLoss(BaseLoss):
    """
    Multi Weighted New loss
    Args:
        gamma (float): the hyper-parameter of focal loss
        beta (float, 0.0 - 0.4):
        type: "zero", "fix", "decrease"
        sigmoid: "normal", "enlarge"
    """
    def __init__(self, para_dict=None):
        super(MWNLoss, self).__init__(para_dict)

        self.gamma = para_dict["GAMMA"]
        self.beta = para_dict["BETA"]
        self.type = para_dict["TYPE"]
        self.sigmoid = para_dict["SIGMOID"]
        if self.beta > 0.4 or self.beta < 0.0:
            raise AttributeError(
                "For MWNLoss, the value of beta must be between 0.0 and 0.0 .")

    def forward(self, x, target):
        labels_one_hot = F.one_hot(target, self.no_of_class).float().to(self.device)
        weights = self.weight
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_class)

        loss = F.binary_cross_entropy_with_logits(input=x, target=labels_one_hot, reduction="none")

        if self.beta > 0.0:
            th = - math.log(self.beta)
            if self.type == "zero":
                other = torch.zeros(loss.shape).to(self.device)
                loss = torch.where(loss <= th, loss, other)
            elif self.type == "fix":
                other = torch.ones(loss.shape).to(self.device)
                other = other * th
                loss = torch.where(loss <= th, loss, other)
            elif self.type == "decrease":
                pt = torch.exp(-1.0 * loss)
                loss = torch.where(loss <= th, loss, pt * th / self.beta)
        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels_one_hot * x
                                  - self.gamma * torch.log(1 + torch.exp(-1.0 * x)))

        loss = modulator * loss

        weighted_loss = weights * loss
        if self.sigmoid == "enlarge":
            weighted_loss = torch.mean(weighted_loss) * 30
        else:
            weighted_loss = weighted_loss.sum() / weights.sum()
        return weighted_loss

'''Focal Loss'''
class FocalLoss(nn.Module):
    def __init__(self, class_num=3, alpha=None, gamma=2, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()
        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average
    def forward(self, pred, target): #TODO
        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0)
        target_ = torch.zeros(target.size(0),self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)
        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        batch_loss = batch_loss.sum(dim=1)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss