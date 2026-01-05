import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *


def get_logits_mask(batch_size, add_dim=True):
    if add_dim:
        logits_mask = (~torch.eye(batch_size, dtype=torch.bool).view(batch_size, batch_size, 1)).float().to(DEVICE)
    else:
        logits_mask = (~torch.eye(batch_size, dtype=torch.bool)).float().to(DEVICE)
    return logits_mask


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits


def compute_cross_entropy_reduce_sum(p, q):
    q = F.log_softmax(q, dim=1)
    loss = - torch.sum(p * q)
    return loss


def compute_cross_entropy_reduce_mean(p, q):
    q = F.log_softmax(q, dim=1)
    loss = torch.sum(p * q, dim=1)
    return - loss.mean()


class MultiLabelSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temp=TEMPERATURE, reg_batch_size=CONTRASTIVE_BATCH_SIZE, labels_num=CUR_LABELS_NUM, use_newer_loss=USE_NEWER_CONTRASTIVE_LOSS, c=OLD_LOSS_C):
        super(MultiLabelSupervisedContrastiveLoss, self).__init__()
        self.temp = temp
        self.labels_num = labels_num
        self.reg_batch_size = reg_batch_size

        if use_newer_loss:
            self.reg_logits_mask = get_logits_mask(reg_batch_size, add_dim=True)
            self.loss_func = self.multi_label_supervised_contrastive_loss_new
        else:
            self.reg_logits_mask = get_logits_mask(reg_batch_size, add_dim=False)
            self.c = c
            self.loss_func = self.multi_label_supervised_contrastive_loss_old

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        return self.loss_func(features, labels)

    def multi_label_supervised_contrastive_loss_new(self, features: torch.Tensor, labels: torch.Tensor):
        to_mul = torch.sum(labels).float() ** -1
        batch_size = features.shape[0]

        if batch_size != self.reg_batch_size:
            logits_mask = get_logits_mask(batch_size, add_dim=True)
        else:
            logits_mask = self.reg_logits_mask

        features = F.normalize(features, dim=-1, p=2)
        logits = (torch.matmul(features, features.T) / self.temp).view(batch_size, batch_size, 1)
        logits = logits - (1 - logits_mask) * 1e12
        logits = stablize_logits(logits).expand(-1, -1, self.labels_num)
        # logits = stablize_logits(logits)

        mask = labels.view(1, batch_size, self.labels_num) * labels.view(batch_size, 1, self.labels_num)
        mask = mask * logits_mask
        p = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        loss = compute_cross_entropy_reduce_sum(p, logits)
        loss = to_mul * loss
        return loss

    def multi_label_supervised_contrastive_loss_old(self, features: torch.Tensor, labels: torch.Tensor):
        batch_size = features.shape[0]

        if batch_size != self.reg_batch_size:
            logits_mask = get_logits_mask(batch_size, add_dim=False)
        else:
            logits_mask = self.reg_logits_mask

        features = F.normalize(features, dim=-1, p=2)
        logits = torch.matmul(features, features.T) / self.temp
        logits = logits - (1 - logits_mask) * 1e12
        logits = stablize_logits(logits)

        labels = labels.bool()
        labels_inter = labels.view(1, batch_size, self.labels_num) & labels.view(batch_size, 1, self.labels_num)
        labels_union = labels.view(1, batch_size, self.labels_num) | labels.view(batch_size, 1, self.labels_num)
        sim_min = torch.sum(labels_inter, dim=2)
        sim_max = torch.sum(labels_union, dim=2).clamp(min=1.0)
        sim_mat = (sim_min / sim_max) * logits_mask
        th_mask = sim_mat >= self.c
        sim_mat = sim_mat * th_mask
        p = sim_mat / th_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        loss = compute_cross_entropy_reduce_mean(p, logits)
        return loss
