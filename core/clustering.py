import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sentence_transformers import models, SentenceTransformer
from sklearn.cluster import KMeans, kmeans_plusplus, AgglomerativeClustering
import numpy as np
import random
random.seed(42)


def put_features_in_device(input_features, device):
    for key in input_features.keys():
        if isinstance(input_features[key], Tensor):
            input_features[key] = input_features[key].to(device)


# Both gt and a are n x k matrix
def get_nmi_loss(a, c, para_labels, device, from_attn=True):
    n = len(para_labels)
    unique_labels = list(set(para_labels))
    k = len(unique_labels)
    gt = torch.zeros(n, k, device=device)
    for i in range(n):
        gt[i][unique_labels.index(para_labels[i])] = 1.0
    G = torch.sum(gt, dim=0)
    C = torch.sum(a, dim=0)
    U = torch.matmul(gt.T, a)
    n = torch.sum(a)
    GxC = torch.outer(G, C)
    mi = torch.sum((U / n) * torch.log(n * U / GxC))
    nmi = 2 * mi / (-torch.sum(G * torch.log(G / n) / n) - torch.sum(C * torch.log(C / n) / n))
    return -nmi


def get_weighted_adj_rand_loss(a, c, para_labels, device, from_attn=True):
    n = len(para_labels)
    unique_labels = list(set(para_labels))
    gt = torch.zeros(n, n, device=device)
    gt_weights = torch.ones(n, n, device=device)
    para_label_freq = {k: para_labels.count(k) for k in unique_labels}
    for i in range(n):
        for j in range(n):
            if para_labels[i] == para_labels[j]:
                gt[i][j] = 1.0
                gt_weights[i][j] = para_label_freq[para_labels[i]]
    if from_attn:
        dist_mat = torch.cdist(a, a)
    else:
        dist_mat = torch.cdist(torch.matmul(a, c), torch.matmul(a, c))
    sim_mat = 1 / (1 + dist_mat)
    loss = torch.sum(((gt - sim_mat) ** 2) * gt_weights) / gt.shape[0]
    #loss = torch.sum(gt * dist_mat) / gt.shape[0] ** 2 - torch.sum((1 - gt) * dist_mat) / gt.shape[0] ** 2
    return loss


def get_adj_rand_loss(a, c, para_labels, device, from_attn=True):
    n = len(para_labels)
    unique_labels = list(set(para_labels))
    GT = torch.zeros(n, len(unique_labels), device=device)
    for i in range(n):
        k = unique_labels.index(para_labels[i])
        GT[i][k] = 1.0
    M = torch.matmul(GT.T, a)
    g = torch.sum(GT, dim=0)
    c = torch.sum(a, dim=0)
    rand = 0.5 * torch.sum(M**2 - M)
    expected = 0.5 * torch.sum(g**2 - g) * torch.sum(c**2 - c) / (n**2 - n)
    maximum = 0.25 * (torch.sum(g**2 - g) + torch.sum(c**2 - c))
    ari = (rand - expected) / (maximum - expected)
    loss = -ari
    return loss


def get_rand_loss(a, c, para_labels, device, from_attn=True):
    n = len(para_labels)
    unique_labels = list(set(para_labels))
    gt = torch.zeros(n, n, device=device)
    for i in range(n):
        for j in range(n):
            if para_labels[i] == para_labels[j]:
                gt[i][j] = 1.0
    if from_attn:
        dist_mat = torch.cdist(a, a)
    else:
        dist_mat = torch.cdist(torch.matmul(a, c), torch.matmul(a, c))
    sim_mat = 2 / (1 + torch.exp(dist_mat))
    loss = torch.sum((gt - sim_mat) ** 2) / n ** 2
    # loss = torch.sum(gt * dist_mat) / gt.shape[0] ** 2 - torch.sum((1 - gt) * dist_mat) / gt.shape[0] ** 2
    return loss