import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import math


def knn_points_normals(x, k1, k2, normal_metric_W=1., normal=False):
    """
    The idea is to design the distance metric for computing
    nearest neighbors such that the normals are not given
    too much importance while computing the distances.
    Note that this is only used in the first layer.
    """
    batch_size = x.shape[0]
    if not normal:
        indices = np.arange(0, k2, k2 // k1)
    else:
        indices = np.arange(0, k2)
        y = np.linspace(0., 3., k2)
        p_n = np.exp(-y ** 2 / 2) / (math.sqrt(2*math.pi)) * 2
        p_n = p_n / p_n.sum()
        indices = np.random.choice(indices, k1, p=p_n, replace=False)
    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            p = x[b: b + 1, 0:3]
            n = x[b: b + 1, 3:6]

            inner = 2 * torch.matmul(p.transpose(2, 1), p)
            xx = torch.sum(p ** 2, dim=1, keepdim=True)
            p_pairwise_distance = xx - inner + xx.transpose(2, 1)

            inner = 2 * torch.matmul(n.transpose(2, 1), n)
            n_pairwise_distance = 2 - inner

            # This pays less attention to normals
            pairwise_distance = p_pairwise_distance * (1 + n_pairwise_distance * normal_metric_W)

            # This pays more attention to normals
            # pairwise_distance = p_pairwise_distance * torch.exp(n_pairwise_distance)

            # pays too much attention to normals
            # pairwise_distance = p_pairwise_distance + n_pairwise_distance

            distances.append(-pairwise_distance)

        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        try:
            idx = distances.topk(k=k2, dim=-1)[1][:, :, indices]
        except:
            import ipdb
            ipdb.set_trace()
    return idx


def get_graph_feature_with_normals(x, k1=20, k2=20, idx=None, normal_metric_W=1., Norm_sample=False):
    """
    normals are treated separtely for computing the nearest neighbor
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.contiguous().view(batch_size, -1, num_points)

    if idx is None:
        idx = knn_points_normals(x, k1=k1, k2=k2, normal_metric_W=normal_metric_W, normal=Norm_sample).contiguous()

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).contiguous().view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    try:
        feature = x.view(batch_size * num_points, -1)[idx, :].contiguous()
    except:
        import ipdb
        ipdb.set_trace()
        print(feature.shape)

    feature = feature.view(batch_size, num_points, k1, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


def knn(x, k1, k2):
    batch_size = x.shape[0]
    indices = np.arange(0, k2, k2 // k1)
    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            inner = -2 * torch.matmul(x[b:b + 1].transpose(2, 1), x[b:b + 1])
            xx = torch.sum(x[b:b + 1] ** 2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            distances.append(pairwise_distance)
        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        try:
            idx = distances.topk(k=k2, dim=-1)[1][:, :, indices]
        except:
            import ipdb;
            ipdb.set_trace()
    return idx


def get_graph_feature(x, k1=20, k2=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k1=k1, k2=k2)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    # print("idx_base shape: ", idx_base.shape)


    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
    except:
        import ipdb;
        ipdb.set_trace()
        print(feature.shape)

    feature = feature.view(batch_size, num_points, k1, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature


class DGCNNEncoderGn(nn.Module):
    def __init__(self, mode=0, input_channels=3, nn_nb=80):
        super(DGCNNEncoderGn, self).__init__()
        self.k = nn_nb
        self.dilation_factor = 1
        self.mode = mode
        self.drop = 0.0
        if self.mode == 0 or self.mode == 5:
            self.bn1 = nn.GroupNorm(2, 64)
            self.bn2 = nn.GroupNorm(2, 64)
            self.bn3 = nn.GroupNorm(2, 128)
            self.bn4 = nn.GroupNorm(4, 256)
            self.bn5 = nn.GroupNorm(8, 1024)

            self.conv1 = nn.Sequential(nn.Conv2d(input_channels * 2, 64, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                       self.bn2,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                       self.bn3,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.mlp1 = nn.Conv1d(256, 1024, 1)
            self.bnmlp1 = nn.GroupNorm(8, 1024)
            self.mlp1 = nn.Conv1d(256, 1024, 1)
            self.bnmlp1 = nn.GroupNorm(8, 1024)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.shape[2]

        if self.mode == 0 or self.mode == 1:
            # print("x: ", x.shape)
            # First edge conv
            x = get_graph_feature(x, k1=self.k, k2=self.k)

            # print("x after get_graph_feature: ", x.shape)
            x = self.conv1(x)
            x1 = x.max(dim=-1, keepdim=False)[0]

            # Second edge conv
            x = get_graph_feature(x1, k1=self.k, k2=self.k)
            x = self.conv2(x)
            x2 = x.max(dim=-1, keepdim=False)[0]

            # Third edge conv
            x = get_graph_feature(x2, k1=self.k, k2=self.k)
            x = self.conv3(x)
            x3 = x.max(dim=-1, keepdim=False)[0]

            x_features = torch.cat((x1, x2, x3), dim=1)
            x = F.relu(self.bnmlp1(self.mlp1(x_features)))

            x4 = x.max(dim=2)[0]

            return x4, x_features

        if self.mode == 5:
            # First edge conv
            x = get_graph_feature_with_normals(x, k1=self.k, k2=self.k)
            x = self.conv1(x)
            x1 = x.max(dim=-1, keepdim=False)[0]

            # Second edge conv
            x = get_graph_feature(x1, k1=self.k, k2=self.k)
            x = self.conv2(x)
            x2 = x.max(dim=-1, keepdim=False)[0]

            # Third edge conv
            x = get_graph_feature(x2, k1=self.k, k2=self.k)
            x = self.conv3(x)
            x3 = x.max(dim=-1, keepdim=False)[0]

            x_features = torch.cat((x1, x2, x3), dim=1)
            x = F.relu(self.bnmlp1(self.mlp1(x_features)))
            x4 = x.max(dim=2)[0]

            return x4, x_features


class PrimitivesEmbeddingDGCNGn(nn.Module):
    """
    Segmentation model that takes point cloud as input and returns per
    point embedding or membership function. This defines the membership loss
    inside the forward function so that data distributed loss can be made faster.
    """

    def __init__(self, emb_size=128, num_primitives=10, primitives=True, embedding=True, mode=0, num_channels=3,
                 loss_function=None, nn_nb=80):
        super(PrimitivesEmbeddingDGCNGn, self).__init__()
        self.mode = mode
        self.encoder = DGCNNEncoderGn(mode=mode, input_channels=num_channels, nn_nb=nn_nb)
        self.drop = 0.0
        self.loss_function = loss_function

        if self.mode == 0 or self.mode == 3 or self.mode == 4 or self.mode == 5 or self.mode == 6:
            self.conv1 = torch.nn.Conv1d(1024 + 256, 512, 1)
        elif self.mode == 1 or self.mode == 2:
            self.conv1 = torch.nn.Conv1d(1024 + 512, 512, 1)

        self.bn1 = nn.GroupNorm(8, 512)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)

        self.bn2 = nn.GroupNorm(4, 256)

        self.softmax = torch.nn.Softmax(dim=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.tanh = torch.nn.Tanh()
        self.emb_size = emb_size
        self.primitives = primitives
        self.embedding = embedding

        if self.embedding:
            self.mlp_seg_prob1 = torch.nn.Conv1d(256, 256, 1)
            self.mlp_seg_prob2 = torch.nn.Conv1d(256, self.emb_size, 1)
            self.bn_seg_prob1 = nn.GroupNorm(4, 256)

        if primitives:
            self.mlp_prim_prob1 = torch.nn.Conv1d(256, 256, 1)
            self.mlp_prim_prob2 = torch.nn.Conv1d(256, num_primitives, 1)
            self.bn_prim_prob1 = nn.GroupNorm(4, 256)

    def forward(self, points, labels=None, compute_loss=False):
        """
        points: [bs, 3, n_point]
        """
        batch_size = points.shape[0]
        num_points = points.shape[2]
        x, first_layer_features = self.encoder(points)

        # first_layer_features = first_layer_features[:, :, self.l_permute]
        x = x.view(batch_size, 1024, 1).repeat(1, 1, num_points)
        x = torch.cat([x, first_layer_features], 1)

        x = F.dropout(F.relu(self.bn1(self.conv1(x))), self.drop)
        x_all = F.dropout(F.relu(self.bn2(self.conv2(x))), self.drop)

        # print("x_all: ", x_all.shape)
        # print("x: ", x.shape)

        if self.embedding:
            x = F.dropout(F.relu(self.bn_seg_prob1(self.mlp_seg_prob1(x_all))), self.drop)
            embedding = self.mlp_seg_prob2(x)

        if self.primitives:
            x = F.dropout(F.relu(self.bn_prim_prob1(self.mlp_prim_prob1(x_all))), self.drop)
            x = self.mlp_prim_prob2(x)
            primitives_log_prob = self.logsoftmax(x)
        if compute_loss:
            embed_loss = self.loss_function(embedding, labels.data.cpu().numpy())
        else:
            embed_loss = torch.zeros(1).cuda()
        return embedding, primitives_log_prob, embed_loss


if __name__ == '__main__':
    anet = PrimitivesEmbeddingDGCNGn().cuda()
    atensor = torch.rand(5, 3, 1000).cuda()

    ares = anet(atensor)
    print(ares[0].size(), ares[1].size())




#################### 以下为训练代码
"""
This scrip trains model to predict per point primitive type.
"""
import json
import logging
import os
import sys
from shutil import copyfile
import wandb

import numpy as np
import torch.optim as optim
import torch.utils.data
# from tensorboard_logger import configure, log_value
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from train_parsenet_e2e import test_seg_iou
from utils.read_parsenet_config import Config
from src.PointNet import PrimitivesEmbeddingDGCNGn
from src.dataset import generator_iter
from src.dataset_segments import Dataset
from src.segment_loss import (
    EmbeddingLoss,
    evaluate_miou,
    primitive_loss
)
from src.residual_utils import Evaluation

os.environ['WANDB_BASE_URL'] = "https://api.bandw.top"

default_config = "configs/config_parsenet.yml"
config = Config(default_config)
model_name = config.model_path.format(
    config.batch_size,
    config.lr,
    config.num_train,
    config.num_test,
    config.loss_weight,
    config.mode,
)
print(model_name)
wandb.init(
    project="cstnet2",
    name="train_parsenet",
    config={},
)

wandb.define_metric("train/*", step_metric="train_step")
wandb.define_metric("val/*", step_metric="val_step")

if_normals = config.normals
if_normal_noise = True

Loss = EmbeddingLoss(margin=1.0, if_mean_shift=False)
if config.mode == 0:
    # Just using points for training
    model = PrimitivesEmbeddingDGCNGn(
        embedding=True,
        emb_size=128,
        primitives=True,
        num_primitives=10,
        loss_function=Loss.triplet_loss,
        mode=config.mode,
        num_channels=3,
    )
elif config.mode == 5:
    # Using points and normals for training
    model = PrimitivesEmbeddingDGCNGn(
        embedding=True,
        emb_size=128,
        primitives=True,
        num_primitives=10,
        loss_function=Loss.triplet_loss,
        mode=config.mode,
        num_channels=6,
    )


model_bkp = model
model_bkp.l_permute = np.arange(7000)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.cuda()

evaluation = Evaluation()

alt_gpu = 0
lamb = 0.1

if config.preload_model:
    model.load_state_dict(torch.load(config.pretrain_model_path))
    print("Loading pretrained model from: {}".format(config.pretrain_model_path))

split_dict = {"train": config.num_train, "val": config.num_val, "test": config.num_test}

dataset = Dataset(
    config.batch_size,
    config.num_train,
    config.num_val,
    config.num_test,
    primitives=True,
    normals=True,
)

get_train_data = dataset.get_train(
    randomize=True, augment=True, align_canonical=True, anisotropic=False, if_normal_noise=if_normal_noise
)
get_val_data = dataset.get_val(align_canonical=True, anisotropic=False, if_normal_noise=if_normal_noise)
optimizer = optim.Adam(model.parameters(), lr=config.lr)

loader = generator_iter(get_train_data, int(1e10))
get_train_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=2,
        pin_memory=False,
    )
)

loader = generator_iter(get_val_data, int(1e10))
get_val_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=2,
        pin_memory=False,
    )
)

scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4, verbose=True, min_lr=1e-4
)

model_bkp.triplet_loss = Loss.triplet_loss
prev_test_loss = 1e4

for e in range(config.epochs):
    train_emb_losses = []
    train_prim_losses = []
    train_iou = []
    train_losses = []
    train_seg_iou = []
    model.train()

    # this is used for gradient accumulation because of small gpu memory.
    num_iter = 3
    for train_b_id in range(config.num_train // config.batch_size):
        optimizer.zero_grad()
        losses = 0
        ious = 0
        p_losses = 0
        embed_losses = 0
        torch.cuda.empty_cache()
        for _ in range(num_iter):
            points, labels, normals, primitives = next(get_train_data)[0]
            l = np.arange(10000)
            np.random.shuffle(l)
            # randomly sub-sampling points to increase robustness to density and
            # saving gpu memory
            rand_num_points = 10000
            l = l[0:rand_num_points]
            points = points[:, l]
            labels = labels[:, l]
            normals = normals[:, l]
            primitives = primitives[:, l]
            points = torch.from_numpy(points).cuda()
            normals = torch.from_numpy(normals).cuda()

            # 从点云中获取逐点特征，以及基元类型的 log_softmax
            primitives = torch.from_numpy(primitives.astype(np.int64)).cuda()
            if if_normals:
                input = torch.cat([points, normals], 2)
                embedding, primitives_log_prob, embed_loss = model(
                    input.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
                )
            else:
                embedding, primitives_log_prob, embed_loss = model(
                    points.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
                )
            embed_loss = torch.mean(embed_loss)

            p_loss = primitive_loss(primitives_log_prob, primitives)
            iou = evaluate_miou(
                primitives.data.cpu().numpy(),
                primitives_log_prob.permute(0, 2, 1).data.cpu().numpy(),
            )
            loss = embed_loss + p_loss
            loss.backward()

            losses += loss.data.cpu().numpy() / num_iter
            p_losses += p_loss.data.cpu().numpy() / num_iter
            ious += iou / num_iter
            embed_losses += embed_loss.data.cpu().numpy() / num_iter

        optimizer.step()
        train_iou.append(ious)
        train_losses.append(losses)
        train_prim_losses.append(p_losses)
        train_emb_losses.append(embed_losses)
        wandb.log({
            "iou": iou,
            "prim_loss": p_losses,
            "emb_loss": embed_losses,
            "train_step": train_b_id + e * (config.num_train // config.batch_size)
        })

    test_emb_losses = []
    test_prim_losses = []
    test_losses = []
    test_iou = []
    model.eval()

    for val_b_id in range(config.num_test // config.batch_size - 1):
        points, labels, normals, primitives = next(get_val_data)[0]
        l = np.arange(10000)
        np.random.shuffle(l)
        l = l[0:10000]
        points = points[:, l]
        labels = labels[:, l]
        normals = normals[:, l]
        primitives_ = primitives[:, l]
        points = torch.from_numpy(points).cuda()
        primitives = torch.from_numpy(primitives_.astype(np.int64)).cuda()
        normals = torch.from_numpy(normals).cuda()
        with torch.no_grad():
            if if_normals:
                input = torch.cat([points, normals], 2)
                embedding, primitives_log_prob, embed_loss = model(
                    input.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
                )
            else:
                embedding, primitives_log_prob, embed_loss = model(
                    points.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
                )

            res_loss, _ = evaluation.fitting_loss(
                embedding.permute(0, 2, 1).to(torch.device("cuda:{}".format(alt_gpu))),
                points.to(torch.device("cuda:{}".format(alt_gpu))),
                normals.to(torch.device("cuda:{}".format(alt_gpu))),
                labels,
                primitives_,
                primitives_log_prob.to(torch.device("cuda:{}".format(alt_gpu))),
                quantile=0.025,
                iterations=10,
                lamb=1.0,
                debug=False,
                eval=True,
            )

        embed_loss = torch.mean(embed_loss)
        p_loss = primitive_loss(primitives_log_prob, primitives)
        res_loss[0] = res_loss[0].to(torch.device("cuda:0"))
        loss = embed_loss + p_loss
        iou = evaluate_miou(
            primitives.data.cpu().numpy(),
            primitives_log_prob.permute(0, 2, 1).data.cpu().numpy(),
        )
        s_iou = res_loss[3:4]
        test_seg_iou.append(s_iou)
        test_iou.append(iou)
        test_prim_losses.append(p_loss.data.cpu().numpy())
        test_emb_losses.append(embed_loss.data.cpu().numpy())
        test_losses.append(loss.data.cpu().numpy())
    torch.cuda.empty_cache()

    wandb.log({
        "train loss": np.mean(train_losses),
        "test loss": np.mean(test_losses),
        "train prim loss": np.mean(train_prim_losses),
        "test prim loss": np.mean(test_prim_losses),
        "train emb loss": np.mean(train_emb_losses),
        "test emb loss": np.mean(test_emb_losses),
        "train iou": np.mean(train_iou),
        "test iou": np.mean(test_iou),
        "seg iou": np.mean(test_seg_iou),
        "val_step": e
    })

    scheduler.step(np.mean(test_emb_losses))
    # if prev_test_loss > np.mean(test_emb_losses):
    # logger.info("improvement, saving model at epoch: {}".format(e))
    prev_test_loss = np.mean(test_emb_losses)
    torch.save(
        model.state_dict(),
        "logs/trained_models/{}.pth".format(model_name),
    )
    torch.save(
        optimizer.state_dict(),
        "logs/trained_models/{}_optimizer.pth".format(model_name),
    )




