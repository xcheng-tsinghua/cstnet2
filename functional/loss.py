from __future__ import annotations

import torch.nn.functional as F
import torch
import numpy as np


def _contingency_matrix(y_true_idx, y_pred_idx):
    n_true = int(y_true_idx.max().item()) + 1
    n_pred = int(y_pred_idx.max().item()) + 1
    mat = torch.zeros((n_true, n_pred), device=y_true_idx.device, dtype=torch.float32)
    ones = torch.ones_like(y_true_idx, dtype=torch.float32)
    mat.index_put_((y_true_idx, y_pred_idx), ones, accumulate=True)
    return mat


def _ari_from_contingency(cont):
    n = cont.sum()
    if n <= 1:
        return torch.tensor(0.0, device=cont.device)
    a = cont.sum(dim=1)
    b = cont.sum(dim=0)
    comb_c = (cont * (cont - 1.0) * 0.5).sum()
    comb_a = (a * (a - 1.0) * 0.5).sum()
    comb_b = (b * (b - 1.0) * 0.5).sum()
    comb_n = n * (n - 1.0) * 0.5
    expected = comb_a * comb_b / (comb_n + 1e-12)
    max_index = 0.5 * (comb_a + comb_b)
    return (comb_c - expected) / (max_index - expected + 1e-12)


def _nmi_from_contingency(cont):
    n = cont.sum()
    if n <= 0:
        return torch.tensor(0.0, device=cont.device)
    p_ij = cont / n
    p_i = p_ij.sum(dim=1, keepdim=True)
    p_j = p_ij.sum(dim=0, keepdim=True)
    outer = p_i @ p_j
    valid = p_ij > 0
    mi = (p_ij[valid] * torch.log((p_ij[valid] + 1e-12) / (outer[valid] + 1e-12))).sum()
    h_i = -(p_i[p_i > 0] * torch.log(p_i[p_i > 0] + 1e-12)).sum()
    h_j = -(p_j[p_j > 0] * torch.log(p_j[p_j > 0] + 1e-12)).sum()
    return (2.0 * mi) / (h_i + h_j + 1e-12)


def evaluate_clustering(gt_labels, point_emb):
    """
    评估聚类效果
    Args:
        gt_labels: torch.size([ns, n_point])
        point_emb: torch.size([bs, n_point, emb])
        delta_v:
    Returns:

    """
    gt_labels = gt_labels.detach().long()
    point_emb = point_emb.detach().float()
    bs = point_emb.shape[0]

    accs, nmis, aris = [], [], []

    for b in range(bs):
        emb = F.normalize(point_emb[b], dim=-1)
        gt = gt_labels[b]
        _, gt_idx = torch.unique(gt, sorted=True, return_inverse=True)
        k = int(gt_idx.max().item()) + 1
        if k <= 1:
            continue

        centers = torch.zeros((k, emb.shape[-1]), device=emb.device, dtype=emb.dtype)
        centers.index_add_(0, gt_idx, emb)
        counts = torch.bincount(gt_idx, minlength=k).to(emb.dtype).clamp_min(1.0).unsqueeze(1)
        centers = F.normalize(centers / counts, dim=-1)
        pred_idx = torch.argmax(emb @ centers.transpose(0, 1), dim=1)
        acc = (pred_idx == gt_idx).float().mean()
        cont = _contingency_matrix(gt_idx, pred_idx)
        nmi = _nmi_from_contingency(cont)
        ari = _ari_from_contingency(cont)

        accs.append(acc)
        nmis.append(nmi)
        aris.append(ari)

    if len(accs) == 0:
        z = torch.zeros((), device=point_emb.device, dtype=torch.float32)
        return z, z, z
    return torch.stack(accs).mean(), torch.stack(nmis).mean(), torch.stack(aris).mean()


# def discriminative_loss(pnt_fea, affiliate_idx,
#                         delta_v=0.4, delta_d=1.5,
#                         alpha=1.0, beta=1.2, gamma=0.001):
#     """
#     同簇靠近，否则远离
#     Args:
#         pnt_fea: torch.size([bs, n_point, emb])
#         affiliate_idx: torch.size([bs, n_point])
#         delta_v:
#         delta_d:
#         alpha:
#         beta:
#         gamma:

#     Returns:

#     """
#     bs, N, D = pnt_fea.shape
#     total_var, total_dist, total_reg = 0.0, 0.0, 0.0

#     for b in range(bs):
#         fea = pnt_fea[b]            # [N, D]
#         labels = affiliate_idx[b]  # [N]

#         unique_labels = labels.unique()
#         K = len(unique_labels)

#         if K <= 1:
#             continue

#         centers = []
#         var_loss = 0.0

#         # ---------- 类内紧凑 ----------
#         for lbl in unique_labels:
#             mask = labels == lbl
#             fea_k = fea[mask]             # [Nk, D]
#             center = fea_k.mean(dim=0)    # [D]
#             centers.append(center)

#             dist = torch.norm(fea_k - center, dim=1)
#             var_loss += torch.mean(torch.clamp(dist - delta_v, min=0.0) ** 2)

#         var_loss /= K
#         centers = torch.stack(centers)  # [K, D]

#         # ---------- 类间分离 ----------
#         dist_loss = 0.0
#         for i in range(K):
#             for j in range(i+1, K):
#                 dist_ij = torch.norm(centers[i] - centers[j])
#                 dist_loss += torch.clamp(delta_d - dist_ij, min=0.0) ** 2

#         dist_loss /= (K * (K - 1) / 2)

#         # ---------- 正则 ----------
#         reg_loss = torch.mean(torch.norm(centers, dim=1))

#         total_var += var_loss
#         total_dist += dist_loss
#         total_reg += reg_loss

#     total_var /= bs
#     total_dist /= bs
#     total_reg /= bs

#     loss = alpha * total_var + beta * total_dist + gamma * total_reg
#     return loss

def discriminative_loss(pnt_fea, affiliate_idx,
                        delta_v=0.4, delta_d=1.5,
                        alpha=1.0, beta=1.2, gamma=0.001):
    """
    同簇靠近，否则远离
    Args:
        pnt_fea: torch.size([bs, n_point, emb])
        affiliate_idx: torch.size([bs, n_point])
        delta_v:
        delta_d:
        alpha:
        beta:
        gamma:

    Returns:

    """
    bs, _, _ = pnt_fea.shape
    device = pnt_fea.device
    dtype = pnt_fea.dtype
    total_var = torch.zeros((), device=device, dtype=dtype)
    total_dist = torch.zeros((), device=device, dtype=dtype)
    total_reg = torch.zeros((), device=device, dtype=dtype)

    for b in range(bs):
        fea = pnt_fea[b]            # [N, D]
        labels = affiliate_idx[b]   # [N]
        _, inv = torch.unique(labels, sorted=True, return_inverse=True)
        K = int(inv.max().item()) + 1

        if K <= 1:
            continue

        # ---------- 类内紧凑（向量化） ----------
        centers = torch.zeros((K, fea.shape[1]), device=device, dtype=dtype)
        centers.index_add_(0, inv, fea)
        counts = torch.bincount(inv, minlength=K).to(dtype).clamp_min(1.0).unsqueeze(1)
        centers = centers / counts

        dist_per_point = torch.norm(fea - centers[inv], dim=1)
        var_per_point = torch.clamp(dist_per_point - delta_v, min=0.0) ** 2
        var_sum = torch.zeros((K,), device=device, dtype=dtype)
        var_sum.index_add_(0, inv, var_per_point)
        var_loss = (var_sum / counts.squeeze(1)).mean()

        # ---------- 类间分离（向量化） ----------
        center_dist = torch.cdist(centers, centers, p=2)
        pair_mask = torch.triu(torch.ones((K, K), device=device, dtype=torch.bool), diagonal=1)
        pair_d = center_dist[pair_mask]
        dist_loss = torch.clamp(delta_d - pair_d, min=0.0).pow(2).mean()

        # ---------- 正则 ----------
        reg_loss = torch.mean(torch.norm(centers, dim=1))

        total_var += var_loss
        total_dist += dist_loss
        total_reg += reg_loss

    total_var /= bs
    total_dist /= bs
    total_reg /= bs

    loss = alpha * total_var + beta * total_dist + gamma * total_reg
    return loss


class EmbeddingLoss:
    """
    从 parsenet 转移过来的损失函数
    """
    def __init__(self, margin=1.0, if_mean_shift=False):
        """
        Defines loss function to train embedding network.
        :param margin: margin to be used in triplet loss.
        :param if_mean_shift: bool, whether to use mean shift
        iterations. This is only used in end to end training.
        """
        self.margin = margin
        self.if_mean_shift = if_mean_shift
        self.meanshift = MeanShift()

    def triplet_loss(self, output, labels: np.ndarray, iterations=5):
        """
        Triplet loss
        :param output: output embedding from the network. size: B x 128 x N
        where B is the batch size, 128 is the dim size and N is the number of points.
        :param labels: B x N
        """
        max_segments = 5
        batch_size = output.shape[0]
        N = output.shape[2]
        loss_diff = torch.tensor([0.], requires_grad=True).cuda()
        relu = torch.nn.ReLU()

        output = output.permute(0, 2, 1)
        output = torch.nn.functional.normalize(output, p=2, dim=2)
        new_output = []

        if self.if_mean_shift:
            for b in range(batch_size):
                new_X, bw = self.meanshift.mean_shift(output[b], 4000,
                                                 0.015, iterations=iterations,
                                                 nms=False)
                new_output.append(new_X)
            output = torch.stack(new_output, 0)

        num_sample_points = {}
        sampled_points = {}
        for i in range(batch_size):
            sampled_points[i] = {}
            p = labels[i]
            # print("labels: ", labels.shape)
            unique_labels = np.unique(p)
            # print("unique_labels: ", unique_labels.shape)

            # number of points from each cluster.
            num_sample_points[i] = min([N // unique_labels.shape[0] + 1, 30])
            # print("num_sample_points: ", num_sample_points[i])
            for l in unique_labels:
                ix = np.isin(p, l)
                sampled_indices = np.where(ix)[0]
                # print("sampled_indices: ", sampled_indices.shape)
                # point indices that belong to a certain cluster.
                sampled_points[i][l] = np.random.choice(
                    list(sampled_indices),
                    num_sample_points[i],
                    replace=True)
                # print(f"sampled_points[{i}][{l}]: ", sampled_points[i][l].shape)

        sampled_predictions = {}
        for i in range(batch_size):
            sampled_predictions[i] = {}
            for k, v in sampled_points[i].items():
                pred = output[i, v, :]
                # print("pred: ", pred.shape)
                sampled_predictions[i][k] = pred

        all_satisfied = 0
        only_one_segments = 0
        for i in range(batch_size):
            len_keys = len(sampled_predictions[i].keys())
            keys = list(sorted(sampled_predictions[i].keys()))
            num_iterations = min([max_segments * max_segments, len_keys * len_keys])
            normalization = 0
            if len_keys == 1:
                only_one_segments += 1
                continue

            loss_shape = torch.tensor([0.], requires_grad=True).cuda()
            for _ in range(num_iterations):
                k1 = np.random.choice(len_keys, 1)[0]
                k2 = np.random.choice(len_keys, 1)[0]
                if k1 == k2:
                    continue
                else:
                    normalization += 1

                pred1 = sampled_predictions[i][keys[k1]]
                pred2 = sampled_predictions[i][keys[k2]]

                Anchor = pred1.unsqueeze(1)
                Pos = pred1.unsqueeze(0)
                Neg = pred2.unsqueeze(0)

                diff_pos = torch.sum(torch.pow((Anchor - Pos), 2), 2)
                diff_neg = torch.sum(torch.pow((Anchor - Neg), 2), 2)
                constraint = diff_pos - diff_neg + self.margin
                constraint = relu(constraint)

                # remove diagonals corresponding to same points in anchors
                loss = torch.sum(constraint) - constraint.trace()

                satisfied = torch.sum(constraint > 0) + 1.0
                satisfied = satisfied.type(torch.cuda.FloatTensor)

                loss_shape = loss_shape + loss / satisfied.detach()

            loss_shape = loss_shape / (normalization + 1e-8)
            loss_diff = loss_diff + loss_shape
        loss_diff = loss_diff / (batch_size - only_one_segments + 1e-8)
        return loss_diff


class MeanShift:
    def __init__(self, ):
        """
        Differentiable mean shift clustering inspired from
        https://arxiv.org/pdf/1712.08273.pdf
        """
        pass

    def mean_shift(self, X, num_samples, quantile, iterations, kernel_type="gaussian", bw=None, nms=True):
        """
        Complete function to do mean shift clutering on the input X
        :param num_samples: number of samples to consider for band width
        calculation
        :param X: input, N x d
        :param quantile: to be used for computing number of nearest
        neighbors, 0.05 works fine.
        :param iterations:
        """
        if bw == None:
            with torch.no_grad():
                bw = self.compute_bandwidth(X, num_samples, quantile)

                print("bandwidth: ", bw.item())

                # avoid numerical issues.
                bw = torch.clamp(bw, min=0.003)
        new_X, _ = self.mean_shift_(X, b=bw, iterations=iterations, kernel_type=kernel_type)
        if not nms:
            return new_X, bw

        with torch.no_grad():
            _, indices, new_labels = self.nms(new_X, X, b=bw)
        center = new_X[indices]

        return new_X, center, bw, new_labels

    def mean_shift_(self, X, b, iterations=10, kernel_type="gaussian"):
        """
        Differentiable mean shift clustering.
        X are assumed to lie on the hyper shphere, and thus are normalized
        to have unit norm. This is done for computational
        efficiency and will not work if the assumptions are voilated.
        :param X: N x d, points to be clustered
        :param b: bandwidth
        :param iterations: number of iterations
        """
        # initialize all the points as the seed points
        new_X = X.clone()
        delta = 1
        for i in range(iterations):
            if kernel_type == "gaussian":
                dist = 2.0 - 2.0 * new_X @ torch.transpose(X, 1, 0)

                # TODO Normalization is still remaining.
                K = guard_exp(- dist / (b ** 2) / 2)
            else:
                # epanechnikov
                dist = 2.0 - 2.0 * new_X @ torch.transpose(X, 1, 0)
                dist = 3 / 4 * (1 - dist / (b ** 2))
                K = torch.nn.functional.relu(dist)

            D = 1 / (torch.sum(K, 1, keepdim=True))

            # K: N x N, X: N x d, D: N x 1
            M = (K @ X) * D - new_X
            new_X = new_X + delta * M

            # re-normalize it to lie on unit hyper-sphere.
            new_X = new_X / torch.norm(new_X, dim=1, p=2, keepdim=True)
        # new_X: center of the clusters
        return new_X, X

    def guard_mean_shift(self, embedding, quantile, iterations, kernel_type="gaussian"):
        """
        Some times if band width is small, number of cluster can be larger than 50, that
        but we would like to keep max clusters 50 as it is the max number in our dataset.
        in that case you increase the quantile to increase the band width to decrease
        the number of clusters.
        """
        while True:
            _, center, bandwidth, cluster_ids = self.mean_shift(
                embedding, 5000, quantile, iterations, kernel_type=kernel_type
            )
            if torch.unique(cluster_ids).shape[0] > 49:
                quantile *= 2
            else:
                break
        return center, bandwidth, cluster_ids

    def kernel(self, X, kernel_type, bw):
        """
        Assuing that the feature vector in X are normalized.
        """
        if kernel_type == "gaussian":
            # gaussian
            dist = 2.0 - 2.0 * X @ torch.transpose(X, 1, 0)
            # TODO not considering the normalization factor
            K = guard_exp(- dist / (bw ** 2) / 2)

        elif kernel_type == "epa":
            # epanechnikov
            dist = 2.0 - 2.0 * X @ torch.transpose(X, 1, 0)
            dist = 3 / 4 * (1 - dist / (bw ** 2))
            K = torch.nn.functional.relu(dist)
        return K

    def compute_bandwidth(self, X, num_samples, quantile):
        """
        Compute the bandwidth for mean shift clustering.
        Assuming the X is normalized to lie on hypersphere.
        :param X: input data, N x d
        :param num_samples: number of samples to be used
        for computing distance, <= N
        :param quantile: nearest neighbors used for computing
        the bandwidth.
        """
        N = X.shape[0]
        L = np.arange(N)
        np.random.shuffle(L)
        X = X[L[0:num_samples]]
        # dist = (torch.unsqueeze(X, 1) - torch.unsqueeze(X, 0)) ** 2
        dist = 2 - 2 * X @ torch.transpose(X, 1, 0)
        # dist = torch.sum(dist, 1)
        K = int(quantile * num_samples)
        top_k = torch.topk(dist, k=K, dim=1, largest=False)[0]

        max_top_k = guard_sqrt(top_k[:, -1], 1e-6)

        return torch.mean(max_top_k)

    def nms(self, centers, X, b):
        """
        Non max suprression.
        :param centers: center of clusters
        :param X: points to be clustered
        :param b: band width used to get the centers
        """
        membership = 2.0 - 2.0 * centers @ torch.transpose(X, 1, 0)

        # which cluster center is closer to the points
        membership = torch.min(membership, 0)[1]

        # Find the unique clusters which is closer to at least one point
        uniques, counts_ = np.unique(membership.data.cpu().numpy(), return_counts=True)

        # count of the number of points belonging to unique cluster ids above
        counts = torch.from_numpy(counts_.astype(np.float32)).cuda(torch.get_device(centers))

        num_mem_cluster = torch.zeros((X.shape[0])).cuda(torch.get_device(centers))

        # Contains the count of number of points belonging to a
        # unique cluster
        num_mem_cluster[uniques] = counts

        # distance of clusters from each other
        dist = 2.0 - 2.0 * centers @ torch.transpose(centers, 1, 0)

        # find the nearest neighbors to each cluster based on some threshold
        # TODO this could be b ** 2
        cluster_nbrs = dist < b
        cluster_nbrs = cluster_nbrs.float()

        cluster_center_ids = torch.unique(torch.max(cluster_nbrs[uniques] * num_mem_cluster.reshape((1, -1)), 1)[1])
        # pruned centers
        centers = centers[cluster_center_ids]

        # assign labels to the input points
        # It is assumed that the embeddings lie on the hypershphere and are normalized
        temp = centers @ torch.transpose(X, 1, 0)
        labels = torch.max(temp, 0)[1]
        return centers, cluster_center_ids, labels

    def pdist(self, x, y):
        x = torch.unsqueeze(x, 1)
        y = torch.unsqueeze(y, 0)
        dist = torch.sum((x - y) ** 2, 2)
        return dist


def guard_exp(x, max_value=75, min_value=-75):
    x = torch.clamp(x, max=max_value, min=min_value)
    return torch.exp(x)


def guard_sqrt(x, minimum=1e-5):
    x = torch.clamp(x, min=minimum)
    return torch.sqrt(x)


def mse_loss_with_pmt_considered(attr_pred, attr_gt, pmt_gt, valid_pmt):
    """
    计算 attr_pred 和 attr_gt 之间的 mse_loss，只有有效类型的店才会参与计算

    :param attr_pred: [bs, point]
    :param attr_gt: [bs, point]
    :param pmt_gt: [bs, point] (int, index)
    :param valid_pmt: tuple: (1, 2, ...), (0=plane,1=cylinder,2=cone,3=sphere,4=freeform)
    :return:
    """
    # 筛选 mask：GT or 预测的基元类型是否在有效集合里
    mask = torch.isin(pmt_gt, torch.tensor(valid_pmt, device=pmt_gt.device))

    if mask.sum() > 0:

        # 只对有效类型计算 loss
        loss = F.mse_loss(attr_pred[mask], attr_gt[mask])
        return loss

    else:
        return 0.0


def unit_len_loss(attr_pred):
    """
    计算长度为1的loss
    :param attr_pred: [bs, ..., X]
    """
    # 计算每个向量的长度 (L2 norm)
    lengths = torch.norm(attr_pred, dim=-1)  # shape [bs]

    # 希望每个长度接近 1，可以用 MSE loss
    loss = F.mse_loss(lengths, torch.ones_like(lengths))
    return loss


def unit_len_loss_with_pmt_considered(attr_pred, pmt_gt, valid_pmt):
    """
    计算 attr_pred 长度与 1 之间的 loss，只有有效类型的店才会参与计算

    :param attr_pred: [bs, point, 3]
    :param pmt_gt: [bs, point] (int, index)
    :param valid_pmt: tuple: (1, 2, ...), (0=plane,1=cylinder,2=cone,3=sphere,4=freeform)
    :return:
    """
    # 筛选 mask：GT or 预测的基元类型是否在有效集合里
    mask = torch.isin(pmt_gt, torch.tensor(valid_pmt, device=pmt_gt.device))

    if mask.sum() > 0:
        valid_attr_pred = attr_pred[mask]
        loss = unit_len_loss(valid_attr_pred)
        return loss

    else:
        return 0.0


def perpendicular_loss(vec1, vec2, eps=1e-6):
    """
    计算对应位置向量垂直的 Loss
    :param vec1: [..., X]
    :param vec2: [..., X]
    :param eps: [..., X]
    """
    # 归一化（防止除零）
    a_norm = vec1 / (vec1.norm(dim=-1, keepdim=True) + eps)
    b_norm = vec2 / (vec2.norm(dim=-1, keepdim=True) + eps)

    # 点积
    dot = (a_norm * b_norm).sum(dim=-1)  # [bs, point]

    # loss: 点积平方的平均值
    loss = (dot ** 2).mean()
    return loss


def parallel_loss(vec1, vec2, eps=1e-6):
    # 归一化（防止除零）
    a_norm = vec1 / (vec1.norm(dim=-1, keepdim=True) + eps)
    b_norm = vec2 / (vec2.norm(dim=-1, keepdim=True) + eps)

    # 计算余弦相似度
    cos_sim = (a_norm * b_norm).sum(dim=-1)  # [-1, 1]

    # 平行时 |cos|=1
    loss = ((1 - cos_sim.abs()) ** 2).mean()
    return loss


def geom_loss_plane(xyz, mad_pred, nor_pred, loc_pred, pmt_gt):
    """
    :param xyz: [bs, point, 3]
    :param mad_pred: [bs, point, 3]
    :param nor_pred: [bs, point, 3]
    :param loc_pred: [bs, point, 3]
    :param pmt_gt: [bs, point] (int, index)
    """
    # 找到全部平面类型的点
    mask = (pmt_gt == 0)  # [bs, point]

    if mask.sum() > 0:

        xyz = xyz[mask]  # [n_item, 3]
        mad_pred = mad_pred[mask]  # [n_item, 3]
        nor_pred = nor_pred[mask]  # [n_item, 3]
        loc_pred = loc_pred[mask]  # [n_item, 3]

        # 点到预测平面的距离为 0，主要
        foot_to_xyz = xyz - loc_pred  # [bs, n, 3]
        nor_pred = nor_pred / (nor_pred.norm(dim=-1, keepdim=True) + 1e-8)
        dist = (foot_to_xyz * nor_pred).sum(dim=-1)
        on_plane_loss = (dist ** 2).mean()

        # 原点到垂足的向量与主方向平行，次要
        foot_pall_mad = parallel_loss(loc_pred, mad_pred)

        # 主方向和法线平行，次要
        mad_pall_nor = parallel_loss(mad_pred, nor_pred)

        return on_plane_loss + 0.2 * foot_pall_mad + 0.2 * mad_pall_nor

    else:
        return 0.0


def geom_loss_cylinder(xyz, mad_pred, nor_pred, dim_pred, loc_pred, pmt_gt):
    """
    :param xyz: [bs, point, 3]
    :param mad_pred: [bs, point, 3]
    :param nor_pred: [bs, point, 3]
    :param dim_pred: [bs, point]
    :param loc_pred: [bs, point, 3]
    :param pmt_gt: [bs, point] (int, index)
    """
    # 找到全部圆柱类型的点
    mask = (pmt_gt == 1)  # [bs, point]

    if mask.sum() > 0:

        xyz = xyz[mask]  # [n_item, 3]
        mad_pred = mad_pred[mask]  # [n_item, 3]
        nor_pred = nor_pred[mask]  # [n_item, 3]
        dim_pred = dim_pred[mask]  # [n_item, 3]
        loc_pred = loc_pred[mask]  # [n_item, 3]

        # 半径与预测主尺寸相等，即点在圆柱上
        radius = torch.cross(xyz - loc_pred, mad_pred, dim=1)
        radius = radius.norm(dim=1)
        xyz_on_cylin = (radius - dim_pred).abs().mean()

        # 原点到垂足的向量与主方向垂直
        foot_prep_mad_loss = perpendicular_loss(loc_pred, mad_pred)
        # dot_product = torch.einsum('ij, ij -> i', loc_pred, mad_pred).abs().mean()

        # 法线与主方向垂直
        mad_prep_nor = perpendicular_loss(mad_pred, nor_pred)

        return xyz_on_cylin + 0.2 * foot_prep_mad_loss + 0.2 * mad_prep_nor

    else:
        return 0.0


# def geom_loss_cone(xyz, mad_pred, dim_pred, loc_pred, pmt_gt):
#     """
#     点在圆锥上
#     :param xyz: [bs, point, 3]
#     :param mad_pred: [bs, point, 3]
#     :param dim_pred: [bs, point]
#     :param loc_pred: [bs, point, 3]
#     :param pmt_gt: [bs, point] (int, index)
#     """
#     # 找到全部圆锥类型的点
#     mask = (pmt_gt == 2)  # [bs, point]
#
#     if mask.sum() > 0:
#
#         xyz = xyz[mask]  # [n_item, 3]
#         mad_pred = mad_pred[mask]  # [n_item, 3]
#         dim_pred = dim_pred[mask]  # [n_item, ]
#         loc_pred = loc_pred[mask]  # [n_item, 3]
#
#         # 从锥角到圆锥面上的点构成的向量与主方向之间的夹角等于主尺寸
#         apex_to_xyz = xyz - loc_pred
#         dot1 = torch.einsum('ij, ij -> i', mad_pred, apex_to_xyz)
#         dot2 = mad_pred * apex_to_xyz.norm(dim=1) * torch.cos(dim_pred)
#         semi_angle = (dot1 - dot2).abs().mean()
#
#         return semi_angle
#
#     else:
#         return 0.0


def geom_loss_cone(xyz, mad_pred, dim_pred, loc_pred, pmt_gt):
    """
    点在圆锥上
    :param xyz: [bs, point, 3]
    :param mad_pred: [bs, point, 3]
    :param dim_pred: [bs, point]
    :param loc_pred: [bs, point, 3]
    :param pmt_gt: [bs, point] (int, index)
    """
    # 找到全部圆锥类型的点
    mask = (pmt_gt == 2)  # [bs, point]

    if mask.sum() > 0:

        mad_pred = mad_pred[mask]  # [n_item, 3]
        loc_pred = loc_pred[mask]  # [n_item, 3]

        # 原点到垂足的向量与主方向垂直
        foot_perp_mad = perpendicular_loss(loc_pred, mad_pred)
        # foot_perp_mad = torch.einsum('ij, ij -> i', loc_pred, mad_pred).abs().mean()

        return foot_perp_mad

    else:
        return 0.0


def geom_loss_sphere(xyz, dim_pred, nor_pred, loc_pred, pmt_gt):
    """
    :param xyz: [bs, point, 3]
    :param dim_pred: [bs, point]
    :param nor_pred: [bs, point, 3]
    :param loc_pred: [bs, point, 3]
    :param pmt_gt: [bs, point] (int, index)
    """
    # 找到全部球类型的点
    mask = (pmt_gt == 3)  # [bs, point]

    if mask.sum() > 0:

        xyz = xyz[mask]  # [n_item, 3]
        dim_pred = dim_pred[mask]  # [n_item, ]
        nor_pred = nor_pred[mask]  # [n_item, 3]
        loc_pred = loc_pred[mask]  # [n_item, 3]

        # 球面上的点到主位置的距离等于主尺寸
        center_to_xyz = xyz - loc_pred
        xyz_on_sphere_loss = (center_to_xyz.norm(dim=1) - dim_pred).abs().mean()

        # 球心到 xyz 的向量与 nor 垂直
        center_to_xyz_pall_nor_loss = parallel_loss(center_to_xyz, nor_pred)

        return xyz_on_sphere_loss + 0.2 * center_to_xyz_pall_nor_loss

    else:
        return 0.0


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _primitive_mask(pmt_gt: torch.Tensor, valid_pmt: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros_like(pmt_gt, dtype=torch.bool)
    for prim_idx in valid_pmt:
        mask = mask | (pmt_gt == prim_idx)
    return mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return values[mask].mean()
    return _zero_loss(reference)


def _masked_vector_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sign_invariant: bool = False,
    canonicalize: bool = False,
) -> torch.Tensor:
    if not mask.any():
        return _zero_loss(pred)
    pred_m = F.normalize(pred[mask], dim=-1, eps=1e-6)
    target_m = F.normalize(target[mask], dim=-1, eps=1e-6)
    if canonicalize:
        pred_m = canonicalize_vectors_hard(pred_m)
        target_m = canonicalize_vectors_hard(target_m)
    if sign_invariant:
        direct = (pred_m - target_m).pow(2).sum(dim=-1)
        flipped = (pred_m + target_m).pow(2).sum(dim=-1)
        return torch.minimum(direct, flipped).mean()
    return F.mse_loss(pred_m, target_m)


def _masked_scalar_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return _zero_loss(pred)
    return F.mse_loss(pred[mask], target[mask])


def _masked_parallel_loss(vec1: torch.Tensor, vec2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return _zero_loss(vec1)
    a = F.normalize(vec1[mask], dim=-1, eps=1e-6)
    b = F.normalize(vec2[mask], dim=-1, eps=1e-6)
    cos = (a * b).sum(dim=-1).abs()
    return (1.0 - cos).pow(2).mean()


def _masked_perpendicular_loss(vec1: torch.Tensor, vec2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return _zero_loss(vec1)
    a = F.normalize(vec1[mask], dim=-1, eps=1e-6)
    b = F.normalize(vec2[mask], dim=-1, eps=1e-6)
    return (a * b).sum(dim=-1).pow(2).mean()


def _stage1_geometry_losses(
    xyz: torch.Tensor,
    mad_pred: torch.Tensor,
    dim_pred: torch.Tensor,
    nor_pred: torch.Tensor,
    loc_pred: torch.Tensor,
    pmt_gt: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mad_pred = F.normalize(mad_pred, dim=-1, eps=1e-6)
    nor_pred = F.normalize(nor_pred, dim=-1, eps=1e-6)
    dim_pred = dim_pred.clamp_min(0.0)

    plane_mask = pmt_gt == 0
    if plane_mask.any():
        n = mad_pred
        plane_dist = ((xyz - loc_pred) * n).sum(dim=-1).pow(2)
        on_plane = _masked_mean(plane_dist, plane_mask, xyz)
        loc_nonzero = plane_mask & (loc_pred.norm(dim=-1) > 1e-4)
        loc_parallel = _masked_parallel_loss(loc_pred, mad_pred, loc_nonzero)
        mad_nor_parallel = _masked_parallel_loss(mad_pred, nor_pred, plane_mask)
        loss_plane = on_plane + 0.2 * loc_parallel + 0.2 * mad_nor_parallel
    else:
        loss_plane = _zero_loss(xyz)

    cylinder_mask = pmt_gt == 1
    if cylinder_mask.any():
        v = xyz - loc_pred
        axis = mad_pred
        axial = (v * axis).sum(dim=-1, keepdim=True) * axis
        radial = (v - axial).norm(dim=-1)
        radius_residual = (radial - dim_pred).pow(2)
        on_cylinder = _masked_mean(radius_residual, cylinder_mask, xyz)
        loc_perp_axis = _masked_perpendicular_loss(loc_pred, mad_pred, cylinder_mask)
        nor_perp_axis = _masked_perpendicular_loss(nor_pred, mad_pred, cylinder_mask)
        loss_cylinder = on_cylinder + 0.2 * loc_perp_axis + 0.2 * nor_perp_axis
    else:
        loss_cylinder = _zero_loss(xyz)

    cone_mask = pmt_gt == 2
    if cone_mask.any():
        v = xyz - loc_pred
        axis = mad_pred
        signed_axial = (v * axis).sum(dim=-1)
        axial = signed_axial.abs().clamp_min(1e-4)
        radial_vec = v - signed_axial.unsqueeze(-1) * axis
        radial = radial_vec.norm(dim=-1)
        semi_angle = dim_pred.clamp(min=1e-4, max=1.55)
        cone_residual = (radial - axial * torch.tan(semi_angle)).pow(2)
        loss_cone = _masked_mean(cone_residual, cone_mask, xyz)
    else:
        loss_cone = _zero_loss(xyz)

    sphere_mask = pmt_gt == 3
    if sphere_mask.any():
        center_to_xyz = xyz - loc_pred
        radius_residual = (center_to_xyz.norm(dim=-1) - dim_pred).pow(2)
        on_sphere = _masked_mean(radius_residual, sphere_mask, xyz)
        normal_parallel = _masked_parallel_loss(center_to_xyz, nor_pred, sphere_mask)
        loss_sphere = on_sphere + 0.2 * normal_parallel
    else:
        loss_sphere = _zero_loss(xyz)

    geom_loss = loss_plane + loss_cylinder + loss_cone + loss_sphere
    return {
        "geom_loss": geom_loss,
        "loss_plane": loss_plane,
        "loss_cylinder": loss_cylinder,
        "loss_cone": loss_cone,
        "loss_sphere": loss_sphere,
    }


def instance_consistency_loss(log_pmt_pred, mad_pred, dim_pred, loc_pred, affil_idx, pmt_gt=None):
    """
    log_pmt_pred: [B, P, 5] log-softmax后的基元类型预测
    mad_pred: [B, P, 3] 主方向预测
    dim_pred: [B, P] 尺寸预测
    loc_pred: [B, P, 3] 主位置预测
    affil_idx: [B, P] 每个点所属实例的索引 (int)
    pmt_gt: [B, P] optional primitive type labels used for valid property masks
    """
    bs = log_pmt_pred.size(0)
    terms = []
    mad_pred = canonicalize_vectors_hard(F.normalize(mad_pred, dim=-1, eps=1e-6))
    probs = log_pmt_pred.exp()

    for b in range(bs):
        # 找到无重复的实例id
        inst_ids = affil_idx[b].unique()

        for inst_id in inst_ids:
            mask = (affil_idx[b] == inst_id)  # 当前实例的点
            if mask.sum() <= 1:
                continue  # 只有1个点不计算一致性

            if pmt_gt is None:
                inst_prim = None
            else:
                inst_labels = pmt_gt[b][mask].long()
                inst_prim = int(torch.bincount(inst_labels, minlength=5).argmax().item())

            # ---- 基元类型一致性（对logits取均值，再和每个点对齐）----
            pmt_prob = probs[b][mask]   # [N, 5]
            mean_prob = pmt_prob.mean(0, keepdim=True)  # [1, 5]
            terms.append(F.mse_loss(pmt_prob, mean_prob.expand_as(pmt_prob)))

            # ---- 主方向一致性 ----
            if inst_prim is None or inst_prim in (0, 1, 2):
                mad = mad_pred[b][mask]  # [N, 3]
                mean_mad = F.normalize(mad.mean(0, keepdim=True), dim=-1, eps=1e-6)
                terms.append(F.mse_loss(mad, mean_mad.expand_as(mad)))

            # ---- 尺寸一致性 ----
            if inst_prim is None or inst_prim in (1, 2, 3):
                dim = dim_pred[b][mask]  # [N]
                mean_dim = dim.mean()
                terms.append(F.mse_loss(dim, mean_dim.expand_as(dim)))

            # ---- 主位置一致性 ----
            if inst_prim is None or inst_prim in (0, 1, 2, 3):
                loc = loc_pred[b][mask]  # [N, 3]
                mean_loc = loc.mean(0, keepdim=True)
                terms.append(F.mse_loss(loc, mean_loc.expand_as(loc)))

    if len(terms) == 0:
        return _zero_loss(log_pmt_pred)
    return torch.stack(terms).mean()


def linear_ramp(global_epoch, start_epoch, ramp_epochs):
    """Linear schedule used by all delayed Stage 1 auxiliary losses."""
    if ramp_epochs <= 0:
        return 1.0 if global_epoch >= start_epoch else 0.0
    progress = (float(global_epoch) - float(start_epoch)) / float(ramp_epochs)
    return max(0.0, min(1.0, progress))


def constraint_loss(xyz, log_pmt_pred, mad_pred, dim_pred, nor_pred, loc_pred,
                    pmt_gt, mad_gt, dim_gt, nor_gt, loc_gt, affil_idx,
                    point_emb=None, weights=None, global_epoch=None,
                    geom_start_epoch=20, geom_ramp_epochs=20,
                    enabled_losses=None, eps=1e-6):
    """
    计算损失，包含一般损失和几何损失

    点坐标
    :param xyz: [bs, point, 3]

    预测数据
    :param log_pmt_pred: [bs, point, 5]
    :param mad_pred: [bs, point, 3]
    :param dim_pred: [bs, point]
    :param nor_pred: [bs, point, 3]
    :param loc_pred: [bs, point, 3]

    标签数据
    :param pmt_gt: [bs, point] (int, index)
    :param mad_gt: [bs, point, 3]
    :param dim_gt: [bs, point]
    :param nor_gt: [bs, point, 3]
    :param loc_gt: [bs, point, 3]
    :param affil_idx: [bs, point]

    :param eps: 防止除 0 的调整实数
    """

    if global_epoch is None:
        raise ValueError("constraint_loss requires global_epoch")

    weights = {} if weights is None else weights
    enabled_losses = {} if enabled_losses is None else enabled_losses
    w_pmt = float(weights.get("w_pmt", 1.0))
    w_cluster = float(weights.get("w_cluster", 0.5))
    w_mad = float(weights.get("w_mad", 0.02))
    w_dim = float(weights.get("w_dim", 0.05))
    w_nor = float(weights.get("w_nor", 0.1))
    w_loc = float(weights.get("w_loc", 0.02))
    w_geom = float(weights.get("w_geom", 0.02))
    w_inst = float(weights.get("w_inst", 0.005))

    pmt_loss = F.nll_loss(log_pmt_pred.reshape(-1, 5), pmt_gt.reshape(-1))
    cluster_loss = discriminative_loss(point_emb, affil_idx) if point_emb is not None else _zero_loss(log_pmt_pred)

    mad_pred = F.normalize(mad_pred, dim=-1, eps=eps)
    nor_pred = F.normalize(nor_pred, dim=-1, eps=eps)
    mad_gt = F.normalize(mad_gt, dim=-1, eps=eps)
    nor_gt = F.normalize(nor_gt, dim=-1, eps=eps)

    mad_mask = _primitive_mask(pmt_gt, (0, 1, 2))
    dim_mask = _primitive_mask(pmt_gt, (1, 2, 3))
    nor_mask = _primitive_mask(pmt_gt, (0, 1, 2, 3, 4))
    loc_mask = _primitive_mask(pmt_gt, (0, 1, 2, 3))

    mad_loss = _masked_vector_mse(mad_pred, mad_gt, mad_mask, sign_invariant=False, canonicalize=True)
    dim_loss = _masked_scalar_mse(dim_pred, dim_gt, dim_mask)
    nor_loss = _masked_vector_mse(nor_pred, nor_gt, nor_mask, sign_invariant=False, canonicalize=False)
    loc_loss = _masked_scalar_mse(loc_pred, loc_gt, loc_mask)

    geom_losses = _stage1_geometry_losses(xyz, mad_pred, dim_pred, nor_pred, loc_pred, pmt_gt)
    inst_loss = instance_consistency_loss(log_pmt_pred, mad_pred, dim_pred, loc_pred, affil_idx, pmt_gt)

    aux_factor = linear_ramp(global_epoch, geom_start_epoch, geom_ramp_epochs)
    raw_losses = {
        "pmt": pmt_loss,
        "cluster": cluster_loss,
        "mad": mad_loss,
        "dim": dim_loss,
        "nor": nor_loss,
        "loc": loc_loss,
        "geom": geom_losses["geom_loss"],
        "inst": inst_loss,
    }
    target_weights = {
        "pmt": w_pmt,
        "cluster": w_cluster,
        "mad": w_mad,
        "dim": w_dim,
        "nor": w_nor,
        "loc": w_loc,
        "geom": w_geom,
        "inst": w_inst,
    }
    ramped_names = {"mad", "dim", "loc", "geom", "inst"}
    weighted_losses = {}
    effective_weights = {}
    for name, raw_loss in raw_losses.items():
        enabled = bool(enabled_losses.get(name, True))
        ramp = aux_factor if name in ramped_names else 1.0
        effective_weight = target_weights[name] * ramp if enabled else 0.0
        effective_weights[name] = effective_weight
        weighted_losses[name] = raw_loss * effective_weight

    loss_all = sum(weighted_losses.values())

    loss_dict = {
        "loss_all": loss_all,
        "pmt_loss": pmt_loss,
        "cluster_loss": cluster_loss,
        "mad_loss": mad_loss,
        "dim_loss": dim_loss,
        "nor_loss": nor_loss,
        "loc_loss": loc_loss,
        "geom_loss": geom_losses["geom_loss"],
        "inst_loss": inst_loss,
        "loss_plane": geom_losses["loss_plane"],
        "loss_cylinder": geom_losses["loss_cylinder"],
        "loss_cone": geom_losses["loss_cone"],
        "loss_sphere": geom_losses["loss_sphere"],
        "aux_factor": torch.tensor(aux_factor, device=xyz.device, dtype=xyz.dtype),
        "schedule/aux_progress": torch.tensor(aux_factor, device=xyz.device, dtype=xyz.dtype),
        "schedule/global_epoch": torch.tensor(float(global_epoch), device=xyz.device, dtype=xyz.dtype),
    }
    for name in raw_losses:
        loss_dict[f"raw/{name}"] = raw_losses[name]
        loss_dict[f"weighted/{name}"] = weighted_losses[name]
        loss_dict[f"effective_weight/{name}"] = torch.tensor(
            effective_weights[name], device=xyz.device, dtype=xyz.dtype
        )

    non_finite = [name for name, val in loss_dict.items() if torch.is_tensor(val) and not torch.isfinite(val).all()]
    if non_finite:
        printable = {name: value_item(val.detach()) for name, val in loss_dict.items() if torch.is_tensor(val) and val.dim() == 0}
        print(f"non-finite Stage 1 losses: {non_finite}; values={printable}")

    return loss_all, loss_dict


def canonicalize_vectors_hard(v, eps=1e-6):
    """
    归一化并进行方向标准化的向量处理函数。
    输入 v: [bs, point, 3]
    """

    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    z_zero = (z.abs() < eps)
    y_zero = (y.abs() < eps)
    x_zero = (x.abs() < eps)

    flip_mask = (
        (z < 0) |
        (z_zero & (y < 0)) |
        (z_zero & y_zero & (x < 0))
    )

    flip_mask = flip_mask.unsqueeze(-1)  # [bs, point, 1]
    v = torch.where(flip_mask, -v, v)

    return v


def safe_normalize(v, eps=1e-6, min_norm=0.05):
    """
    v: [bs, point, 3]
    防止向量长度过短
    """
    norm = v.norm(dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    v_normalized = v / norm

    # 只惩罚过短向量，防止不稳定
    length_loss = torch.relu(min_norm - norm).mean()
    return v_normalized, length_loss


def value_item(atensor):
    if isinstance(atensor, float):
        return atensor
    else:
        return atensor.item()


def test():
    pass


if __name__ == '__main__':
    test()


