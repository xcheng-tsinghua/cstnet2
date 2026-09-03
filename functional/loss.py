from __future__ import annotations

import torch.nn.functional as F
import torch
import numpy as np


# XYZ inputs are normalized to roughly [-1, 1]. These transition widths keep
# small, physically meaningful residuals quadratic while preventing remote
# axes/apices and very large radii from producing unbounded gradients.
PARAMETER_SMOOTH_L1_BETA = 0.1
GEOMETRY_SMOOTH_L1_BETA = 0.05
LARGE_PARAMETER_RATIO_THRESHOLD = 20.0
MIN_OBSERVABILITY_WEIGHT = 0.05


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


def _smooth_l1_zero(values: torch.Tensor, beta: float) -> torch.Tensor:
    """Element-wise Smooth L1 residual with a zero target."""
    return F.smooth_l1_loss(
        values,
        torch.zeros_like(values),
        reduction="none",
        beta=beta,
    )


def _instance_balanced_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    affiliate_idx: torch.Tensor,
    reference: torch.Tensor,
    instance_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average point losses inside each primitive, then average primitives.

    Primitive parameters are repeated at every sampled point. A plain point
    mean therefore gives densely sampled or large faces more influence even
    though they carry only one parameter tuple. This reduction gives every
    visible primitive instance one term instead.
    """
    if values.shape != mask.shape or affiliate_idx.shape != mask.shape:
        raise ValueError("instance-balanced loss inputs must all have shape [B, N]")
    if instance_weights is not None and instance_weights.shape != mask.shape:
        raise ValueError("instance_weights must have shape [B, N]")
    if not bool(mask.any()):
        return _zero_loss(reference)

    batch_ids = torch.arange(
        mask.shape[0], device=mask.device, dtype=affiliate_idx.dtype
    ).unsqueeze(1).expand_as(affiliate_idx)
    keys = torch.stack(
        (batch_ids[mask], affiliate_idx[mask]), dim=-1
    )
    unique_keys, inverse = torch.unique(
        keys, dim=0, sorted=True, return_inverse=True
    )
    instance_count = unique_keys.shape[0]
    sums = torch.zeros(
        instance_count, device=values.device, dtype=values.dtype
    )
    sums.index_add_(0, inverse, values[mask])
    counts = torch.bincount(inverse, minlength=instance_count).to(values.dtype)
    terms = sums / counts.clamp_min(1.0)
    if instance_weights is not None:
        weight_sums = torch.zeros_like(sums)
        weight_sums.index_add_(0, inverse, instance_weights[mask].to(values.dtype))
        # Do not renormalize the final terms by the sum of weights: a
        # low-confidence primitive must genuinely contribute less, including
        # when it is the only primitive of that type.
        terms = terms * (weight_sums / counts.clamp_min(1.0))
    return terms.mean()


@torch.no_grad()
def _parameter_observability_weights(
    xyz: torch.Tensor,
    pmt_gt: torch.Tensor,
    mad_gt: torch.Tensor,
    dim_gt: torch.Tensor,
    loc_gt: torch.Tensor,
    affiliate_idx: torch.Tensor,
    ratio_threshold: float = LARGE_PARAMETER_RATIO_THRESHOLD,
    min_weight: float = MIN_OBSERVABILITY_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return soft confidence weights for weakly observable DIM and LOC labels.

    A radius or remote center/axis/apex that is many times larger than the
    observed patch is intrinsically ill-conditioned. The weight decreases
    smoothly beyond ``ratio_threshold`` but never drops below ``min_weight``.
    Cylinder patch size is measured perpendicular to its GT axis so a long,
    narrow cylindrical strip is not incorrectly considered well observed.
    """
    if ratio_threshold <= 0:
        raise ValueError("ratio_threshold must be positive")
    if not 0 < min_weight <= 1:
        raise ValueError("min_weight must be in (0, 1]")

    batch_ids = torch.arange(
        xyz.shape[0], device=xyz.device, dtype=affiliate_idx.dtype
    ).unsqueeze(1).expand_as(affiliate_idx)
    keys = torch.stack(
        (batch_ids.reshape(-1), affiliate_idx.reshape(-1)), dim=-1
    )
    unique_keys, inverse = torch.unique(
        keys, dim=0, sorted=True, return_inverse=True
    )
    instance_count = unique_keys.shape[0]
    counts = torch.bincount(inverse, minlength=instance_count).float().clamp_min(1.0)

    xyz_flat = xyz.reshape(-1, 3).float()
    xyz_sums = torch.zeros(
        instance_count, 3, device=xyz.device, dtype=torch.float32
    )
    xyz_sums.index_add_(0, inverse, xyz_flat)
    xyz_centers = xyz_sums / counts.unsqueeze(-1)
    centered = xyz_flat - xyz_centers[inverse]

    # Project each cylinder point using its own GT axis. The projection is
    # unchanged for axis v or -v and therefore preserves the direction fix.
    direction_flat = F.normalize(
        mad_gt.reshape(-1, 3).float(), dim=-1, eps=1e-6
    )
    transverse = centered - (
        centered * direction_flat
    ).sum(dim=-1, keepdim=True) * direction_flat
    primitive_flat = pmt_gt.reshape(-1).long()
    extent_vectors = torch.where(
        (primitive_flat == 1).unsqueeze(-1), transverse, centered
    )
    point_extents = extent_vectors.norm(dim=-1)
    max_extents = torch.zeros(
        instance_count, device=xyz.device, dtype=torch.float32
    )
    max_extents.scatter_reduce_(
        0, inverse, point_extents, reduce="amax", include_self=True
    )
    patch_diameters = (2.0 * max_extents).clamp_min(1e-3)

    primitive_counts = torch.zeros(
        instance_count * 5, device=xyz.device, dtype=torch.float32
    )
    primitive_counts.index_add_(
        0,
        inverse * 5 + primitive_flat.clamp(min=0, max=4),
        torch.ones_like(primitive_flat, dtype=torch.float32),
    )
    instance_primitives = primitive_counts.reshape(instance_count, 5).argmax(dim=-1)

    dim_sums = torch.zeros_like(max_extents)
    dim_sums.index_add_(0, inverse, dim_gt.reshape(-1).float().abs())
    radius_scales = dim_sums / counts

    loc_sums = torch.zeros_like(xyz_sums)
    loc_sums.index_add_(0, inverse, loc_gt.reshape(-1, 3).float())
    location_scales = (loc_sums / counts.unsqueeze(-1) - xyz_centers).norm(dim=-1)

    def confidence(parameter_scales: torch.Tensor) -> torch.Tensor:
        return (
            ratio_threshold * patch_diameters
            / parameter_scales.clamp_min(1e-6)
        ).clamp(min=min_weight, max=1.0)

    dim_instance_weights = torch.where(
        (instance_primitives == 1) | (instance_primitives == 3),
        confidence(radius_scales),
        torch.ones_like(radius_scales),
    )
    # Plane offsets remain directly observable. Remote cylinder axes, cone
    # apices, and sphere centers receive a soft confidence weight.
    loc_instance_weights = torch.where(
        (instance_primitives == 1)
        | (instance_primitives == 2)
        | (instance_primitives == 3),
        confidence(location_scales),
        torch.ones_like(location_scales),
    )
    return (
        dim_instance_weights[inverse].reshape_as(dim_gt).to(xyz.dtype),
        loc_instance_weights[inverse].reshape_as(dim_gt).to(xyz.dtype),
    )


def _robust_dimension_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pmt_gt: torch.Tensor,
    affiliate_idx: torch.Tensor,
    observability_weights: torch.Tensor,
) -> torch.Tensor:
    radius_mask = _primitive_mask(pmt_gt, (1, 3))
    cone_mask = pmt_gt == 2
    valid_mask = radius_mask | cone_mask

    # Radius is positive and may span orders of magnitude. log1p preserves
    # resolution near zero while turning large absolute errors into scale-like
    # errors. Cone semi-angle is already bounded and stays in radians.
    pred_radius = torch.log1p(pred.clamp_min(0.0))
    target_radius = torch.log1p(target.clamp_min(0.0))
    radius_errors = F.smooth_l1_loss(
        pred_radius,
        target_radius,
        reduction="none",
        beta=PARAMETER_SMOOTH_L1_BETA,
    )
    cone_errors = F.smooth_l1_loss(
        pred,
        target,
        reduction="none",
        beta=PARAMETER_SMOOTH_L1_BETA,
    )
    point_errors = torch.where(
        radius_mask,
        radius_errors,
        torch.where(cone_mask, cone_errors, torch.zeros_like(pred)),
    )
    return _instance_balanced_mean(
        point_errors,
        valid_mask,
        affiliate_idx,
        pred,
        instance_weights=observability_weights,
    )


def _robust_location_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pmt_gt: torch.Tensor,
    affiliate_idx: torch.Tensor,
    observability_weights: torch.Tensor,
) -> torch.Tensor:
    valid_mask = _primitive_mask(pmt_gt, (0, 1, 2, 3))
    point_errors = F.smooth_l1_loss(
        pred,
        target,
        reduction="none",
        beta=PARAMETER_SMOOTH_L1_BETA,
    ).mean(dim=-1)
    return _instance_balanced_mean(
        point_errors,
        valid_mask,
        affiliate_idx,
        pred,
        instance_weights=observability_weights,
    )


def _masked_vector_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sign_invariant: bool = False,
) -> torch.Tensor:
    if not mask.any():
        return _zero_loss(pred)
    pred_m = F.normalize(pred[mask], dim=-1, eps=1e-6)
    target_m = F.normalize(target[mask], dim=-1, eps=1e-6)
    if sign_invariant:
        # Primitive directions are unoriented axes. Compare both equivalent
        # representatives directly instead of relying on a discontinuous
        # world-axis sign convention during optimization.
        direct = (pred_m - target_m).pow(2).mean(dim=-1)
        flipped = (pred_m + target_m).pow(2).mean(dim=-1)
        return torch.minimum(direct, flipped).mean()
    return F.mse_loss(pred_m, target_m)


def _masked_scalar_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return _zero_loss(pred)
    return F.mse_loss(pred[mask], target[mask])


def _masked_parallel_loss(
    vec1: torch.Tensor,
    vec2: torch.Tensor,
    mask: torch.Tensor,
    affiliate_idx: torch.Tensor | None = None,
    instance_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not mask.any():
        return _zero_loss(vec1)
    a = F.normalize(vec1, dim=-1, eps=1e-6)
    b = F.normalize(vec2, dim=-1, eps=1e-6)
    values = (1.0 - (a * b).sum(dim=-1).abs()).pow(2)
    if affiliate_idx is None:
        return values[mask].mean()
    return _instance_balanced_mean(
        values, mask, affiliate_idx, vec1, instance_weights
    )


def _masked_perpendicular_loss(
    vec1: torch.Tensor,
    vec2: torch.Tensor,
    mask: torch.Tensor,
    affiliate_idx: torch.Tensor | None = None,
    instance_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not mask.any():
        return _zero_loss(vec1)
    a = F.normalize(vec1, dim=-1, eps=1e-6)
    b = F.normalize(vec2, dim=-1, eps=1e-6)
    values = (a * b).sum(dim=-1).pow(2)
    if affiliate_idx is None:
        return values[mask].mean()
    return _instance_balanced_mean(
        values, mask, affiliate_idx, vec1, instance_weights
    )


def _stage1_geometry_losses(
    xyz: torch.Tensor,
    mad_pred: torch.Tensor,
    dim_pred: torch.Tensor,
    loc_pred: torch.Tensor,
    pmt_gt: torch.Tensor,
    affiliate_idx: torch.Tensor,
    dim_observability: torch.Tensor,
    loc_observability: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mad_pred = F.normalize(mad_pred, dim=-1, eps=1e-6)
    dim_pred = dim_pred.clamp_min(0.0)

    plane_mask = pmt_gt == 0
    if plane_mask.any():
        n = mad_pred
        plane_residual = ((xyz - loc_pred) * n).sum(dim=-1)
        plane_error = _smooth_l1_zero(
            plane_residual, GEOMETRY_SMOOTH_L1_BETA
        )
        on_plane = _instance_balanced_mean(
            plane_error, plane_mask, affiliate_idx, xyz
        )
        loc_nonzero = plane_mask & (loc_pred.norm(dim=-1) > 1e-4)
        loc_parallel = _masked_parallel_loss(
            loc_pred, mad_pred, loc_nonzero, affiliate_idx
        )
        loss_plane = on_plane + 0.2 * loc_parallel
    else:
        loss_plane = _zero_loss(xyz)

    cylinder_mask = pmt_gt == 1
    if cylinder_mask.any():
        v = xyz - loc_pred
        axis = mad_pred
        axial = (v * axis).sum(dim=-1, keepdim=True) * axis
        radial = (v - axial).norm(dim=-1)
        radius_error = _smooth_l1_zero(
            radial - dim_pred, GEOMETRY_SMOOTH_L1_BETA
        )
        cylinder_observability = torch.minimum(
            dim_observability, loc_observability
        )
        on_cylinder = _instance_balanced_mean(
            radius_error,
            cylinder_mask,
            affiliate_idx,
            xyz,
            cylinder_observability,
        )
        loc_perp_axis = _masked_perpendicular_loss(
            loc_pred,
            mad_pred,
            cylinder_mask,
            affiliate_idx,
            cylinder_observability,
        )
        loss_cylinder = on_cylinder + 0.2 * loc_perp_axis
    else:
        loss_cylinder = _zero_loss(xyz)

    cone_mask = pmt_gt == 2
    if cone_mask.any():
        v = xyz - loc_pred
        axis = mad_pred
        signed_axial = (v * axis).sum(dim=-1)
        radial_vec = v - signed_axial.unsqueeze(-1) * axis
        radial = radial_vec.norm(dim=-1)
        distance_to_apex = v.norm(dim=-1).clamp_min(1e-4)
        semi_angle = dim_pred.clamp(min=1e-4, max=1.55)
        # Compare normalized radial direction with sin(semi_angle). Unlike
        # radial - axial*tan(angle), this residual and its angle derivative do
        # not explode for a remote apex or an angle close to pi/2.
        cone_error = _smooth_l1_zero(
            radial / distance_to_apex - torch.sin(semi_angle),
            GEOMETRY_SMOOTH_L1_BETA,
        )
        loss_cone = _instance_balanced_mean(
            cone_error,
            cone_mask,
            affiliate_idx,
            xyz,
            loc_observability,
        )
    else:
        loss_cone = _zero_loss(xyz)

    sphere_mask = pmt_gt == 3
    if sphere_mask.any():
        center_to_xyz = xyz - loc_pred
        radius_error = _smooth_l1_zero(
            center_to_xyz.norm(dim=-1) - dim_pred,
            GEOMETRY_SMOOTH_L1_BETA,
        )
        sphere_observability = torch.minimum(
            dim_observability, loc_observability
        )
        loss_sphere = _instance_balanced_mean(
            radius_error,
            sphere_mask,
            affiliate_idx,
            xyz,
            sphere_observability,
        )
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
    mad_pred = F.normalize(mad_pred, dim=-1, eps=1e-6)
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
                # Align signs relative to one detached member before averaging.
                # Thus v and -v reinforce the same cluster axis instead of
                # cancelling each other around the dir_unify boundary.
                reference = mad[:1].detach()
                flip = (mad * reference).sum(dim=-1, keepdim=True) < 0
                aligned_mad = torch.where(flip, -mad, mad)
                mean_mad = F.normalize(
                    aligned_mad.mean(0, keepdim=True), dim=-1, eps=1e-6
                )
                terms.append(
                    F.mse_loss(aligned_mad, mean_mad.expand_as(aligned_mad))
                )

            # ---- 尺寸一致性 ----
            if inst_prim is None or inst_prim in (1, 2, 3):
                dim = dim_pred[b][mask]  # [N]
                if inst_prim in (1, 3):
                    dim = torch.log1p(dim.clamp_min(0.0))
                mean_dim = dim.mean()
                terms.append(
                    F.smooth_l1_loss(
                        dim,
                        mean_dim.expand_as(dim),
                        beta=PARAMETER_SMOOTH_L1_BETA,
                    )
                )

            # ---- 主位置一致性 ----
            if inst_prim is None or inst_prim in (0, 1, 2, 3):
                loc = loc_pred[b][mask]  # [N, 3]
                mean_loc = loc.mean(0, keepdim=True)
                terms.append(
                    F.smooth_l1_loss(
                        loc,
                        mean_loc.expand_as(loc),
                        beta=PARAMETER_SMOOTH_L1_BETA,
                    )
                )

    if len(terms) == 0:
        return _zero_loss(log_pmt_pred)
    return torch.stack(terms).mean()


def linear_ramp(global_epoch, start_epoch, ramp_epochs):
    """Linear schedule used by all delayed Stage 1 auxiliary losses."""
    if ramp_epochs <= 0:
        return 1.0 if global_epoch >= start_epoch else 0.0
    progress = (float(global_epoch) - float(start_epoch)) / float(ramp_epochs)
    return max(0.0, min(1.0, progress))


def constraint_loss(xyz, log_pmt_pred, mad_pred, dim_pred, loc_pred,
                    pmt_gt, mad_gt, dim_gt, loc_gt, affil_idx,
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
    :param loc_pred: [bs, point, 3]

    标签数据
    :param pmt_gt: [bs, point] (int, index)
    :param mad_gt: [bs, point, 3]
    :param dim_gt: [bs, point]
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
    w_loc = float(weights.get("w_loc", 0.02))
    w_geom = float(weights.get("w_geom", 0.02))
    w_inst = float(weights.get("w_inst", 0.005))

    pmt_loss = F.nll_loss(log_pmt_pred.reshape(-1, 5), pmt_gt.reshape(-1))
    cluster_loss = discriminative_loss(point_emb, affil_idx) if point_emb is not None else _zero_loss(log_pmt_pred)

    mad_pred = F.normalize(mad_pred, dim=-1, eps=eps)
    mad_gt = F.normalize(mad_gt, dim=-1, eps=eps)

    mad_mask = _primitive_mask(pmt_gt, (0, 1, 2))
    mad_loss = _masked_vector_mse(
        mad_pred, mad_gt, mad_mask, sign_invariant=True
    )
    dim_observability, loc_observability = _parameter_observability_weights(
        xyz,
        pmt_gt,
        mad_gt,
        dim_gt,
        loc_gt,
        affil_idx,
    )
    dim_loss = _robust_dimension_loss(
        dim_pred,
        dim_gt,
        pmt_gt,
        affil_idx,
        dim_observability,
    )
    loc_loss = _robust_location_loss(
        loc_pred,
        loc_gt,
        pmt_gt,
        affil_idx,
        loc_observability,
    )

    geom_losses = _stage1_geometry_losses(
        xyz,
        mad_pred,
        dim_pred,
        loc_pred,
        pmt_gt,
        affil_idx,
        dim_observability,
        loc_observability,
    )
    inst_loss = instance_consistency_loss(log_pmt_pred, mad_pred, dim_pred, loc_pred, affil_idx, pmt_gt)

    aux_factor = linear_ramp(global_epoch, geom_start_epoch, geom_ramp_epochs)
    raw_losses = {
        "pmt": pmt_loss,
        "cluster": cluster_loss,
        "mad": mad_loss,
        "dim": dim_loss,
        "loc": loc_loss,
        "geom": geom_losses["geom_loss"],
        "inst": inst_loss,
    }
    target_weights = {
        "pmt": w_pmt,
        "cluster": w_cluster,
        "mad": w_mad,
        "dim": w_dim,
        "loc": w_loc,
        "geom": w_geom,
        "inst": w_inst,
    }
    ramped_names = {"geom", "inst"}
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
        "loss_plane": geom_losses["loss_plane"],
        "loss_cylinder": geom_losses["loss_cylinder"],
        "loss_cone": geom_losses["loss_cone"],
        "loss_sphere": geom_losses["loss_sphere"],
        "schedule/aux_progress": torch.tensor(aux_factor, device=xyz.device, dtype=xyz.dtype),
        "observability/dim_mean": _masked_mean(
            dim_observability,
            _primitive_mask(pmt_gt, (1, 2, 3)),
            xyz,
        ),
        "observability/loc_mean": _masked_mean(
            loc_observability,
            _primitive_mask(pmt_gt, (0, 1, 2, 3)),
            xyz,
        ),
        "robust/parameter_beta": torch.tensor(
            PARAMETER_SMOOTH_L1_BETA, device=xyz.device, dtype=xyz.dtype
        ),
        "robust/geometry_beta": torch.tensor(
            GEOMETRY_SMOOTH_L1_BETA, device=xyz.device, dtype=xyz.dtype
        ),
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


