import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
from functools import partial
import torch.nn.functional as F
import os
from pathlib import Path
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, dimension: int, channels: tuple, bias: bool = True, dropout: float = 0.4, final_proc=False):
        """
        :param dimension: input data dimension，[0, 1, 2, 3]
            tensor: [bs, c], dimension = 0
            tensor: [bs, c, d], dimension = 1
            tensor: [bs, c, d, e], dimension = 2
            tensor: [bs, c, d, e, f], dimension = 3
        :param channels: channels along input and output layers，[in, hid1, hid2, ..., out]
        :param bias:
        :param dropout:
        :param final_proc: is adding BatchNormalization, Active function, DropOut after final linear layers
        """
        super().__init__()
        self.dimension = dimension

        self.linear_layers = nn.ModuleList()
        self.batch_normals = nn.ModuleList()
        self.activates = nn.ModuleList()
        self.drop_outs = nn.ModuleList()

        self.n_layers = len(channels)
        self.final_proc = final_proc
        if dropout == 0:
            self.is_drop = False
        else:
            self.is_drop = True

        if dimension == 0:
            fc = nn.Linear
            bn = nn.BatchNorm1d
            dp = nn.Dropout

        elif dimension == 1:
            fc = partial(nn.Conv1d, kernel_size=1)
            bn = nn.BatchNorm1d
            dp = nn.Dropout

        elif dimension == 2:
            fc = partial(nn.Conv2d, kernel_size=1)
            bn = nn.BatchNorm2d
            dp = nn.Dropout2d

        elif dimension == 3:
            fc = partial(nn.Conv3d, kernel_size=1)
            bn = nn.BatchNorm3d
            dp = nn.Dropout3d

        else:
            raise ValueError('error dimension value, [0, 1, 2, 3] is supported')

        for i in range(self.n_layers - 2):
            self.linear_layers.append(fc(channels[i], channels[i + 1], bias=bias))
            self.batch_normals.append(bn(channels[i + 1]))
            self.activates.append(nn.LeakyReLU(negative_slope=0.2))
            self.drop_outs.append(dp(dropout))

        self.outlayer = fc(channels[-2], channels[-1], bias=bias)

        self.outbn = bn(channels[-1])
        self.outat = nn.LeakyReLU(negative_slope=0.2)
        self.outdp = dp(dropout)

    def forward(self, fea):
        """
        :param fea:
        :return:
        """

        for i in range(self.n_layers - 2):
            fc = self.linear_layers[i]
            bn = self.batch_normals[i]
            at = self.activates[i]
            dp = self.drop_outs[i]

            if self.is_drop:
                if self.dimension == 0:
                    fea = dp(at(bn(fc(fea).permute(0, 2, 1)))).permute(0, 2, 1)
                else:
                    fea = dp(at(bn(fc(fea))))
            else:
                if self.dimension == 0:
                    fea = at(bn(fc(fea).permute(0, 2, 1))).permute(0, 2, 1)
                else:
                    fea = at(bn(fc(fea)))

        fea = self.outlayer(fea)

        if self.final_proc:
            fea = self.outbn(fea)
            fea = self.outat(fea)

            if self.is_drop:
                fea = self.outdp(fea)

        return fea


class PointAttention(nn.Module):
    def __init__(self, channel_c, channel_n, channel_out):
        super().__init__()

        channel_mid = int((channel_c * channel_out) ** 0.5)
        self.fai = MLP(2, (channel_c, channel_mid, channel_out))
        self.psi = MLP(2, (channel_n, channel_mid, channel_out))
        self.alpha = MLP(2, (channel_n, channel_mid, channel_out))
        self.gamma = MLP(2, (channel_out, channel_out - 1, channel_out))

    def forward(self, x_i, x_j):
        """
        x_i: [bs, n_point, channel], center
        x_j: [bs, n_point, n_near, channel], neighbor

        """
        x_i = x_i.unsqueeze(2).permute(0, 3, 2, 1)
        x_j = x_j.permute(0, 3, 2, 1)

        bs, channel, n_near, n_point = x_j.size()

        fai_xi = self.fai(x_i)  # -> [bs, channel, 1, npoint]
        psi_xj = self.psi(x_j)  # -> [bs, channel, n_near, npoint]
        alpha_xj = self.alpha(x_j)  # -> [bs, channel, n_near, npoint]

        y_i = (channel * F.softmax(self.gamma(fai_xi - psi_xj), dim=1)) * alpha_xj  # -> [bs, channel, n_near, npoint]
        y_i = torch.sum(y_i, dim=2)  # -> [bs, channel, npoint]
        y_i = y_i / n_near  # -> [bs, channel, npoint]
        y_i = y_i.permute(0, 2, 1)  # -> [bs, npoint, channel]

        return y_i

class PointAttention2(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()

        channel_mid = int((channel_in * channel_out) ** 0.5)
        self.fai = MLP(2, (channel_in, channel_mid, channel_out))
        self.psi = MLP(2, (channel_in, channel_mid, channel_out))

    def forward(self, x_i, x_j):
        """
        x_i: [bs, n_point, channel], center
        x_j: [bs, n_point, n_near, channel], neighbor

        """
        x_i = x_i.unsqueeze(2).permute(0, 3, 2, 1) # -> [bs, channel, 1, npoint]
        x_j = x_j.permute(0, 3, 2, 1)   # -> [bs, channel, n_near, npoint]

        bs, channel, n_near, n_point = x_j.size()
        x_i = x_i.repeat(1, 1, n_near, 1)

        # print("x_i: ", x_i[0, 0, 0, :10])
        # print("x_j: ", x_j[0, 0, 0, :10])

        # print("torch.is_nan(x_i).any(): ", torch.isnan(x_i).any(), "torch.is_nan(x_j).any(): ", torch.isnan(x_j).any())

        # print("x_i - x_j: ", (x_i - x_j).shape)
        # print("x_i - x_j: ", (x_i - x_j)[0, 0, 0, :10])
        # print("(x_i - x_j).min: ", (x_i - x_j).min(), "(x_i - x_j).max(): ", (x_i - x_j).max())
        # print("torch.abs(x_i - x_j): ", torch.abs(x_i - x_j).shape)
        y_i = channel * F.softmax(self.fai(torch.abs(x_i - x_j)), dim=1) * self.psi(x_i)  # -> [bs, channel, n_near, npoint]
        y_i = torch.sum(y_i, dim=2)  # -> [bs, channel, npoint]
        y_i = y_i / n_near  # -> [bs, channel, npoint]
        y_i = y_i.permute(0, 2, 1)  # -> [bs, npoint, channel]

        return y_i

class FeaAttention(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()

        channel_mid = int((channel_in * channel_out) ** 0.5)
        self.fai = MLP(2, (channel_in, channel_mid, channel_out))
        self.psi = MLP(2, (channel_in, channel_mid, channel_out))
        self.alpha = MLP(2, (channel_in, channel_mid, channel_out))
        self.gamma = MLP(2, (channel_out, channel_out, channel_out))

    def forward(self, mad_fea, adj_fea, pt_fea, cst_fea):
        """
        :param mad_fea: [bs, n_points, f]
        :param adj_fea: [bs, n_points, f]
        :param pt_fea: [bs, n_points, f]
        :param cst_fea: [bs, n_points, f]
        :return:
        """

        # x_i: [bs, channel, 1, n_point]
        x_i = cst_fea.unsqueeze(2).permute(0, 3, 2, 1)

        # x_j: [bs, channel, 3, n_point]
        x_j = torch.cat([mad_fea.unsqueeze(2), adj_fea.unsqueeze(2), pt_fea.unsqueeze(2)], dim=2).permute(0, 3, 2, 1)

        bs, channel, n_near, n_point = x_j.size()

        fai_xi = self.fai(x_i)  # -> [bs, channel, 1, npoint]
        psi_xj = self.psi(x_j)  # -> [bs, channel, n_near, npoint]
        alpha_xj = self.alpha(x_j)  # -> [bs, channel, n_near, npoint]

        y_i = (channel * F.softmax(self.gamma(fai_xi - psi_xj), dim=1)) * alpha_xj  # -> [bs, channel, n_near, npoint]
        y_i = torch.sum(y_i, dim=2)  # -> [bs, channel, npoint]
        y_i = y_i / n_near  # -> [bs, channel, npoint]
        y_i = y_i.permute(0, 2, 1)  # -> [bs, npoint, channel]

        return y_i

class FeaAttention2(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()

        channel_mid = int((channel_in * channel_out) ** 0.5)
        self.fai = MLP(2, (channel_in, channel_mid, channel_out))
        self.psi = MLP(2, (channel_in, channel_mid, channel_out))

    def forward(self, xyz_fea, pmt_fea, mad_fea, dim_fea, nor_fea, loc_fea, fea):
        """
        :param mad_fea: [bs, n_points, f]
        :param adj_fea: [bs, n_points, f]
        :param pt_fea: [bs, n_points, f]
        :param cst_fea: [bs, n_points, f]
        :return:
        """

        # x_i: [bs, channel, 1, n_point]
        x_i = fea.unsqueeze(2).permute(0, 3, 2, 1)

        # x_j: [bs, channel, 3, n_point]
        x_j = torch.cat([xyz_fea.unsqueeze(2), pmt_fea.unsqueeze(2), mad_fea.unsqueeze(2),
                         dim_fea.unsqueeze(2), nor_fea.unsqueeze(2), loc_fea.unsqueeze(2)], dim=2).permute(0, 3, 2, 1)

        bs, channel, n_near, n_point = x_j.size()

        y_i = (channel * F.softmax(self.fai(torch.abs(x_i - x_j)), dim=1)) * self.psi(x_i)  # -> [bs, channel, n_near, npoint]
        y_i = torch.sum(y_i, dim=2)  # -> [bs, channel, npoint]
        y_i = y_i / n_near  # -> [bs, channel, npoint]
        y_i = y_i.permute(0, 2, 1)  # -> [bs, npoint, channel]

        return y_i


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def indexes_val(vals, inds):
    bs, n_item, n_vals = inds.size()
    sequence = torch.arange(bs)
    sequence_expanded = sequence.unsqueeze(1)
    sequence_3d = sequence_expanded.repeat((1, n_item))
    sequence_4d = sequence_3d.unsqueeze(-1)
    batch_indices = sequence_4d.repeat(1, 1, n_vals)
    view_shape = [n_item, n_vals]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = [bs, n_item, n_vals]
    repeat_shape[1] = 1
    channel_indices = torch.arange(n_item, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    return vals[batch_indices, channel_indices, inds]


def fps(xyz, n_samples, dim=3):
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, n_samples, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, dim)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask].float()
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn(vertices, k, is_back_dist=False, is_include_self=True):
    """
    返回最近的 k 个点索引
    :param vertices: [bs, n_point, 3]
    :param k: number of neighbor points
    :param is_back_dist: 是否返回平方距离
    :param is_include_self: 是否包含自身
    :return: index of neighbor points [bs, n_point, k]
    """
    # 计算平方距离
    inner = torch.bmm(vertices, vertices.transpose(1, 2))  # (bs, v, v)
    quadratic = torch.sum(vertices**2, dim=2)  # (bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)

    neighbor_index = torch.topk(distance, k=k + 1, dim=-1, largest=False)[1]

    if is_include_self:
        neighbor_index = neighbor_index[:, :, :-1]
    else:
        neighbor_index = neighbor_index[:, :, 1:]

    if is_back_dist:
        return neighbor_index, distance
    else:
        return neighbor_index


def surface_knn(points_all: "(bs, n_pnts, 3)", k_near=100, n_stepk=10):
    ind_neighbor_all, all_dist = knn(points_all, n_stepk, True, False)

    neighbor_index_max = torch.max(all_dist, dim=-1, keepdim=True)[1]

    new_neighinds = ind_neighbor_all.clone()

    num_ita = 0
    while True:
        n_current_neighbors = new_neighinds.size()[-1]
        indexed_all = []
        for j in range(n_current_neighbors):
            indexed_all.append(index_points(ind_neighbor_all, new_neighinds[:, :, j]))
        new_neighinds = torch.cat(indexed_all, dim=-1)

        new_neighinds = torch.sort(new_neighinds, dim=-1)[0]

        duplicates = torch.zeros_like(new_neighinds)
        duplicates[:, :, 1:] = new_neighinds[:, :, 1:] == new_neighinds[:, :, :-1]

        neighbor_index_max2 = neighbor_index_max.repeat(1, 1, new_neighinds.shape[-1])
        new_neighinds[duplicates.bool()] = neighbor_index_max2[duplicates.bool()]

        dist_neighinds = indexes_val(all_dist, new_neighinds)

        sort_dist = torch.sort(dist_neighinds, dim=-1)[0]  # -> [bs, n_point, n_near]

        sort_dist_maxind = torch.max(sort_dist, dim=-1)[1]  # -> [bs, n_point]
        valid_nnear = torch.min(sort_dist_maxind).item() + 1

        is_end_loop = False
        if valid_nnear >= k_near + 1:
            valid_nnear = k_near + 1
            is_end_loop = True

        sub_neighbor_index = torch.topk(dist_neighinds, k=valid_nnear, dim=-1, largest=False)[1]  # [0] val, [1] index

        new_neighinds = indexes_val(new_neighinds, sub_neighbor_index)

        new_neighinds = new_neighinds[:, :, 1:]

        if is_end_loop:
            break

        num_ita += 1
        if num_ita > 20:
            print('max surface knn iteration count, return knn')
            return ind_neighbor_all

    return new_neighinds


def all_metric_cls(all_preds: list, all_labels: list):
    """
    计算分类评价指标：Acc.instance, Acc.class, F1-score, mAP
    :param all_preds: [item0, item1, ...], item: [bs, n_classes]
    :param all_labels: [item0, item1, ...], item: [bs, ], Only int tensor in supported
    :return: Acc.instance, Acc.class, F1-score-macro, F1-score-weighted, mAP
    """
    all_preds = np.vstack(all_preds)  # [n_samples, n_classes]
    all_labels = np.hstack(all_labels)  # [n_samples]
    n_samples, n_classes = all_preds.shape

    if not np.issubdtype(all_labels.dtype, np.integer):
        raise TypeError('Not all int data in all_labels')

    # ---------- Acc.Instance ----------
    pred_choice = np.argmax(all_preds, axis=1)  # -> [n_samples, ]
    correct = np.equal(pred_choice, all_labels).sum()
    acc_ins = correct / n_samples

    # ---------- Acc.class ----------
    acc_cls = []
    for class_idx in range(n_classes):
        class_mask = (all_labels == class_idx)
        if np.sum(class_mask) == 0:
            continue
        cls_acc_sig = np.mean(pred_choice[class_mask] == all_labels[class_mask])
        acc_cls.append(cls_acc_sig)
    acc_cls = np.mean(acc_cls)

    # ---------- F1-score ----------
    f1_m = f1_score(all_labels, pred_choice, average='macro')
    f1_w = f1_score(all_labels, pred_choice, average='weighted')

    # ---------- mAP ----------
    all_labels_one_hot = label_binarize(all_labels, classes=np.arange(n_classes))

    if n_classes == 2:
        all_labels_one_hot_rev = 1 - all_labels_one_hot
        all_labels_one_hot = np.concatenate([all_labels_one_hot_rev, all_labels_one_hot], axis=1)

    ap_sig = []
    for i in range(n_classes):
        ap = average_precision_score(all_labels_one_hot[:, i], all_preds[:, i])
        ap_sig.append(ap)

    mAP = np.mean(ap_sig)

    return acc_ins, acc_cls, f1_m, f1_w, mAP


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def is_suffix_step(filename):
    if filename[-4:] == '.stp' \
            or filename[-5:] == '.step' \
            or filename[-5:] == '.STEP':
        return True

    else:
        return False


def get_allfiles(dir_path, suffix='txt', filename_only=False):
    """
    get all files in dir_path, include files in sub dirs
    """
    filepath_all = []

    def other_judge(file_name):
        if file_name.split('.')[-1] == suffix:
            return True
        else:
            return False

    if suffix == 'stp' or suffix == 'step' or suffix == 'STEP':
        suffix_judge = is_suffix_step
    else:
        suffix_judge = other_judge

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if suffix_judge(file):
                if filename_only:
                    current_filepath = file
                else:
                    current_filepath = str(os.path.join(root, file))
                filepath_all.append(current_filepath)

    return filepath_all


def get_subdirs(dir_path):
    """
    get all 1st sub dirs' name, not dir path
    """
    path_allclasses = Path(dir_path)
    directories = [str(x) for x in path_allclasses.iterdir() if x.is_dir()]
    dir_names = [item.split(os.sep)[-1] for item in directories]

    return dir_names


def vis_confusion_mat(file_name):
    array_from_file = np.loadtxt(file_name, dtype=int)

    matrix_size = array_from_file.max() + 1
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    for i in range(array_from_file.shape[1]):
        x = array_from_file[0, i]
        y = array_from_file[1, i]
        matrix[x, y] += 1

    print(matrix)

    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Counts')
    plt.title('Confusion Matrix')
    plt.xlabel('target')
    plt.ylabel('predict')
    plt.xticks(np.arange(matrix_size))
    plt.yticks(np.arange(matrix_size))
    plt.show()


def save_confusion_mat(pred_list: list, target_list: list, save_name):
    matrix_size = max(max(pred_list), max(target_list)) + 1
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    list_len = len(pred_list)
    if list_len != len(target_list):
        return

    for i in range(list_len):
        x = pred_list[i]
        y = target_list[i]
        matrix[x, y] += 1

    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Counts')
    plt.title('Confusion Matrix')
    plt.xlabel('target')
    plt.ylabel('predict')
    plt.xticks(np.arange(matrix_size))
    plt.yticks(np.arange(matrix_size))
    plt.savefig(save_name)
    plt.close()


def random_points_in_plane(a, b, c, d, n1):
    # Step 1: 计算平面法向量
    normal = np.array([a, b, c], dtype=float)
    normal /= np.linalg.norm(normal)

    # Step 2: 计算原点到平面的垂足
    # 平面方程: a*x + b*y + c*z + d = 0
    t = -d / (a ** 2 + b ** 2 + c ** 2)
    foot_point = t * np.array([a, b, c], dtype=float)

    # Step 3: 构建平面局部坐标系 (u_dir, v_dir)
    # 找一个与normal不平行的向量
    if abs(normal[0]) < 0.9:
        tmp = np.array([1, 0, 0], dtype=float)
    else:
        tmp = np.array([0, 1, 0], dtype=float)

    u_dir = np.cross(normal, tmp)
    u_dir /= np.linalg.norm(u_dir)
    v_dir = np.cross(normal, u_dir)

    # Step 4: 在局部坐标 [-100, 100] 范围内随机选择一个小块
    grid_min, grid_max = -100, 100
    grid_x = np.random.randint(grid_min, grid_max)
    grid_y = np.random.randint(grid_min, grid_max)

    # Step 5: 在该小块内随机生成n1个点
    local_u = grid_x + np.random.rand(n1)
    local_v = grid_y + np.random.rand(n1)

    # Step 6: 将局部坐标映射到三维空间
    points = foot_point + np.outer(local_u, u_dir) + np.outer(local_v, v_dir)

    return points


def fit_plane(points):
    # points: (N,3) numpy array
    pts = np.asarray(points, dtype=float)
    assert pts.shape[1] == 3 and pts.shape[0] >= 3

    mu = pts.mean(axis=0)
    X = pts - mu
    # SVD on centered data
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    n = Vt[-1]  # unit normal
    n = n / np.linalg.norm(n)
    d = -np.dot(n, mu)
    # RMS orthogonal error
    rms = np.sqrt(np.mean((X @ n)**2))
    a, b, c = n
    return a, b, c, d, rms


def generate_unique_random_planes(num_planes, coeff_range=(-10, 10)):
    """
    生成唯一的随机平面参数 (a, b, c, d)，保证 a^2 + b^2 + c^2 != 0 且没有重复

    参数:
        num_planes: int，要生成的平面数量
        coeff_range: tuple，(min_val, max_val) 系数的范围

    返回:
        numpy数组，形状为 (num_planes, 4)，每行是 [a, b, c, d]
    """
    planes_set = set()
    min_val, max_val = coeff_range

    while len(planes_set) < num_planes:
        a, b, c, d = np.random.uniform(min_val, max_val, 4)

        # 检查法向量是否为零
        if a ** 2 + b ** 2 + c ** 2 == 0:
            continue

        # 用元组保存，避免浮点比较误差问题 → 先四舍五入到一定小数位
        key = tuple(np.round([a, b, c, d], 6))

        if key not in planes_set:
            planes_set.add(key)

    return list(planes_set)


def generate_planes(n_planes, d_range=(-1, 1), tol=1e-6, seed=None):
    """
    随机生成一组平面参数 (a, b, c, d)，满足 sqrt(a^2+b^2+c^2)=1，且不重复。

    参数:
        n_planes : int   需要生成的平面数量
        d_range  : tuple (d_min, d_max)  d 的范围
        tol      : float 重复判断的容差
        seed     : int or None 随机数种子

    返回:
        planes : list of (a,b,c,d)
    """
    if seed is not None:
        np.random.seed(seed)

    planes = []

    while len(planes) < n_planes:
        # 随机生成法向量（高斯分布后归一化）
        n = np.random.randn(3)
        n /= np.linalg.norm(n)

        # 随机生成 d
        d = np.random.uniform(d_range[0], d_range[1])

        # 构造平面参数
        plane = (n[0], n[1], n[2], d)

        # 检查是否重复（考虑方向相反的情况也算重复）
        is_duplicate = False
        for (a, b, c, dd) in planes:
            n1 = np.array([a, b, c])
            n2 = np.array([plane[0], plane[1], plane[2]])
            # 如果法向量相同或相反，且 d 也接近
            if (np.allclose(n1, n2, atol=tol) and abs(dd - plane[3]) < tol) \
                    or (np.allclose(n1, -n2, atol=tol) and abs(dd + plane[3]) < tol):
                is_duplicate = True
                break

        if not is_duplicate:
            planes.append(plane)

    return planes


def dir_unify(a, b, c, d, tol=1e-6):
    """
    使平面参数固定，不至于出现同一个平面参数护卫相反数
    """
    # 先看 d
    if d > tol:
        return a, b, c, d
    elif d < -1 * tol:
        return -1 * a, -1 * b, -1 * c, -1 * d

    # d == 0 的情况，看 c
    elif c > tol:
        return a, b, c, d
    elif c < -1 * tol:
        return -1 * a, -1 * b, -1 * c, -1 * d

    # d == 0 and c == 0 的情况，看 b
    elif b > tol:
        return a, b, c, d
    elif b < -1 * tol:
        return -1 * a, -1 * b, -1 * c, -1 * d

    # d == 0 and c == 0 and b == 0 的情况，看 a
    elif a > tol:
        return a, b, c, d
    elif a < -1 * tol:
        return -1 * a, -1 * b, -1 * c, -1 * d

    # 其它情况，出错，因为 sqrt(a^2+b^2+c^2)=1
    else:
        raise ValueError('not satisfy sqrt(a^2+b^2+c^2)=1')


def gen_batched_random_plane_points(save_dir, n_planes, n_point=2000):
    """
    随机生成一系列的点并保存
    每个平面保存在一个txt文件，文件名为 x-y-z.txt，(x y z)为垂足的三维坐标

    n_planes: 平面数
    n_point: 每个平面上采集的点数

    """
    os.makedirs(save_dir, exist_ok=True)

    # 先获取平面参数
    print('生成随机参数')
    plane_coeff = generate_planes(n_planes)

    # 获取点坐标
    for c_coeff in plane_coeff:
        a, b, c, d = dir_unify(*c_coeff)

        c_pnts = random_points_in_plane(a, b, c, d, n_point)

        # 将点云放在 [-1, 1] 之间
        dist = np.max(np.sqrt(np.sum(c_pnts ** 2, axis=1)), 0)
        c_pnts = c_pnts / dist  # scale

        fa, fb, fc, fd, fm = fit_plane(c_pnts)

        if fm > 1e-6:
            print('skip a plane')
            continue

        fa, fb, fc, fd = dir_unify(fa, fb, fc, fd)

        c_save_root = os.path.join(save_dir, f'{fa};{fb};{fc};{fd}.txt')
        np.savetxt(c_save_root, c_pnts)


if __name__ == '__main__':
    gen_batched_random_plane_points(r'D:\document\DeepLearning\DataSet\pcd_cstnet2\test', 1000)
    pass



