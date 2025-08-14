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
            dp = nn.Dropout1d

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
                fea = dp(at(bn(fc(fea))))
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
    sequence_3d = sequence_expanded.tile((1, n_item))
    sequence_4d = sequence_3d.unsqueeze(-1)
    batch_indices = sequence_4d.repeat(1, 1, n_vals)
    view_shape = [n_item, n_vals]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = [bs, n_item, n_vals]
    repeat_shape[1] = 1
    channel_indices = torch.arange(n_item, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    return vals[batch_indices, channel_indices, inds]


def fps(xyz, n_samples):
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, n_samples, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask].float()
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn(vertices: "(bs, vertice_num, 3)",  neighbor_num: int, is_backdis: bool = False):
    bs, v, _ = vertices.size()
    inner = torch.bmm(vertices, vertices.transpose(1, 2)) #(bs, v, v)
    quadratic = torch.sum(vertices**2, dim= 2) #(bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    # print('distance.shape: ', distance.shape)

    neighbor_index = torch.topk(distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    if is_backdis:
        return neighbor_index, distance
    else:
        return neighbor_index


def surface_knn(points_all: "(bs, n_pnts, 3)", k_near: int = 100, n_stepk = 10):
    ind_neighbor_all, all_dist = knn(points_all, n_stepk, True)

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

    return points, foot_point


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


def gen_batched_random_plane_points(save_dir, n_planes=1000, n_point=2000):
    """
    随机生成一系列的点并保存
    每个平面保存在一个txt文件，文件名为 x-y-z.txt，(x y z)为垂足的三维坐标

    n_planes: 平面数
    n_point: 每个平面上采集的点数

    """
    os.makedirs(save_dir, exist_ok=True)

    # 先获取平面参数
    plane_coeff = generate_unique_random_planes(n_planes)

    # 获取点坐标
    for c_coeff in plane_coeff:
        c_pnts, c_root = random_points_in_plane(c_coeff[0], c_coeff[1], c_coeff[2], c_coeff[3], n_point)

        c_save_root = os.path.join(save_dir, f'{c_root[0]};{c_root[1]};{c_root[2]}.txt')
        np.savetxt(c_save_root, c_pnts)


if __name__ == '__main__':
    gen_batched_random_plane_points(r'D:\document\DeepLearning\DataSet\pcd_cstnet2\test')
    pass