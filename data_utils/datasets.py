import os
import numpy as np
from torch.utils.data import Dataset

from modules import utils


class CstPntDataset(Dataset):
    def __init__(self,
                 root,
                 n_points=2500,
                 data_augmentation=True
                 ):
        print('CstPnt dataset, from:' + root)

        self.n_points = n_points
        self.data_augmentation = data_augmentation
        self.datapath = utils.get_allfiles(root)

        print('instance all:', len(self.datapath))

    def __getitem__(self, index):
        fn = self.datapath[index].strip()
        point_set = np.loadtxt(fn)  # [x, y, z, ex, ey, ez, adj, pt]

        try:
            choice = np.random.choice(point_set.shape[0], self.n_points, replace=True)
        except:
            exit(f'insufficient point number of the point cloud: all points: {point_set.shape[0]}, required points: {self.n_points}')

        point_set = point_set[choice, :]

        xyz = point_set[:, :3]
        mad = point_set[:, 3: 6]
        adj = point_set[:, 6]
        pt = point_set[:, 7]

        xyz = xyz - np.expand_dims(np.mean(xyz, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)), 0)
        xyz = xyz / dist  # scale

        if self.data_augmentation:
            xyz += np.random.normal(0, 0.02, size=xyz.shape)

        return xyz, mad, adj, pt

    def __len__(self):
        return len(self.datapath)


class Param20KDataset(Dataset):
    def __init__(self,
                 root,
                 is_train=True,
                 n_points=2500,
                 data_augmentation=True,
                 is_backcst=True
                 ):
        """
        定位文件的路径如下：
        root
        ├─ train
        │   ├─ Bushes
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ├─ Clamps
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ...
        │
        ├─ test
        │   ├─ Bushes
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ├─ Clamps
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ...
        │

        """
        print('Param20K dataset, from:' + root)

        self.n_points = n_points
        self.data_augmentation = data_augmentation
        self.is_backcst = is_backcst

        if is_train:
            inner_root = os.path.join(root, 'train')
        else:
            inner_root = os.path.join(root, 'test')

        category_all = utils.get_subdirs(inner_root)
        category_path = {}  # {'plane': [Path1,Path2,...], 'car': [Path1,Path2,...]}

        for c_class in category_all:
            class_root = os.path.join(inner_root, c_class)
            file_path_all = utils.get_allfiles(class_root)
            category_path[c_class] = file_path_all

        self.datapath = []
        for item in category_path:
            for fn in category_path[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(sorted(category_path), range(len(category_path))))
        print(self.classes)
        print('instance all:', len(self.datapath))

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[fn[0]]
        point_set = np.loadtxt(fn[1])  # (x, y, z, mad, adj, pt)

        try:
            choice = np.random.choice(point_set.shape[0], self.n_points, replace=False)
        except:
            exit(f'insufficient point number of the point cloud: all points: {point_set.shape[0]}, required points: {self.n_points}')

        point_set = point_set[choice, :]
        xyz = point_set[:, :3]

        # scale points to [-1, 1]^2
        xyz = xyz - np.expand_dims(np.mean(xyz, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)), 0)
        xyz = xyz / dist

        if self.data_augmentation:
            xyz += np.random.normal(0, 0.02, size=xyz.shape)

        if self.is_backcst:
            mad = point_set[:, 3: 6]
            adj = point_set[:, 6]
            pt = point_set[:, 7]

            return xyz, cls, mad, adj, pt
        else:
            return xyz, cls

    def __len__(self):
        return len(self.datapath)

    def n_classes(self):
        return len(self.classes)


class RegressionDataset(Dataset):
    def __init__(self, root, is_train, n_points):

        print('RegressionDataset dataset, from:' + root)
        self.n_points = n_points

        if is_train:
            inner_root = os.path.join(root, 'train')
        else:
            inner_root = os.path.join(root, 'test')

        self.path_label = []  # [([x, y ,z], Path1), ...]
        files_all = utils.get_allfiles(inner_root)

        for c_file in files_all:
            c_base = os.path.basename(c_file)
            c_base = os.path.splitext(c_base)[0]

            c_a, c_b, c_c, c_d = c_base.split(';')

            coefficient = np.array([float(c_a), float(c_b), float(c_c), float(c_d)])
            self.path_label.append((coefficient, c_file))

        print('instance all:', len(self.path_label))

    def __getitem__(self, index):

        c_perpendicular, c_file = self.path_label[index]
        point_set = np.loadtxt(c_file)  # (x, y, z, mad, adj, pt)

        try:
            choice = np.random.choice(point_set.shape[0], self.n_points, replace=False)
        except:
            exit(f'insufficient point number of the point cloud: all points: {point_set.shape[0]}, required points: {self.n_points}')

        point_set = point_set[choice, :]
        xyz = point_set[:, :3]

        return xyz, c_perpendicular

    def __len__(self):
        return len(self.path_label)


class ConeDataset(Dataset):
    def __init__(self, root, is_train, n_points):

        print('cone dataset, from:' + root)
        self.n_points = n_points

        if is_train:
            inner_root = os.path.join(root, 'train')
        else:
            inner_root = os.path.join(root, 'test')

        self.path_label = []  # [([x, y ,z], Path1), ...]
        files_all = utils.get_allfiles(inner_root)

        for c_file in files_all:
            c_base = os.path.basename(c_file)
            c_base = os.path.splitext(c_base)[0]

            apex_x, apex_y, apex_z, axis_x, axis_y, axis_z, perp_x, perp_y, perp_z, semi_angle, foot_to_apex = c_base.split('_')

            coefficient = np.array([float(apex_x), float(apex_y), float(apex_z), float(axis_x), float(axis_y), float(axis_z), float(perp_x), float(perp_y), float(perp_z), float(semi_angle), float(foot_to_apex)])
            self.path_label.append((coefficient, c_file))

        print('instance all:', len(self.path_label))

    def __getitem__(self, index):
        c_coefficient, c_file = self.path_label[index]
        point_set = np.loadtxt(c_file)  # (x, y, z, mad, adj, pt)

        try:
            choice = np.random.choice(point_set.shape[0], self.n_points, replace=False)
        except:
            exit(f'insufficient point number of the point cloud: all points: {point_set.shape[0]}, required points: {self.n_points}')

        point_set = point_set[choice, :]
        xyz = point_set[:, :3]

        return xyz, c_coefficient

    def __len__(self):
        return len(self.path_label)


class CstNet2Dataset(Dataset):
    """
    CstNet2 具备五个属性的数据集读取
    """
    def __init__(self,
                 root,
                 is_train=True,
                 n_points=2000,
                 data_augmentation=False
                 ):
        """
        定位文件的路径如下：
        root
        ├─ train
        │   ├─ Bushes
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ├─ Clamps
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ...
        │
        ├─ test
        │   ├─ Bushes
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ├─ Clamps
        │   │   ├─0.obj
        │   │   ├─1.obj
        │   │   ...
        │   │
        │   ...
        │

        """
        print('CstNet2 dataset, from:' + root)

        self.n_points = n_points
        self.data_augmentation = data_augmentation

        if is_train:
            inner_root = os.path.join(root, 'train')
        else:
            inner_root = os.path.join(root, 'test')

        category_all = utils.get_subdirs(inner_root)
        category_path = {}  # {'plane': [Path1,Path2,...], 'car': [Path1,Path2,...]}

        for c_class in category_all:
            class_root = os.path.join(inner_root, c_class)
            file_path_all = utils.get_allfiles(class_root)
            category_path[c_class] = file_path_all

        self.datapath = []
        for item in category_path:
            for fn in category_path[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(sorted(category_path), range(len(category_path))))
        print(self.classes)
        print('instance all:', len(self.datapath))

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[fn[0]]
        point_set = np.loadtxt(fn[1])

        # 随机选出指定数量的点
        try:
            choice = np.random.choice(point_set.shape[0], self.n_points, replace=False)
            point_set = point_set[choice, :]
        except:
            exit(f'insufficient points, current: {point_set.shape[0]}, required: {self.n_points}')

        xyz = point_set[:, :3]  # [n, 3]
        pmt = point_set[:, 3].astype(np.int32)  # 基元类型 [n, ]. plane 0, cylinder 1, cone 2, sphere 3, freeform 4
        mad = point_set[:, 4:7]  # 主方向 [n, 3]
        dim = point_set[:, 7]  # 主尺寸 [n, ]
        nor = point_set[:, 8:11]  # 法线 [n, 3]
        loc = point_set[:, 11:14]  # 主位置 [n, 3]
        affiliate_idx = point_set[:, 14].astype(np.int32)  # 从属索引 [n, ]

        # 已弃用在加载时调整点云，直接在点云生成时归一化三维模型，使其处于 [-1, 1]^3
        # # 质心平移到原点，三轴范围缩放到 [-1, 1]^3
        # move_dir = -np.mean(xyz, axis=0)
        # xyz = xyz + move_dir
        # scale = 1.0 / np.max(np.sqrt(np.sum(xyz ** 2, axis=1)), 0)
        # xyz = xyz * scale
        #
        # # 平移缩放后，pmt, mad, nor 不变，dim 除圆锥外与原本进行相同比例缩放，loc 先平移，再缩放
        # dim = update_dim(pmt, dim, scale)
        # loc = update_loc(pmt, loc, mad, move_dir)
        # loc = loc * scale

        # 将圆锥的主位置替换为原点到轴线的垂足坐标
        # cone_mask = (pmt == 2)
        # if cone_mask.sum() != 0:
        #     cone_mad = mad[cone_mask]
        #     cone_apex = loc[cone_mask]
        #
        #     t = - np.einsum('ij,ij->i', cone_mad, cone_apex)
        #
        #     # 垂足坐标
        #     perpendicular_foot = cone_apex + t[:, None] * cone_mad
        #     loc[cone_mask] = perpendicular_foot

        if self.data_augmentation:
            xyz += np.random.normal(0, 0.02, size=xyz.shape)

        return xyz, cls, pmt, mad, dim, nor, loc, affiliate_idx

    def __len__(self):
        return len(self.datapath)

    def n_classes(self):
        return len(self.classes)


def trans_loc_for_planes(pmt: np.ndarray, loc: np.ndarray, mad: np.ndarray, trans: np.ndarray, eps: float = 1e-8):
    """
    对于平面，点云平移 trans = (a, b, c) 后，原点到该平面的垂足应变为 原点到旧平面的垂足 加上平移向量在法向方向的投影

    pmt: [n, ]
    loc: [n, 3]
    mad: [n, 3]
    trans: [3, ]
    """

    denom = np.linalg.norm(mad, axis=1) ** 2  # ||n||^2 for each normal, [n, ]
    dots = mad.dot(trans)  # [n, ]  # n·t for each row

    # 获取有效位置
    is_plane = (pmt == 0)  # [n, ]
    valid = is_plane & (denom > eps)  # [n, ]
    updated_idx = np.where(valid)[0]  # [n_valid, ]

    if updated_idx.size > 0:
        scalars = dots[updated_idx] / denom[updated_idx]  # [n, ]
        deltas = scalars[:, None] * mad[updated_idx]  # [n, 3]
        loc[updated_idx] += deltas

    return loc


def trans_loc_for_cylinders(pmt, loc, mad, trans):
    """
    更新圆柱几何体的loc

    参数:
        pmt: [n, ] numpy数组，几何体类型 (0=plane, 1=cylinder, ...)
        loc: [n, 3] numpy数组，原点到几何体的垂足
        mad: [n, 3] numpy数组，几何体的法线/轴向（需为单位向量）
        translation: (a, b, c) 平移向量
    返回:
        loc_updated: [n, 3] numpy数组，更新后的垂足
    """

    # 找到所有圆柱
    cyl_mask = (pmt == 1)

    if np.any(cyl_mask):
        d = mad[cyl_mask]  # 轴向
        d = d / np.linalg.norm(d, axis=1, keepdims=True)  # 单位化
        p0 = loc[cyl_mask]  # 原垂足

        # 投影到轴方向
        proj = np.sum(trans * d, axis=1, keepdims=True) * d

        # 新的垂足
        loc[cyl_mask] = p0 + (trans - proj)

    return loc


def update_dim(pmt, dim, scale):
    # 找到圆锥
    mask_plane = (pmt == 2)
    other_pmt = ~mask_plane
    dim[other_pmt] = dim[other_pmt] * scale

    return dim


def update_loc(pmt, loc, mad, trans):
    """
    更新几何体loc

    参数:
        pmt: [n, ] numpy数组，几何体类型 (0=plane, 1=cylinder, 其它=直接平移)
        loc: [n, 3] numpy数组，原点到几何体的垂足
        mad: [n, 3] numpy数组，几何体的法线/轴向（需为单位向量）
        translation: (a, b, c) 平移向量
    返回:
        loc_updated: [n, 3] numpy数组，更新后的垂足
    """
    loc_updated = loc.copy()

    # ---------- 平面 ----------
    mask_plane = (pmt == 0)
    if np.any(mask_plane):
        n = mad[mask_plane]
        proj = np.sum(trans * n, axis=1, keepdims=True) * n
        loc_updated[mask_plane] = loc[mask_plane] + proj

    # ---------- 圆柱 ----------
    mask_cyl = (pmt == 1)
    if np.any(mask_cyl):
        d = mad[mask_cyl]
        proj = np.sum(trans * d, axis=1, keepdims=True) * d
        loc_updated[mask_cyl] = loc[mask_cyl] + (trans - proj)

    # ---------- 其它 ----------
    mask_other = ~(mask_plane | mask_cyl)
    if np.any(mask_other):
        loc_updated[mask_other] = loc[mask_other] + trans

    return loc_updated


def single_load(pcd_file):
    point_set = np.loadtxt(pcd_file)

    xyz = point_set[:, :3]  # [n, 3]
    pmt = point_set[:, 3].astype(np.int32)  # 基元类型 [n, ]
    mad = point_set[:, 4:7]  # 主方向 [n, 3]
    dim = point_set[:, 7]  # 主尺寸 [n, ]
    nor = point_set[:, 8:11]  # 法线 [n, 3]
    loc = point_set[:, 11:14]  # 主位置 [n, 3]
    affil_idx = point_set[:, 14]  # 从属索引 [n, ]

    # 质心平移到原点，三轴范围缩放到 [-1, 1]^3
    move_dir = -np.mean(xyz, axis=0)
    xyz = xyz + move_dir
    scale = 1.0 / np.max(np.sqrt(np.sum(xyz ** 2, axis=1)), 0)
    xyz = xyz * scale

    # 平移缩放后，pmt, mad, nor 不变，dim 除圆锥外与原本进行相同比例缩放，loc 先平移，再缩放
    dim = update_dim(pmt, dim, scale)
    loc = update_loc(pmt, loc, mad, move_dir)
    loc = loc * scale

    return xyz, pmt, mad, dim, nor, loc, affil_idx


def test():
    afile = r'C:\Users\ChengXi\Desktop\cstnet2\comb.txt'
    xyz, pmt, mad, dim, nor, loc, affil_idx = single_load(afile)



if __name__ == '__main__':
    test()




    pass

