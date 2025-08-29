import os
import numpy as np
from torch.utils.data import Dataset

from models import utils


class CstPntDataset(Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 data_augmentation=True
                 ):
        print('CstPnt dataset, from:' + root)

        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.datapath = utils.get_allfiles(root)

        print('instance all:', len(self.datapath))

    def __getitem__(self, index):
        fn = self.datapath[index].strip()
        point_set = np.loadtxt(fn)  # [x, y, z, ex, ey, ez, adj, pt]

        try:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        except:
            exit(f'insufficient point number of the point cloud: all points: {point_set.shape[0]}, required points: {self.npoints}')

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
                 npoints=2500,
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

        self.npoints = npoints
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
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        except:
            exit(f'insufficient point number of the point cloud: all points: {point_set.shape[0]}, required points: {self.npoints}')

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
    def __init__(self, root, is_train, npoints):

        print('RegressionDataset dataset, from:' + root)
        self.npoints = npoints

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
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        except:
            exit(f'insufficient point number of the point cloud: all points: {point_set.shape[0]}, required points: {self.npoints}')

        point_set = point_set[choice, :]
        xyz = point_set[:, :3]

        return xyz, c_perpendicular

    def __len__(self):
        return len(self.path_label)


class CstNet2Dataset(Dataset):
    """
    CstNet2 具备五个属性的数据集读取
    """
    def __init__(self,
                 root,
                 is_train=True,
                 npoints=2000,
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

        self.npoints = npoints
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
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
            point_set = point_set[choice, :]
        except:
            exit(f'insufficient point number of the point cloud: all points: {point_set.shape[0]}, required points: {self.npoints}')

        xyz = point_set[:, :3]
        pmt = point_set[:, 3]  # 基元类型
        main_dir = point_set[:, 4:7]  # 主方向
        main_dim = point_set[:, 7]  # 主尺寸
        normal = point_set[:, 8:11]  # 法线
        main_loc = point_set[:, 11:14]  # 主位置
        affil_idx = point_set[:, 14]  # 从属索引

        # scale points to [-1, 1]^2
        xyz = xyz - np.expand_dims(np.mean(xyz, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)), 0)
        xyz = xyz / dist

        if self.data_augmentation:
            xyz += np.random.normal(0, 0.02, size=xyz.shape)

        return xyz, cls, pmt, main_dir, main_dim, normal, main_loc, affil_idx

    def __len__(self):
        return len(self.datapath)

    def n_classes(self):
        return len(self.classes)


if __name__ == '__main__':
    pass

