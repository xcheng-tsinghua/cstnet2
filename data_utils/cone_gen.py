import math
import os.path
import shutil

import numpy as np
import matplotlib.pyplot as plt
import random

import torch
from tqdm import tqdm


class Cone(object):
    """
    一个圆锥使用如下量定义：
    顶点 apex (x, y, z)
    轴线方向 axis (x, y, z). len = 1
    半锥角 semi_angle (float / rad)
    """
    def __init__(self, apex, axis, semi_angle, points=None):
        """
        apex: np.array.size[3]
        axis: np.array.size[3]
        semi_angle: float
        points: np.array.size[n, 3]
        """
        super().__init__()

        self.apex = apex
        self.axis = axis
        self.unify_axis()
        self.semi_angle = semi_angle
        self.points = points

        # 计算原点到轴线的垂足
        t = - np.dot(self.apex, self.axis)
        self.foot_to_apex = -t
        self.perp_foot = self.apex + t * self.axis

    def unify_axis(self, eps=1e-6):
        """
        保证 轴线方向唯一，同一轴线可定义两个向量
        """
        # 先归一化
        self.axis = self.axis / np.linalg.norm(self.axis)

        ax_x, ax_y, ax_z = self.axis[0], self.axis[1], self.axis[2]

        if ax_z < -eps:  # z < 0 时, 反转
            self.axis = -1.0 * self.axis
        elif abs(ax_z) <= eps and ax_y < -eps:  # z为零, y为负数, 反转
            self.axis = -1.0 * self.axis
        elif abs(ax_z) <= eps and abs(ax_y) <= eps and ax_x < -eps:  # z为零, y为零, x为负数, 反转
            self.axis = -1.0 * self.axis
        else:
            # 无需反转
            pass

    def sample_points(self, n, theta_range=(0, 2 * np.pi), h_range=(0.5, 2.0), is_normalize=True):
        """
        在圆锥表面采样点
        apex: 顶点坐标 (3,)
        axis: 旋转轴向量 (3,) (需要归一化)
        angle: 半锥角 (弧度)
        n: 采样点数
        theta_range: 角度的范围
        h_range: 采样的高度范围
        is_normalize: 是否将点云缩放平移到 [-1, 1] 之间
        返回: (n,3) 点坐标
        """

        # 找到与 axis 垂直的两个正交向量 u, v
        tmp_vec = np.array([1, 0, 0]) if abs(self.axis[0]) < 0.9 else np.array([0, 1, 0])

        u = np.cross(self.axis, tmp_vec)
        u /= np.linalg.norm(u)
        v = np.cross(self.axis, u)

        # 随机采样高度和角度
        h = np.random.uniform(h_range[0], h_range[1], size=n)
        theta = np.random.uniform(theta_range[0], theta_range[1], size=n)

        # 半径 = h * tan(angle)
        r = h * np.tan(self.semi_angle)

        # 构造点
        points = []
        for hi, ri, ti in zip(h, r, theta):
            dir_vec = self.axis * hi + u * (ri * np.cos(ti)) + v * (ri * np.sin(ti))
            points.append(self.apex + dir_vec)

        points = np.array(points)

        if is_normalize:
            points_new, apex_new, scale, translation = normalize_to_unit_cube(points, self.apex)
            return Cone(apex_new, self.axis, self.semi_angle, points_new)

        else:
            self.points = points
            return self

    def show(self, length=1.5):
        """
        使用 matplotlib 可视化点云和圆锥参数
        length: 绘制圆锥方向线的长度
        """
        if self.points is None:
            raise ValueError('without points')

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # 绘制采样点
        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2],
                   c="blue", s=8, alpha=0.6, label="Points")

        # 绘制 apex
        ax.scatter([self.apex[0]], [self.apex[1]], [self.apex[2]],
                   c="red", s=80, marker="^", label="Apex")

        ax.scatter([self.perp_foot[0]], [self.perp_foot[1]], [self.perp_foot[2]],
                   c="black", s=80, marker="^", label="perp_foot")

        ax.scatter([0], [0], [0],
                   c="green", s=80, marker="^", label="orign")

        # 绘制 axis (方向线)
        axis_line = np.vstack([self.apex, self.apex + self.axis * length])
        ax.plot(axis_line[:, 0], axis_line[:, 1], axis_line[:, 2],
                c="green", linewidth=2, label="Axis")

        # 绘制一个圆截面，帮助直观理解半锥角
        h = length
        r = h * np.tan(self.semi_angle)
        theta = np.linspace(0, 2 * np.pi, 50)

        # 找到两个正交基 (和 sample_points_on_cone 里一样)
        if abs(self.axis[0]) < 0.9:
            tmp = np.array([1, 0, 0])
        else:
            tmp = np.array([0, 1, 0])

        u = np.cross(self.axis, tmp)
        u /= np.linalg.norm(u)
        v = np.cross(self.axis, u)

        circle = []
        for t in theta:
            circle.append(self.apex + self.axis * h + u * (r * np.cos(t)) + v * (r * np.sin(t)))
        circle = np.array(circle)
        ax.plot(circle[:, 0], circle[:, 1], circle[:, 2],
                c="orange", linewidth=1.5, label="Cone base")

        # 设置坐标轴范围
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        set_axes_equal(ax)
        plt.show()

    def save_data(self, dir_root):
        """
        将点坐标及 label 保存到 txt 文件
        dir_root: 存储到的文件夹名
        """
        if self.points is None:
            raise ValueError('without points')

        # 生成文件名，文件名为 apex_axis_SemiAngle_PerpFoot.txt
        file_name = f'{self.apex[0]:.6f}_{self.apex[1]:.6f}_{self.apex[2]:.6f}_{self.axis[0]:.6f}_{self.axis[1]:.6f}_{self.axis[2]:.6f}_{self.perp_foot[0]:.6f}_{self.perp_foot[1]:.6f}_{self.perp_foot[2]:.6f}_{self.semi_angle:.6f}_{self.foot_to_apex:.6f}.txt'
        file_name = os.path.join(dir_root, file_name)
        np.savetxt(file_name, self.points)


def random_unit_vector():
    """生成一个随机单位向量"""
    vec = np.random.normal(size=3)
    return vec / np.linalg.norm(vec)


def generate_cones(n, apex_range=(-100, 100), angle_range=(0.01, np.pi / 2.5)):
    """
    生成 n 个圆锥参数 (apex, axis, angle)

    参数:
        n: int，圆锥数量
        apex_range: tuple，顶点 x y z 坐标范围 (min, max)
        angle_range: tuple，半锥角范围 (rad)
    返回:
        cones: list，每个元素是 (apex, axis, angle)
    """
    cones = set()
    results = []

    while len(results) < n:
        apex = np.random.uniform(apex_range[0], apex_range[1], size=3)
        axis = random_unit_vector()
        angle = np.random.uniform(angle_range[0], angle_range[1])

        # 用 tuple 定义唯一性 key
        key = (tuple(np.round(apex, 4)),
               tuple(np.round(axis, 4)),
               round(angle, 4))

        if key not in cones:
            cones.add(key)
            results.append(Cone(apex, axis, angle))

    return results


def normalize_to_unit_cube(points, apex):
    """
    将点集和 apex 一起平移缩放到 [-1,1]^3
    返回: 变换后的 points, apex, scale, translation
    """
    all_pts = np.vstack([points])
    min_vals = all_pts.min(axis=0)
    max_vals = all_pts.max(axis=0)

    center = (min_vals + max_vals) / 2
    half_range = (max_vals - min_vals).max() / 2

    scale = 1.0 / half_range
    translation = -center

    # 应用变换
    points_new = (points + translation) * scale
    apex_new = (apex + translation) * scale

    return points_new, apex_new, scale, translation


def generate_random_theta_and_h_range(n, theta_len_range=(0.5, 5), h_len_range=(0.5, 3)):
    """
    theta_len: [0, 2 * np.pi], [1 - 4] 为宜
    h_len: [0, 3], [1.5 - 2.5] 为宜
    """
    theta_len = np.random.uniform(low=theta_len_range[0], high=theta_len_range[1], size=n)  # [n, ]
    theta_start_start = 0
    theta_start_end = 2 * np.pi - theta_len  # [n, ]

    random_theta_start = []
    for i in range(n):
        random_theta_start.append(random.uniform(theta_start_start, theta_start_end[i].item()))

    random_theta_start = np.array(random_theta_start)
    random_theta_end = random_theta_start + theta_len
    random_theta = np.stack([random_theta_start, random_theta_end], axis=1)

    h_len = np.random.uniform(low=h_len_range[0], high=h_len_range[1], size=n)  # [n, ]
    h_start_start = 0
    h_start_end = 3
    random_h_start = np.random.uniform(low=h_start_start, high=h_start_end, size=n)  # [n, ]
    random_h_end = random_h_start + h_len
    random_h = np.stack([random_h_start, random_h_end], axis=1)

    return random_theta, random_h


def set_axes_equal(ax):
    """让三维坐标轴等比例"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def show_cones(n_cones=5000, n_points=2000, test_rate=0.2, save_dir=r'D:\document\DeepLearning\DataSet\pcd_cstnet2\cone'):
    # 生成圆锥参数
    random_cones = generate_cones(n_cones)
    random_theta, random_h = generate_random_theta_and_h_range(n_cones)

    n_fail = 0

    train_dir = os.path.join(save_dir, 'train')
    test_dir = os.path.join(save_dir, 'test')

    if os.path.exists(train_dir) or os.path.exists(test_dir):
        user = input(f"数据集文件夹已存在文件，删除它？(yes / no)").strip()
        if user.lower() == 'yes':
            print('delete dir' + train_dir)
            shutil.rmtree(train_dir)
            print('delete dir' + test_dir)
            shutil.rmtree(test_dir)
        else:
            exit(0)

    os.makedirs(train_dir)
    os.makedirs(test_dir)

    n_test = math.ceil(n_cones * test_rate)
    for idx, c_cone in tqdm(enumerate(random_cones), total=n_cones):
        try:
            c_theta = random_theta[idx]
            c_h = random_h[idx]

            new_cone = c_cone.sample_points(n_points, c_theta, c_h)
            # new_cone.show()

            if idx < n_test:
                new_cone.save_data(test_dir)
            else:
                new_cone.save_data(train_dir)

        except:
            n_fail += 1
            print('an error cone')

    print(f'failed file save: {n_fail}')


if __name__ == "__main__":
    show_cones()

    # atenor = torch.tensor([1., 0., 0.])
    # btenor = torch.tensor([0., 1., 0.])
    #
    # print(atenor * btenor)


