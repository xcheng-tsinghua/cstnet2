import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def vis_3dpnts(points):
    # 假设已有 n×3 的数组 points
    # points = np.random.rand(100, 3)  # 这里用随机数据做演示

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # 如果你的数组叫 points，直接写：
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c='dodgerblue',  # 颜色
               s=15,  # 点大小
               marker='o',
               alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')

    plt.tight_layout()
    plt.show()


def vis_pcd_plt(points1, points2=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])

    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], color='black', alpha=0.2)

    if points2 is not None:
        ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], color='red')

    plt.show()


def vis_pcd(pcd_file):
    pnts_all = np.loadtxt(pcd_file)

    xyz = pnts_all[:, :3]
    pmt = pnts_all[:, 3].astype(np.int32)
    mad = pnts_all[:, 4:7]
    dim = pnts_all[:, 7]
    nor = pnts_all[:, 8:11]
    loc = pnts_all[:, 11:14]
    affil_idx = pnts_all[:, 14].astype(np.int32)

    print(f'xmax {xyz[:, 0].max()}, xmin {xyz[:, 0].min()}, ymax {xyz[:, 1].max()}, ymin {xyz[:, 1].min()}, zmax {xyz[:, 2].max()}, zmin {xyz[:, 2].min()}')

    vis_pcd_plt(xyz, loc)


if __name__ == '__main__':
    vis_pcd(r'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend\test\belt_wheel\9N_J400X38X2X76X76.txt')
