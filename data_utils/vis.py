import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch


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

def vis_3d_points_knn(points: torch.Tensor, index: int, neighbors: torch.IntTensor):
    dim = points.dim()
    print("points.dim: ", dim)
    # 如果有batch维度，选择第一个batch
    if dim == 3:
        points = points[0, :, :]
        neighbors = neighbors[0, :, :]
        print("可视化batch中的第一个点云")

    # 将torch.Tensor转为numpy数组
    points = points.detach().cpu().numpy()
    neighbors = neighbors.detach().cpu().numpy()

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # 3d points全部画出来
    print("points: ", points[:, :].shape)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c='dodgerblue',  # 颜色
               s=15,  # 点大小
               marker='o',
               alpha=0.4)

    # 画出来被索引的点points[index]
    print("index points: ", points[index, :].shape)
    ax.scatter(points[index, 0], points[index, 1], points[index, 2],
               c='r',  # 颜色
               s=20,  # 点大小
               marker='*',
               alpha=0.4)

    # 画出来所有的临近点 注意neighbors [N, K] -> 索引
    print("neighbors: ", points[neighbors[index], :].shape)
    ax.scatter(points[neighbors[index], 0], points[neighbors[index], 1], points[neighbors[index], 2],
               c='k',  # 颜色
               s=17,  # 点大小
               marker='o',
               alpha=0.4)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    vis_pcd(r'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend\test\belt_wheel\9N_J400X38X2X76X76.txt')
