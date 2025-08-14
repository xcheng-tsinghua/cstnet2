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



