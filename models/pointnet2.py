import torch.nn as nn
import torch.nn.functional as F
import torch


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    返回局部点索引                   torch.Size([batch_size, n_centroid,  n_local_sample])   sa1: torch.Size([16, 512, 32])
    xyz：    点云中全部点            torch.Size([batch_size, n_all_point, channel])          sa1: torch.Size([16, 2048, 3])
    new_xyz：点云中当作采样球心的点    torch.Size([batch_size, n_centroid,  channel])          sa1: torch.Size([16, 512, 3])
    radius： 采样局部区域半径
    nsample：每个局部区域最大采样点数

    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def farthest_point_sample(xyz, n_samples):
    """
    最远采样法进行采样，返回采样点的索引
    Input:
        xyz: pointcloud data, [batch_size, n_points_all, 3]
        n_samples: number of samples
    Return:
        centroids: sampled pointcloud index, [batch_size, n_samples]
    """
    device = xyz.device

    # xyz: [24, 1024, 3], B: batch_size, N: number of points, C: channels
    B, N, C = xyz.shape

    # 生成 B 行，n_samples 列的全为零的矩阵
    centroids = torch.zeros(B, n_samples, dtype=torch.long).to(device)

    # 生成 B 行，N 列的矩阵，每个元素为 1e10
    distance = torch.ones(B, N).to(device) * 1e10

    # 生成随机整数tensor，整数范围在[0，N)之间，包含0不包含N，矩阵各维度长度必须用元组传入，因此写成(B,)
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    # 生成 [0, B) 整数序列
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(n_samples):
        centroids[:, i] = farthest

        # print('batch_indices', batch_indices.shape)
        # print('farthest', farthest.shape)
        # print('xyz', xyz[batch_indices, farthest, :].shape)
        # exit()

        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)

        # print('xyz', xyz.shape)
        # print('centroid', centroid.shape)
        # exit()

        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask].float()
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """
    返回 points 中 索引 idx 对应的点
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    采样并以采样点为圆心集群
    Input:
        npoint: 最远采样法的采样点数，即集群数
        radius: 集群过程中的半径
        nsample: 每个集群中的点数
        xyz: input points position data, [B, N, 3] 点坐标
        points: input points data, [B, N, D] 法向量
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    # xyz: [24, 1024, 3], B: batch_size, N: number of points, C: channels
    B, N, C = xyz.shape
    S = npoint # 对原始点云进行采样后获得的点数
    fps_idx = farthest_point_sample(xyz, npoint) # 采样后的点索引 troch.size([B, npoint])
    new_xyz = index_points(xyz, fps_idx) # 获取 xyz 中，索引 fps_idx 对应的点
    idx = query_ball_point(radius, nsample, xyz, new_xyz) # 以 new_xyz 为圆心，半径为 radius，在xyz中获取 nsample 个点，返回其索引 [B, S, nsample]
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    返回 0：全为零的 tensor，shape 为 [batch_size, 1, channels]
    返回 1：先把xyz view成[B, 1, N, C]，xyz输入时是[B, N, C]，然后返回(如果points为none的话)

    Input:
        xyz: 输入的 xyz 位置特征 torch.Size([24, 1024, 3])
        input points position data, [B, N, 3]

        points: 输入的 ijk 法向量特征
        input points data, [B, N, D]
    Return:
        new_xyz: 全为零的 tensor，shape 为 [24, 1, 3]
        sampled points position data, [B, 1, 3]

        new_points: 把输入的 xyz 和 points 拼在一起，先 view 成 [24, 1, 1024, -1]
        若 points 没有，即为 none，则不拼接，仅仅 view

        sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device

    # B: batch_size, N: point number, C: channels
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """
    set abstraction 层
    包含sampling、grouping、PointNet层
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        '''
        :param npoint: 使用FPS查找的中心点数，因此该点数也是池化到的点数
        :param radius: 沿每个中心点进行 ball query 的半径
        :param nsample: 每个ball里的点数最大值，感觉就是查找这个数目的点，和半径无关
        :param in_channel: 输入特征维度
        :param mlp: list，表示最后接上的 MLP 各层维度
        :param group_all: 是否将全部特征集中到一个点
        '''
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp: # mlp：数组
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: 点的 xyz 特征。input points position data, [B, C, N]
            points: 点的 ijk 特征。input points data, [B, D, N]
        Return:
            new_xyz: 处理后的 xyz 特征。sampled points position data, [B, C, S]
            new_points_concat: 处理后的 ijk 特征。sample points feature data, [B, D', S]
        """
        # 交换 xyz 的 1,2 维度，交换之前：torch.Size([24, 3, 1024])
        xyz = xyz.permute(0, 2, 1)

        # 对法向量特征进行处理
        if points is not None:
            points = points.permute(0, 2, 1)

        # 前两层该参数为 false，最后一层为 true
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            # xyz: torch.Size([24, 1024, 3])
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points normal data, [B, npoint, nsample, C+D]

        new_points = new_points.permute(0, 3, 2, 1).to(torch.float)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # 每个采样点的邻近点的特征维度取最大值
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class PointNet2(nn.Module):
    # num_class = 40，normal_channel = false
    def __init__(self, num_class: int, fea_channel=0):
        super().__init__()

        # 三个 set abstraction 层，in_channel = 3,
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=fea_channel + 3, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz, fea=None):
        """
        xyz: [bs, 3, n_points]
        """
        B, _, _ = xyz.shape

        l1_xyz, l1_points = self.sa1(xyz, fea)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x


if __name__ == '__main__':
    x = torch.rand((2, 3, 1024)).cuda()
    featensor = torch.rand((2, 7, 1024)).cuda()

    anet = PointNet2(20).cuda()
    y = anet(x)
    print("Input Shape of PointNet: ", x.shape, "\nOutput Shape of PointNet: ", y.shape)






