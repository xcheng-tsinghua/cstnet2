# 1. 项目概述
我写了一个点云项目，利用约束实现机械零件点云的分类与生成。

# 2. 点云的约束定义
点云的约束表达为逐点形式，即每个点都有对应的向量，描述其约束，即每个点表达为 (x, y, z, constraint)

其中 constraint 包含如下五项内容：
1. primitive_type: 基元类型，即该点所在的基元的类型。例如某个点事从CAD模型上的一个圆柱面上采样获得的，那么它的基元类型是圆柱，用 one-hot vector 表示。
```
plane: (1, 0, 0, 0, 0)
cylinder: (0, 1, 0, 0, 0)
cone: (0, 0, 1, 0, 0)
sphere: (0, 0, 0, 1, 0)
free-form surface (and other): (0, 0, 0, 0, 1)
```

2. direction: 主方向，即该点所在的基元的方向，用三维向量表示。
```
plane: normal (需要通过 dir_unify 函数处理保证方向唯一性)
cylinder: rotation axis (需要通过 dir_unify 函数处理保证方向唯一性)
cone: rotation axis (需要通过 dir_unify 函数处理保证方向唯一性)
sphere: (0, 0, -1) (表示没有主方向)
free-form surface (and other): (0, 0, -1) (表示没有主方向)

def dir_unify(direction):
    ax_x = direction.X()
    ax_y = direction.Y()
    ax_z = direction.Z()

    zero_lim = precision.Confusion()
    if ax_z < -zero_lim:  # z < 0 时, 反转
        direction *= -1.0
    elif abs(ax_z) <= zero_lim and ax_y < -zero_lim:  # z为零, y为负数, 反转
        direction *= -1.0
    elif abs(ax_z) <= zero_lim and abs(ax_y) <= zero_lim and ax_x < -zero_lim:  # z为零, y为零, x为负数, 反转
        direction *= -1.0
    else:
        # 无需反转
        pass

    return direction

```

3. dimension: 主尺寸，即该点所在的基元的尺寸，用 float 类型实数表示。
```
plane: -1.0  (表示没有主尺寸)
cylinder: radius
cone: semiAngle
sphere: radius
free-form surface (and other): -1.0  (表示没有主尺寸)
```

4. continuity: 连续性，用机械零件上该点所在位置的法线表示，用三维向量表示。

5. location: 基元位置，即该点所在的基元的位置，用三维向量表示
```
plane: 从原点作平面的法线，垂足位置未基元位置
cylinder: 从原点作圆柱旋转轴的法线，垂足位置未基元位置
cone: 圆锥顶点坐标
sphere: 球心坐标
free-form surface (and other): (0, 0, 0)  (表示没有基元位置)
```

# 3. 模型运行方式
总体运行路径为：输入点云，一阶段通过点云，获得逐点的约束表达。二阶段利用点云和约束表达，实现对点云的高精度分类与生成。
## 3.1. 一阶段运行方式
### 3.1.1. 获取逐点基元类型和逐点聚类特征
模型输入三维点云 tensor.size([bs, npoints, 3]), 模型输出两个数据，一是逐点基元类型，而是逐点聚类特征。聚类特征的训练目标是将属于同一基元的点的聚类特征聚集在一起，属于不同基元的点的聚了特征相互远离，类似聚类训练。

训练数据集中可获取如下信息（见data_utils/datasets.py中的 CstNet2Dataset 类）：
```
xyz: 坐标
cls: 类别
pmt: 基元类型
mad: 主方向
dim: 主尺寸
nor: 法线
loc: 主位置
affiliate_idx: 点所属基元的序号，可用于聚类训练
```

### 3.1.2. 获取逐点约束表达
获取逐点基元类型和聚类特征后，进行点的聚类，同一个点簇里的点属于同一个基元，之后利用逐点基元类型，确定每一个点簇对应的基元类型。之后进行基元拟合，拟合完毕后根据约束表达定义计算逐点的约束表达。


## 3.2. 二阶段运行方式
大概设计这样一个点云encoder和decoder，模型输入点云和约束表达，然后模型根据点云的xyz和约束表达的五项内容，分别提取五项约束特征。然后将五项约束特征合并时，使用注意力机制，以让不同点关注到不同的约束特征。encoder需要下采样，decoder使用上采样。

进行分类时，仅用encoder，获得全局特征。
进行点云生成时，使用encoder和decoder，获取逐点特征，使用Diffusion结构生成


