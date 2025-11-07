# Constraint Learning for Parametric Point Cloud
Main modules: pytorch, pyocc

# 约束表达定义
带约束表达的每个点定义为 (xyz, cst)

cst = (pmt, dir, dim, nor, loc)

## pmt: Primitive Type 基元类型

plane: 0

cylinder: 1

cone: 2

sphere: 3

freeform: 4

## dir：主方向

## dim：主尺寸

## nor：连续性

## loc：主位置

# 环境
conda env export --no-builds | grep -v '^prefix:' > environment.yml

conda activate dp

# 模块
## modules
包含一些全局特征、局部特征提取模块，例如 pointnet 或 pointnet++ (pointnet2)

## cst_pred
包含从点云获取约束表达的模块

可能需要用到 modules 中的一些模块

cst_pcd.py: 本研究所开发的从点云中提取约束的模块

## cst_fea
输入点云及约束表达，输出约束特征







