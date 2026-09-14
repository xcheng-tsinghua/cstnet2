# Stage1 三阶段直接约束预测

输入 XYZ（可选在线曲率、密度特征），共享骨干输出逐点特征：

- 类型头：五类 primitive_type。
- 聚类特征头：仅用于 affiliate_idx 监督下的判别损失。
- 三个独立 MLP：mad（方向）、dim（尺寸）、loc（位置）。

已删除主流程中的共享 geometry_decoder、AMP、训练期真实/Oracle 聚类指标、每轮全量拟合评估，以及训练/评估/导出/冻结提取器中的拟合参数。最终输出不依赖聚类结果。已有独立基线实验和历史几何工具保留，不属于该入口的执行路径。

## 训练

以下命令按顺序运行，把 DATASET 换成相同的数据目录；模型、输入特征和 checkpoint_root 在三个阶段保持一致。

```shell
python train_cst_pred.py --data_root DATASET --train_phase semantic
python train_cst_pred.py --data_root DATASET --train_phase geometry
python train_cst_pred.py --data_root DATASET --train_phase joint
```

| 阶段 | 更新的模块 | 损失 | 默认学习率 |
| --- | --- | --- | --- |
| semantic | 骨干 + 类型头 + 聚类特征头 | pmt + cluster（2 项） | 1e-4 |
| geometry | 仅 mad_head、dim_head、loc_head | mad + dim + loc（3 项） | 1e-4 |
| joint | 整个骨干 + 全部预测头 | pmt + cluster + mad + dim + loc（5 项） | 全部参数统一 1e-5 |

可用 --lr 显式覆盖学习率。geometry 冻结骨干及语义头的权重和 BatchNorm 状态；joint 全部解冻并恢复训练模式。Stage2 仍然冻结整个 Stage1。

当前训练方案为 `simple_five_losses_v1`：pmt 使用 NLL（等价于对 logits 的交叉熵），cluster 保留已有判别式聚类损失；mad 使用归一化方向的正反等价 MSE，dim 和 loc 使用 MSE。属性损失只在 GT 基元类型对应的有效点上取平均，不做实例均衡或可观测性降权。各项权重默认均为 1，可通过 `--w_pmt`、`--w_cluster`、`--w_mad`、`--w_dim`、`--w_loc` 设置正权重。

不计算几何残差、实例一致性和辅助损失 ramp；不提供这些参数或单项损失关闭开关。非当前阶段的损失完全跳过，日志只记录当前阶段的损失项。现有网络结构、头宽度、输入特征配置及最终约束规范化规则不变。

## 保存与续训

### Hugging Face 数据集

训练入口的 `--data_root` 同时支持本地路径和 Hugging Face 数据集链接：

```shell
python -m pip install huggingface_hub
python train_cst_pred.py --data_root "https://huggingface.co/datasets/ZXCCHENGXI/cstnet2_s1_small_v2/tree/main" --train_phase semantic
```

第二、三步使用同一链接，把 `--train_phase` 分别改为 `geometry`、`joint`。
先下载缺失的 TXT/HDF5 文件到标准 Hugging Face 缓存，再启动 DataLoader；
后续运行复用未变化的文件，仍会检查远端版本。可用 `--hf_cache_dir /data/hf_cache`
指定缓存位置；不指定时遵循 Hugging Face 默认缓存配置（包括 HF_HOME）。
`--data_format h5` 或 `txt` 可限制下载格式。只提供本地路径时无需安装 huggingface_hub。

支持仓库首页、`/tree/<revision>` 和其子目录链接；不支持压缩包自动解压或文件预览链接。
需要固定数据版本时使用 commit 对应的 tree 链接，避免 main 更新后改变训练数据。
下载失败会中止启动，可原命令重试。私有数据集使用 Hugging Face 登录状态或 HF_TOKEN，
不要把令牌写进 URL。checkpoint 同时记录原始链接和实际缓存路径。

### Checkpoint

默认目录为 model_trained/stage1_direct/<model>/<phase>。不覆盖旧 model_trained/stage1。

- 默认 auto：本阶段 last.pth 存在时续训；否则第二步从第一步最佳权重初始化，第三步从第二步 last.pth 初始化。
- --checkpoint_policy restart：重启本阶段（第二、三步仍从前一阶段初始化）。
- checkpoint 额外记录 `training_recipe=simple_five_losses_v1`。旧方案的 checkpoint 仍可作为前一阶段的模型权重来源，但不能恢复旧优化器并混入新方案续训。已有 semantic 权重可复用；若 geometry 已按旧方案训练，使用下面的命令从 semantic 权重重新训练 geometry。该命令会更新原 geometry 目录的 checkpoint；若要保留旧 geometry 实验，请先备份该目录。

  ```shell
  python train_cst_pred.py --data_root DATASET --train_phase geometry --checkpoint_policy restart
  python train_cst_pred.py --data_root DATASET --train_phase joint
  ```

  若 joint 目录也已有旧方案的 checkpoint，joint 同样加 `--checkpoint_policy restart`。
- 新结构删除了共享几何解码器，旧结构权重不能直接续训或推理，需要重新运行三步。
- checkpoint 写入 constraint_route=direct_mlp_v1。导出和冻结提取器要求 geometry/joint 权重，拒绝尚未训练属性头的 semantic 权重。
- 保留 last.pth、best_pmt_miou.pth、best_constraint_score.pth。constraint_score 为当前阶段总训练损失的负值，不使用聚类 ARI；它是训练集指标，不是验证集精度，且受损失权重影响。

## 最终输出与评估

### 输出与存储 I/O 故障

`train_cst_pred.py` 对控制台打印、Colorama/W&B 控制台捕获和进度条的输出 I/O 错误做容错，输出恢复后提示期间丢失的输出操作。checkpoint 与训练 JSON 日志使用临时文件和原子替换，失败后最多尝试三次；警告打印失败也不会打断重试。最终失败时继续训练并保留此前的目标文件，下一轮再次尝试保存。W&B 指标记录遇到 `OSError` 时跳过当轮上传，下一轮继续尝试；JSON 日志恢复后会写入内存中累计的训练记录。

若 `last.pth` 已成功保存，使用原训练命令加 `--checkpoint_policy resume` 续训，保持原模型、数据、阶段和训练配置。不要使用 `restart` 来处理此次输出故障。模型计算、数据读取、非有限数值和配置错误仍会正常中止；持续的存储故障仍会导致无法保存新权重，需要排查服务器磁盘或挂载状态。

四个分量拼接为 [B,N,12]：5 + 3 + 1 + 3。类型 argmax 后 one-hot；方向单位化并统一符号；平面位置沿法向投影，圆柱位置去掉轴向分量；半径为正、半角位于 (0, pi/2)，无效属性置零。

```shell
python eval_cst_pred.py --data_root DATASET --checkpoint model_trained/stage1_direct/attn_3dgcn/joint/last.pth
python gen_cst_pred.py --input_dir INPUT --output_dir OUTPUT --checkpoint model_trained/stage1_direct/attn_3dgcn/joint/last.pth
```

评估报告保留原始预测头属性误差，direct_* 为规范化后的最终约束误差；训练日志中的属性误差对应最终约束。方向/尺寸/位置误差仍按 GT 类型的有效属性掩码统计，类型正确率单独报告。

评估损失同样使用简化五项方案，并在 JSON 中记录 `evaluation_loss_recipe`。旧方案权重也可评估，但新计算的 loss 不应与旧日志中的正则化总 loss 直接比较。

导出的 TXT 保持 xyz,pmt,mad,dim,loc,affiliate_idx 共 12 列的基础格式，附加任务列按原规则保留。由于不预测实例编号，affiliate_idx 固定为 -1，不能当成实例训练标签。Stage2 仅使用四个约束分量，并始终冻结 Stage1。
