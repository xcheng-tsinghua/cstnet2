# Stage 2 分类消融（9 项，单随机种子）

统一入口：`train_cls_ablation.py`。仅对 CSTNet2 的 Stage 2 分类模型做消融，读取已有 Stage 1 预测约束缓存，不涉及分割或 Stage 1 训练。

## 实验列表

所有实验保留 XYZ、五条特征流、网络宽度、注意力结构及分类头。被移除分量在进入全部编码器、token 和统计分支之前置零，因此是同容量的信息消融。

| --experiments | 保留的约束 |
|---|---|
| xyz_only | 无 |
| no_primitive_type | direction、dimension、location |
| no_direction | primitive_type、dimension、location |
| no_dimension | primitive_type、direction、location |
| no_location | primitive_type、direction、dimension |
| only_primitive_type | primitive_type |
| only_direction | direction |
| only_dimension | dimension |
| only_location | location |

`all` 为这 9 项的快捷选择，每项仅训练一次。默认 xyz_only。可以一次指定多个实验名称，重复名称自动去重。

方法参考 CSTNet 第 4.4 节表 3 的输入置零对照，扩展到四分量的逐项移除和单项保留。xyz_only 仍保留全部特征流，但这些流只能从 XYZ 学习。完整模型使用项目原分类训练入口作为参照。

## 数据划分：完全使用已有 train/test

数据获取使用与 `train_cls.py` 相同的 `Stage2ClassificationDataset.create_dataloaders` 工厂，读取全部训练和测试样本，保持类别映射一致。支持：

- TXT：已有 `root/train/<class>/...txt` 和 `root/test/<class>/...txt`。
- TXT：已有 `root/<class>/...txt`，由 `split_file.json` 或兼容的 `split_file` 文件指定 train/test。
- HDF5：已有分类 train/test 分片与元数据。

**不重新划分，不创建划分文件，不从 train 留出验证集。** TXT 缺少上述已有划分时直接报错。移除了 val_ratio、test_ratio、split_seed 等划分参数以及分割相关参数。

与 `train_cls.py` 一致：每轮训练后在 test 上评估，按 test instance_accuracy 保存 best.pth（相等时也更新）。日志只有 train/test，无 val。训练使用全部 train，测试使用全部 test，DataLoader 不丢弃最后一批。

点采样只是对每个模型内部的点选择，不改变模型所属的训练/测试集合。训练点采样由 seed、epoch、样本索引确定；测试点采样和评估 FPS 随机数固定，以便比较不同消融。

## 参数与运行

`--seed` 是单个随机种子，默认 42；没有多种子循环，也不计算多种子均值和标准差。相同对照组请使用同一 seed 和超参数。

在项目根目录、已有训练依赖的 Python 环境中执行，替换 data_root：

```bash
python train_cls_ablation.py --list
python train_cls_ablation.py --experiments all --seed 42 --dry_run

# 单项分类消融
python train_cls_ablation.py --data_root /data/param20k_pred --experiments xyz_only --seed 42
python train_cls_ablation.py --data_root /data/param20k_pred --experiments no_location --seed 42

# 四个 no_*，每项训练一次
python train_cls_ablation.py --data_root /data/param20k_pred --experiments no_primitive_type no_direction no_dimension no_location --seed 42

# 全部 9 项，每项使用同一个种子
python train_cls_ablation.py --data_root /data/param20k_pred --experiments all --seed 42
```

不再需要 `--task`；入口只支持分类。原 --seeds 已移除，改变单次随机种子可使用 --seed 123。

默认 2048 点、batch=20、200 epochs、FP32；Adam + StepLR(20,0.7)，lr=1e-4、weight decay=1e-4，label smoothing=0.05，两个辅助头权重各 0.1。可通过参数调整，同组保持一致。

数据支持现有 12 列分类 TXT/HDF5，文件中的 primitive_type ID 会转为五维 one-hot，模型约束总维数为 12。所有对照应使用同一冻结 Stage 1 和预处理生成的预测缓存。约束直接读取自 data_root 指定的数据文件。

## 恢复、评估与结果

```bash
# 用原训练参数恢复
python train_cls_ablation.py --data_root /data/param20k_pred --experiments no_location --seed 42 --resume auto

# 自动加载 best.pth 保存的配置和数据路径，在已有 test 上复评
python train_cls_ablation.py --mode evaluate --experiments no_location --seed 42

# 导出指定 seed 下每个分类实验的结果
python train_cls_ablation.py --mode summarize --seed 42
```

`--resume` 为空字符串时（默认，可省略），即使已有结果也从第 1 轮重新训练，覆盖当前实验目录的 checkpoint、配置和结果，并清除旧的逐轮日志与评估文件。重新训练创建新的 W&B Run，已有线上 Run 保留；其他实验目录不受影响。

`--resume auto` 会尝试恢复每个实验目录中的 last.pth；文件缺失、损坏、协议不匹配或状态恢复失败时，自动重置模型、优化器、调度器、随机状态和最佳指标，从头训练并创建新 W&B Run。`--resume /path/to/last.pth` 则严格恢复指定检查点，加载或恢复失败直接报错。成功恢复会继续原 W&B Run。旧版曾留出验证集的消融 checkpoint 不符合当前协议，可不传 --resume 重新训练并覆盖本地结果。

```text
model_trained/stage2_ablation/cls/constraint_aware/<experiment>/seed_<n>/
  config.json              实验与已有数据划分信息
  parameters.json          参数量
  epoch_0001.json ...       每轮 train/test 指标
  last.pth                 完整训练恢复状态
  best.pth                 test instance_accuracy 最优模型
  result.json              最优模型的测试结果
```

checkpoint 保存模型、优化器、调度器、随机状态和数据协议。顶层 summary.csv、summary.json 每行一个运行，保留原始指标，仅导出当前 seed 的分类结果，不跨种子聚合。

分类指标包括 OA、mAcc、macro F1、mAP、top-5、逐类指标和混淆矩阵。本入口与 train_cls.py 一样使用 W&B 记录训练数据，同时保留本地 JSON/CSV。

## 验证

```bash
python -m unittest discover -s tests -p test_cls_ablation.py -v
```

测试覆盖九项模型的反向传播和输入防泄漏、已有 train/test 目录、既有划分文件不被修改、无划分时拒绝创建、HDF5 划分保留、训练/测试评估、断点恢复及单种子导出。

## 与 train_cls.py 对齐的训练与 W&B

分类损失和约束组装直接复用 train_cls.py 的 classification_loss 与 constraints_from_dataset_batch。优化器、StepLR、辅助损失、label smoothing、梯度裁剪、按 batch 平均的 loss 及 test OA 选优规则一致。仍严格使用已有 train/test，不自行划分。

支持相同参数名 --bs、--epoch、--lr、--decay_rate、--n_point，同时兼容旧消融入口的 --batch_size、--epochs、--learning_rate、--weight_decay。默认 batch=20。数据集路径统一使用 --data_root。支持 --is_sample，以及相同的 token/Transformer、stage2_norm、stream_dropout、use_stats_token 设置。--is_sample 仅用于抽样快速调试，不改变已有数据划分；正式实验默认读取全部数据。

训练强制通过与 train_cls.py 相同的 initialize_wandb_run 读取项目 .env 中的 WANDB_API_KEY 并创建在线 Run。每个实验独立创建 Run，默认项目 cstnet2-s2；可设置 --wandb_project、--wandb_entity、--wandb_run_name。Run 名称始终附加实验名和 seed，避免九项实验混在同一条曲线中。

记录字段包括 epoch、learning_rate、loss/train、loss/test、best/test_instance_accuracy、train/metric/*、test/metric/*、训练梯度范数、checkpoint 保存状态以及 train/test 混淆矩阵，命名与 train_cls.py 一致。W&B config 额外记录 experiment 和保留的 constraint_components。

last.pth 和 best.pth 保存 wandb_run_id；--resume 成功恢复时使用该 ID 恢复原 Run，并从恢复的 epoch 继续记录。训练结束或异常退出都会 finish 当前 Run。evaluate、summarize、list、dry_run 不创建 W&B Run。

```bash
python train_cls_ablation.py --data_root /data/param20k_pred --experiments all --seed 42 --bs 20 --epoch 200 --lr 0.0001 --wandb_project cstnet2-s2
```

每个实验仍使用独立 checkpoint 目录及单个可设置的 seed；原 train_cls.py 无需修改。W&B 验证测试使用 mock，不会将合成测试数据上传到真实项目。

本地检查点、JSON 日志或汇总写入失败时，打印提示并跳过该次输出，继续训练；后续输出仍会尝试保存。W&B 的 checkpoint/last_saved 和 checkpoint/best_saved 记录本轮实际保存状态（未尝试或失败为 0）。最佳轮次的测试指标保留在内存并写入后续检查点，训练结束不再依赖重新读取 best.pth。写盘失败期间的模型可能无法恢复；指定路径的 resume 和 evaluate 仍要求能够读取相应检查点；resume auto 读取失败则从头训练。
