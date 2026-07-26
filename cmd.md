cd /opt/data/private/networks/cstnet2/ && conda activate dp

# 常用命令行

## github 同步

1. 查看当前状态

git status

2. 将修改添加到暂存区

git add .

3. 将更改信息暂存到本地

git commit -m "change"

4. 推送到远程仓库

git push origin main

5. 统一指令

git pull && git status && git add . && git commit -m "change" && git push

## 删除跟踪的文件

1. 对于目录

需要将 ${directory} 更换为已被Git同步但是需要解除同步的文件夹

git rm --cached -r ${directory}

2. 对于文件

需要将 ${file} 更换为已被Git同步但是需要解除同步的文件

git rm --cached ${file}

## 后台运行进程

1. nohup 末尾添加该命令可以指定log文件

> out.log 2>&1 &

例如：nohup python script.py > out.log 2>&1 &

2. 查看 nohup 的进程输出

tail -f nohup.out

3. 查看 nohup 的进程

ps -ef | grep python

## 创建新分支

1. 查看当前分支

git branch

2. 基于当前分支创建新分支

git branch v2026-07-26

3. 切换到新分支

git switch v2026-07-26

4. 将本地分支关联到远程分支，并推送新分支到远程仓库

git push -u origin v2026-07-26

5. 备份完成后切换回主分支

git switch main

## 将当前分支强制覆盖 main 分支

1. 假设用于覆盖 main 的分支叫 edge_tag
git fetch origin
git switch edge_tag

2. 可选：给旧 main 建一个备份分支
git branch backup-main origin/main
git push origin backup-main

3. 用 edge_tag 强制覆盖远程 main
git push origin edge_tag:main --force-with-lease

4. 切回 main，并同步覆盖后的内容
git switch main
git fetch origin
git reset --hard origin/main

## 将远程 main 分支备份

1. 更新远程分支信息
git fetch origin

2. 基于远程 main 创建本地备份分支
git branch valid_2026_7_12 origin/main

3. 将备份分支推送到 GitHub，并强制将本地分支关联到远程同名分支
git push -u origin valid_2026_7_17






nohup python train_cls.py --bs=20 --epoch=500 --model=constraint_aware --save_name=stage2_cls_gt > out_gt.log 2>&1 &

nohup python train_cls.py --bs=20 --epoch=500 > out.log 2>&1 &


nohup python train_cst_pred.py --bs=40 > out_s1.log 2>&1 &
tail -f out_s1.log

python train_cst_pred.py --epoch 1 --bs 2 --n_points 512 --train_phase semantic --use_extra_features --normal_source gt


------ 新 stage1 训练
1. Semantic 阶段
nohup python train_cst_pred.py --epoch 100 --bs 30 --model attn_3dgcn --train_phase semantic --use_extra_features --normal_source gt > out_s1s.log 2>&1 &

2. Geometry 阶段
从 semantic 最佳权重初始化，使用新 optimizer：
nohup python train_cst_pred.py --epoch 50 --bs 30 --model attn_3dgcn --train_phase geometry --use_extra_features --normal_source gt --geom_start_epoch 0 --geom_ramp_epochs 10 --init_from_checkpoint model_trained/attn_3dgcn_multitask_semantic_pmt_prim_cluster/best_pmt_miou.pth > out_s1g.log 2>&1 &

3. Joint 阶段
nohup python train_cst_pred.py --epoch 100 --bs 30 --model attn_3dgcn --train_phase joint --use_extra_features --normal_source gt --geom_start_epoch 0 --geom_ramp_epochs 10 --joint_backbone_lr_scale 0.1 --init_from_checkpoint model_trained/attn_3dgcn_multitask_geometry_pmt_prim_cluster/best_constraint_score.pth > out_s1j.log 2>&1 &


中断后完整续训
必须保持 phase、点数、特征设置、loss 权重和 ramp 配置一致：
python train_cst_pred.py --epoch 200 --bs 20 --model attn_3dgcn --train_phase joint --use_extra_features --normal_source gt --geom_start_epoch 0 --geom_ramp_epochs 10 --joint_backbone_lr_scale 0.1 --resume_checkpoint model_trained/attn_3dgcn_multitask_joint_pmt_prim_cluster/last.pth
其中 resume 的 --epoch 200 表示训练到 global epoch 200，不是额外训练 200 轮。

One-batch overfit
python train_cst_pred.py --epoch 20 --bs 20 --train_phase joint --overfit_one_batch --geom_start_epoch 0 --geom_ramp_epochs 5

-- stage2 seg
nohup python train_seg.py > out_s2seg.log 2>&1 &
tail -f out_s2seg.log


// pointnet++
nohup python train_seg.py --model pointnet2 --batch_size=100 --epochs=70 --not_resume > pointnet2.log 2>&1 &
tail -f pointnet2.log

nohup python train_seg.py --model pointnet --batch_size=100 && python train_seg.py --model dgcnn --batch_size=100 && python train_seg.py --model attn3dgcn --batch_size=100 &

训练无约束版本的baseline

nohup bash -c '
  python train_seg.py --model pointnet --batch_size=100 --epochs=70 --not_resume 2>&1 | tee pointnet.log
  python train_seg.py --model pointnet2 --batch_size=100 --epochs=70 --not_resume 2>&1 | tee pointnet2.log
  python train_seg.py --model dgcnn --batch_size=100 --epochs=70 --not_resume 2>&1 | tee dgcnn.log
  python train_seg.py --model attn3dgcn --batch_size=100 --epochs=70 --not_resume 2>&1 | tee attn3dgcn.log
' > /dev/null 2>&1 &

nohup bash -c '
  python train_seg.py --model pointtransformer --batch_size=100 --epochs=70 --not_resume 2>&1 | tee pointtransformer.log
  python train_seg.py --model pointmamba --batch_size=100 --epochs=70 --not_resume 2>&1 | tee pointmamba.log
  python train_seg.py --model pointnext --batch_size=100 --epochs=70 --not_resume 2>&1 | tee pointnext.log
  python train_seg.py --model pointmlp --batch_size=100 --epochs=70 --not_resume 2>&1 | tee pointmlp.log
' > /dev/null 2>&1 &

训练有约束版本的baseline

nohup bash -c '
  python train_seg.py --model pointnet --batch_size=100 --epochs=70 --not_resume --baseline_use_constraints 2>&1 | tee pointnet_cst_gt.log
  python train_seg.py --model pointnet2 --batch_size=100 --epochs=70 --not_resume --baseline_use_constraints 2>&1 | tee pointnet2_cst_gt.log
  python train_seg.py --model dgcnn --batch_size=100 --epochs=70 --not_resume --baseline_use_constraints 2>&1 | tee dgcnn_cst_gt.log
  python train_seg.py --model attn3dgcn --batch_size=100 --epochs=70 --not_resume --baseline_use_constraints 2>&1 | tee attn3dgcn_cst_gt.log
' > /dev/null 2>&1 &

nohup bash -c '
  python train_seg.py --model pointtransformer --batch_size=100 --epochs=70 --not_resume --baseline_use_constraints 2>&1 | tee pointtransformer_cst_gt.log
  python train_seg.py --model pointmamba --batch_size=100 --epochs=70 --not_resume --baseline_use_constraints 2>&1 | tee pointmamba_cst_gt.log
  python train_seg.py --model pointnext --batch_size=100 --epochs=70 --not_resume --baseline_use_constraints 2>&1 | tee pointnext_cst_gt.log
  python train_seg.py --model pointmlp --batch_size=100 --epochs=70 --not_resume --baseline_use_constraints 2>&1 | tee pointmlp_cst_gt.log
' > /dev/null 2>&1 &

