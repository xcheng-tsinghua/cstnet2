"""
训练约束预测模块
"""
import os
import argparse

from data_utils.datasets import CstNet2Dataset
from functional.cst_pred_trainer import CstPredTrainer
from networks.cst_pred_wrapper import CstPredWrapper
from colorama import init, Fore, Back


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch', default=2000, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate in training')
    parser.add_argument('--n_points', type=int, default=2000, help='Point Number')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--is_load_weight', default='True', choices=['True', 'False'], type=str)
    parser.add_argument('--model', default='pointnet2', choices=['pointnet2', 'pointnet', 'attn_3dgcn'], type=str)
    parser.add_argument('--is_sample', default='False', choices=['True', 'False'], type=str)

    parser.add_argument('--local', default='False', choices=['True', 'False'], type=str)
    parser.add_argument('--root_sever', type=str, default=rf'/opt/data/private/data_set/pcd_cstnet2/Param20K_Extend')
    parser.add_argument('--root_local', type=str, default=rf'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend')

    args = parser.parse_args()
    return args


def main(args):
    save_str = f'{args.model}_pmt_prim_cluster'
    print(Fore.BLUE + Back.CYAN + f'-> save str: {save_str} <-')

    os.makedirs('log', exist_ok=True)
    os.makedirs('model_trained', exist_ok=True)

    # data
    data_root = args.root_local if eval(args.local) else args.root_sever
    train_loader, test_loader = CstNet2Dataset.create_dataloader(
        root=data_root,
        bs=args.bs,
        n_points=args.n_points,
        num_workers=args.workers,
        is_sample=eval(args.workers.is_sample)
    )

    # trainer
    trainer = CstPredTrainer(
        model = CstPredWrapper(args.model).cuda(),
        train_loader = train_loader,
        test_loader = test_loader,
        model_savepth = 'model_trained/' + save_str + '.pth',
        log_savepth = os.path.join('log', save_str + '.json'),
        max_epoch = args.epoch,
        lr = args.lr,
        is_load_weight = eval(args.is_load_weight)
    )
    trainer.start()



#
#     if eval(args.is_load_weight):
#         try:
#             predictor.load_state_dict(torch.load(model_savepth))
#             print(Fore.WHITE + Back.CYAN + 'training from exist model: ' + model_savepth)
#         except:
#             print(Fore.RED + Back.CYAN + 'no existing model, training from scratch')
#     else:
#         print(Fore.BLACK + Back.CYAN + 'does not load weight, training from scratch')
#
#     # optimizer
#     optimizer = torch.optim.Adam(
#         predictor.parameters(),
#         lr=args.lr,
#         betas=(0.9, 0.999),
#         eps=1e-08,
#         weight_decay=args.decay_rate
#     )
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
#
#     # 训练
#     train_batch = 0
#     test_batch = 0
#     for epoch in range(args.epoch):
#         train_loss_list_pmt = []
#         train_loss_list_tri = []
#
#         train_acc = []
#         train_nmi = []
#         train_ari = []
#         train_pmt_acc = []
#
#         # 设置为训练模式，启用 dropout、batchNormalization 等模块
#         predictor = predictor.train()
#
#         progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
#         for batch_id, data in progress_bar:
#             xyz, pmt_gt, affiliate_idx = data[0].float().cuda(), data[2].long().cuda(), data[-1].long().cuda()
#
#             # 清空梯度，否则梯度会累加
#             optimizer.zero_grad()
#
#             # 将数据输入模型进行推理
#             pnt_fea, log_pmt = predictor(xyz)
#
#             # 计算损失
#             pmt_loss, tri_loss = compute_loss(pnt_fea, affiliate_idx, log_pmt, pmt_gt)
#
#             # 梯度反向传播
#             loss = pmt_loss + tri_loss
#             loss.backward()
#
#             # 优化器根据梯度进行权重更新
#             optimizer.step()
#
#             # 记录损失
#             train_loss_list_pmt.append(pmt_loss.item())
#             train_loss_list_tri.append(tri_loss.item())
#
#             # 计算准确率
#             acc, nmi, ari = evaluate_clustering(affiliate_idx, pnt_fea)
#             pmt_acc = compute_seg_acc(log_pmt, pmt_gt)
#
#             train_acc.append(acc.item())
#             train_nmi.append(nmi.item())
#             train_ari.append(ari.item())
#             train_pmt_acc.append(pmt_acc)
#
#             writer.add_scalar('train/loss_batch_pmt', pmt_loss.item(), train_batch)
#             writer.add_scalar('train/loss_batch_tri', tri_loss.item(), train_batch)
#
#             writer.add_scalar('train/batch_acc', acc.item(), train_batch)
#             writer.add_scalar('train/batch_nmi', nmi.item(), train_batch)
#             writer.add_scalar('train/batch_ari', ari.item(), train_batch)
#             writer.add_scalar('train/batch_pmt_acc', pmt_acc, train_batch)
#             train_batch += 1
#
#             # 更新进度条
#             progress_bar.set_postfix({
#                 'pmt_acc': f"{pmt_acc:.4f}",
#                 'cluster_acc': f"{acc:.4f}",
#                 'LR': f"{optimizer.param_groups[0]['lr']:.6f}"}
#             )
#
#         train_loss_epoch_pmt = np.mean(train_loss_list_pmt).item()
#         train_loss_epoch_tri = np.mean(train_loss_list_tri).item()
#
#         writer.add_scalar('train/loss_epoch_pmt', train_loss_epoch_pmt, epoch)
#         writer.add_scalar('train/loss_epoch_tri', train_loss_epoch_tri, epoch)
#
#         epoch_acc = np.mean(train_acc).item()
#         epoch_nmi = np.mean(train_nmi).item()
#         epoch_ari = np.mean(train_ari).item()
#         epoch_pmt_acc = np.mean(train_pmt_acc).item()
#
#         writer.add_scalar('train/epoch_acc', epoch_acc, epoch)
#         writer.add_scalar('train/epoch_nmi', epoch_nmi, epoch)
#         writer.add_scalar('train/epoch_ari', epoch_ari, epoch)
#         writer.add_scalar('train/epoch_pmt_acc', epoch_pmt_acc, epoch)
#
#         # 学习率调整器计数加一
#         scheduler.step()
#
#         # 保存权重
#         torch.save(predictor.state_dict(), model_savepth)
#
#         # 测试
#         with torch.no_grad():
#             test_loss_list_pmt = []
#             test_loss_list_tri = []
#
#             test_acc = []
#             test_nmi = []
#             test_ari = []
#             test_pmt_acc = []
#
#             # 设置为评估模式，禁用 dropout、batchNormalization 等模块
#             predictor = predictor.eval()
#
#             progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
#             for batch_id, data in progress_bar:
#                 xyz, pmt_gt, affiliate_idx = data[0].float().cuda(), data[2].long().cuda(), data[-1].long().cuda()
#
#                 pnt_fea, log_pmt = predictor(xyz)
#
#                 pmt_loss, tri_loss = compute_loss(pnt_fea, affiliate_idx, log_pmt, pmt_gt)
#
#                 test_loss_list_pmt.append(pmt_loss.item())
#                 test_loss_list_tri.append(tri_loss.item())
#
#                 # 计算准确率
#                 acc, nmi, ari = evaluate_clustering(affiliate_idx, pnt_fea)
#                 pmt_acc = compute_seg_acc(log_pmt, pmt_gt)
#
#                 test_acc.append(acc.item())
#                 test_nmi.append(nmi.item())
#                 test_ari.append(ari.item())
#                 test_pmt_acc.append(pmt_acc)
#
#                 writer.add_scalar('test/loss_batch_pmt', pmt_loss.item(), test_batch)
#                 writer.add_scalar('test/loss_batch_tri', tri_loss.item(), test_batch)
#
#                 writer.add_scalar('test/batch_acc', acc.item(), train_batch)
#                 writer.add_scalar('test/batch_nmi', nmi.item(), train_batch)
#                 writer.add_scalar('test/batch_ari', ari.item(), train_batch)
#                 writer.add_scalar('test/batch_pmt_acc', pmt_acc, train_batch)
#                 test_batch += 1
#
#                 # 更新进度条
#                 progress_bar.set_postfix({
#                     'pmt_acc': f"{pmt_acc:.4f}",
#                     'cluster_acc': f"{acc:.4f}",
#                     'LR': f"{optimizer.param_groups[0]['lr']:.6f}"}
#                 )
#
#             test_loss_epoch_pmt = np.mean(test_loss_list_pmt).item()
#             test_loss_epoch_tri = np.mean(test_loss_list_tri).item()
#
#             writer.add_scalar('test/loss_epoch_pmt', test_loss_epoch_pmt, epoch)
#             writer.add_scalar('test/loss_epoch_tri', test_loss_epoch_tri, epoch)
#
#             epoch_acc = np.mean(test_acc).item()
#             epoch_nmi = np.mean(test_nmi).item()
#             epoch_ari = np.mean(test_ari).item()
#             epoch_pmt_acc = np.mean(test_pmt_acc).item()
#
#             writer.add_scalar('test/epoch_acc', epoch_acc, epoch)
#             writer.add_scalar('test/epoch_nmi', epoch_nmi, epoch)
#             writer.add_scalar('test/epoch_ari', epoch_ari, epoch)
#             writer.add_scalar('test/epoch_pmt_acc', epoch_pmt_acc, epoch)
#
#             print(f'''{epoch} / {args.epoch}:
#                     train_loss_epoch_pmt: {train_loss_epoch_pmt:.6f},
#                     train_loss_epoch_tri: {train_loss_epoch_tri:.6f}.
#                     test_loss_epoch_pmt: {test_loss_epoch_pmt:.6f},
#                     test_loss_epoch_tri: {test_loss_epoch_tri:.6f},
#                     acc: {epoch_acc}
#                     nmi: {epoch_nmi}
#                     ari: {epoch_ari}
#                     pmt_acc: {epoch_pmt_acc}
# ''')
#     writer.close()


if __name__ == '__main__':
    init(autoreset=True)
    main(parse_args())

