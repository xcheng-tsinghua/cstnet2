from data_utils import vis
from cst_pred.cst_pcd import CstPcd
import os
import torch
from data_utils import cone_gen


def test_cst_pcd():
    test_tensor = torch.rand(16, 3, 2000).cuda()
    anet = CstPcd().cuda()

    log_pmt, pnt_fea = anet(test_tensor)
    print(log_pmt.size(), pnt_fea.size())


if __name__ == '__main__':
    test_cst_pcd()




    # rand_pnts, foot = utils.random_points_in_plane(1, 1, 1, -1, 1000)
    # vis.vis_3dpnts(rand_pnts)

    # c_file = r'D:\document\DeepLearning\DataSet\pcd_cstnet2\test\0.300912028194215;-0.03974005626696904;0.34182670070329935.txt'
    #
    # c_base = os.path.basename(c_file)
    # c_base = os.path.splitext(c_base)[0]
    #
    # c_x, c_y, c_z = c_base.split(';')
    #
    # c_x1, c_y1, c_z1 = float(c_x), float(c_y), float(c_z)
    #
    # print(c_x1, c_y1, c_z1)










