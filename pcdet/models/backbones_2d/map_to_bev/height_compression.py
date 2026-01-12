import torch.nn as nn
import torch
from ....ops.pointnet2.pointnet2_batch import pointnet2_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        # 高度的特征数
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_fc = nn.Linear(96, 64, bias=False)
        self.point_cls = nn.Linear(64, 1, bias=False)
        self.point_reg = nn.Linear(64, 3, bias=False)

    def tensor2points(self, tensor, offset=(0., -40., -3.), voxel_size=(.05, .05, .1)):
        indices = tensor.indices.float()
        offset = torch.Tensor(offset).to(indices.device)
        voxel_size = torch.Tensor(voxel_size).to(indices.device)
        indices[:, 1:] = indices[:, [3, 2, 1]] * voxel_size + offset + .5 * voxel_size
        return tensor.features, indices

    def generate_valid_points_mask(self, target_points, match_points_1):
        '''
            按阈值进行距离筛选, 超参数为0.5-->小于0.5m的点被选中
        '''
        target_points = target_points[0]
        # 计算A和B中每个点对间的距离
        distances = torch.cdist(target_points, match_points_1)
        # 找到每个点最小距离和索引
        min_distances, indices = distances.min(dim=1)
        # 构建匹配mask
        mask = min_distances < 0.5
        return mask.unsqueeze(0)

    def gauss_fun(self, points_mean, gt_boxes):

        
        gt_center = gt_boxes[:3]
        w_gt = gt_boxes[3]
        l_gt = gt_boxes[4]
        h_gt = gt_boxes[5]
        offset_gt = (points_mean - gt_center).view(-1,3)
        _COVARIANCE_1 = 4/(w_gt ** 2 + l_gt ** 2)
        _COVARIANCE_2 = 4/(w_gt ** 2 + h_gt ** 2)
        _COVARIANCE_3 = 4/(h_gt ** 2 + l_gt ** 2)

        _COVARIANCE = (torch.tensor([[_COVARIANCE_1, 0., 0.],
                    [0., _COVARIANCE_2, 0.],
                    [0., 0., _COVARIANCE_3]])).cuda()
        value_matric = torch.mm(torch.mm(offset_gt, _COVARIANCE), offset_gt.t())
        diag_value = torch.diag(value_matric)
        gt_hm = torch.exp(-0.5 * diag_value)

        return gt_hm

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        # training: our method ###########################################################################
        if self.training and (int(batch_dict["mode"][0])==1):
            voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
            # 获取一个batch中各个data的数量
            batch_indices1 = voxel_coords[:, 0].int()
            # 使用bincount()函数计算每个batch中的行数
            row_counts1 = torch.bincount(batch_indices1)

            match_points = batch_dict['match_points']
            # 获取一个batch中各个data的数量
            batch_indices_match = match_points[:, 0].int()
            # 使用bincount()函数计算每个batch中的行数
            row_counts_match = torch.bincount(batch_indices_match)
            assert row_counts1.shape[0] == row_counts_match.shape[0] #batch_size一致
            # print("row_counts_match:{}".format(row_counts_match))

            vx_feat_conv2, vx_nxyz_conv2 = self.tensor2points(batch_dict['multi_scale_3d_features']['x_conv2'], (0, -40., -3.), voxel_size=(.1, .1, .2))
            batch_indices_conv2 = vx_nxyz_conv2[:, 0].int()
            row_counts_conv2 = torch.bincount(batch_indices_conv2)
            # print("row_counts_conv2:{}".format(row_counts_conv2))
            assert row_counts1.shape[0] == row_counts_conv2.shape[0] #batch_size一致

            vx_feat_conv3, vx_nxyz_conv3 = self.tensor2points(batch_dict['multi_scale_3d_features']['x_conv3'], (0, -40., -3.), voxel_size=(.2, .2, .4))
            batch_indices_conv3 = vx_nxyz_conv3[:, 0].int()
            row_counts_conv3 = torch.bincount(batch_indices_conv3)
            # print("row_counts_conv3:{}".format(row_counts_conv3))
            assert row_counts1.shape[0] == row_counts_conv3.shape[0] #batch_size一致

            vx_feat_conv4, vx_nxyz_conv4 = self.tensor2points(batch_dict['multi_scale_3d_features']['x_conv4'], (0, -40., -3.), voxel_size=(.4, .4, .8))
            batch_indices_conv4 = vx_nxyz_conv4[:, 0].int()
            row_counts_conv4 = torch.bincount(batch_indices_conv4)
            # print("row_counts_conv4:{}".format(row_counts_conv4))
            assert row_counts1.shape[0] == row_counts_conv4.shape[0] #batch_size一致
            

            start_idx = 0
            start_idx_match = 0
            start_idx_conv2 = 0
            start_idx_conv3 = 0
            start_idx_conv4 = 0
            # pred_hm_list = torch.empty((0,))
            # gt_hm_list = torch.empty((0,))
            # 遍历batch中的第idx个data
            for idx in range(row_counts1.shape[0]):
                points_mean = voxel_features[start_idx : start_idx+row_counts1[idx], :][:, :3].view(1, -1 ,3).contiguous()

                # conv2输出的特征
                vx_nxyz_conv = vx_nxyz_conv2[start_idx_conv2 : start_idx_conv2+row_counts_conv2[idx], :][:, 1:].view(1, -1 ,3).contiguous()
                vx_feat_conv = vx_feat_conv2[start_idx_conv2 : start_idx_conv2+row_counts_conv2[idx], :].view(1, 32, -1).contiguous()
                  #voxel聚合点的特征
                p0 = nearest_neighbor_interpolate(points_mean, vx_nxyz_conv, vx_feat_conv).view(1, -1, 32)
                # conv3输出的特征
                vx_nxyz_conv = vx_nxyz_conv3[start_idx_conv3 : start_idx_conv3+row_counts_conv3[idx], :][:, 1:].view(1, -1 ,3).contiguous()
                vx_feat_conv = vx_feat_conv3[start_idx_conv3 : start_idx_conv3+row_counts_conv3[idx], :].view(1, 32, -1).contiguous()
                  #voxel聚合点的特征
                p1 = nearest_neighbor_interpolate(points_mean, vx_nxyz_conv, vx_feat_conv).view(1, -1, 32)
                # # conv4输出的特征
                vx_nxyz_conv = vx_nxyz_conv4[start_idx_conv4 : start_idx_conv4+row_counts_conv4[idx], :][:, 1:].view(1, -1 ,3).contiguous()
                vx_feat_conv = vx_feat_conv4[start_idx_conv4 : start_idx_conv4+row_counts_conv4[idx], :].view(1, 32, -1).contiguous()
                 #voxel聚合点的特征
                p2 = nearest_neighbor_interpolate(points_mean, vx_nxyz_conv, vx_feat_conv).view(1, -1, 32)

                # 三个特征拼接
                pointwise_1 = self.point_fc(torch.cat([p0, p1, p2], dim=-1))
                # 分类和回归
                point_cls_1 = self.point_cls(pointwise_1) #torch.Size([1, N1, 1])
                # print("point_cls_1:{}".format(point_cls_1.shape))
                if idx==0:
                    pred_hm_list = point_cls_1
                else:
                    pred_hm_list = torch.cat([pred_hm_list, point_cls_1], dim=1) #torch.Size([1, N+N1, 1])

                # 获取gt_hm
                # denoise ##############################################################
                match_points_batch = match_points[start_idx_match:start_idx_match+row_counts_match[idx], 1:4]
                points_hm_1 = self.generate_valid_points_mask(points_mean, match_points_batch)
                points_hm_1 = points_hm_1.float()
                if idx==0:
                    gt_hm_list = points_hm_1
                else:
                    gt_hm_list = torch.cat([gt_hm_list, points_hm_1], dim=1)

                start_idx += row_counts1[idx]
                start_idx_match += row_counts_match[idx]
                start_idx_conv2 += row_counts_conv2[idx]
                start_idx_conv3 += row_counts_conv3[idx]
                start_idx_conv4 += row_counts_conv4[idx]
                # debug输出
                if int(batch_dict["debug_mode"][0]):
                    import os
                    import numpy as np
                    gt_boxes = batch_dict['gt_boxes'][idx,:,:][:,:7].view(1,-1,7)
                    current_working_directory = os.getcwd()
                    parent_dir = os.path.dirname(current_working_directory)
                    np.save(parent_dir + '/output_debug/final_points_mean.npy', points_mean[0].cpu().numpy())
                    np.save(parent_dir + '/output_debug/final_gt_boxes.npy', gt_boxes[0].cpu().numpy())
                    np.save(parent_dir + '/output_debug/final_match_points.npy', match_points_batch.cpu().numpy())
                    np.save(parent_dir + '/output_debug/final_mask.npy', points_hm_1.cpu().numpy())
            batch_dict['pred_hm'] = pred_hm_list
            batch_dict['points_hm'] = gt_hm_list
            # # exit()
        # training: hanet method ###########################################################################
        elif self.training and (int(batch_dict["mode"][0])==2):
            voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
            # 获取一个batch中各个data的数量
            batch_indices1 = voxel_coords[:, 0].int()
            # 使用bincount()函数计算每个batch中的行数
            row_counts1 = torch.bincount(batch_indices1)

            # match_points = batch_dict['match_points']
            # # 获取一个batch中各个data的数量
            # batch_indices_match = match_points[:, 0].int()
            # # 使用bincount()函数计算每个batch中的行数
            # row_counts_match = torch.bincount(batch_indices_match)
            # assert row_counts1.shape[0] == row_counts_match.shape[0] #batch_size一致
            # # print("row_counts_match:{}".format(row_counts_match))

            vx_feat_conv2, vx_nxyz_conv2 = self.tensor2points(batch_dict['multi_scale_3d_features']['x_conv2'], (0, -40., -3.), voxel_size=(.1, .1, .2))
            batch_indices_conv2 = vx_nxyz_conv2[:, 0].int()
            row_counts_conv2 = torch.bincount(batch_indices_conv2)
            # print("row_counts_conv2:{}".format(row_counts_conv2))
            assert row_counts1.shape[0] == row_counts_conv2.shape[0] #batch_size一致

            vx_feat_conv3, vx_nxyz_conv3 = self.tensor2points(batch_dict['multi_scale_3d_features']['x_conv3'], (0, -40., -3.), voxel_size=(.2, .2, .4))
            batch_indices_conv3 = vx_nxyz_conv3[:, 0].int()
            row_counts_conv3 = torch.bincount(batch_indices_conv3)
            # print("row_counts_conv3:{}".format(row_counts_conv3))
            assert row_counts1.shape[0] == row_counts_conv3.shape[0] #batch_size一致

            vx_feat_conv4, vx_nxyz_conv4 = self.tensor2points(batch_dict['multi_scale_3d_features']['x_conv4'], (0, -40., -3.), voxel_size=(.4, .4, .8))
            batch_indices_conv4 = vx_nxyz_conv4[:, 0].int()
            row_counts_conv4 = torch.bincount(batch_indices_conv4)
            # print("row_counts_conv4:{}".format(row_counts_conv4))
            assert row_counts1.shape[0] == row_counts_conv4.shape[0] #batch_size一致
            
            start_idx = 0
            # start_idx_match = 0
            start_idx_conv2 = 0
            start_idx_conv3 = 0
            start_idx_conv4 = 0
            # 遍历batch中的第idx个data
            for idx in range(row_counts1.shape[0]):
                points_mean = voxel_features[start_idx : start_idx+row_counts1[idx], :][:, :3].view(1, -1 ,3).contiguous()

                # conv2输出的特征
                vx_nxyz_conv = vx_nxyz_conv2[start_idx_conv2 : start_idx_conv2+row_counts_conv2[idx], :][:, 1:].view(1, -1 ,3).contiguous()
                vx_feat_conv = vx_feat_conv2[start_idx_conv2 : start_idx_conv2+row_counts_conv2[idx], :].view(1, 32, -1).contiguous()
                  #voxel聚合点的特征
                p0 = nearest_neighbor_interpolate(points_mean, vx_nxyz_conv, vx_feat_conv).view(1, -1, 32)
                # conv3输出的特征
                vx_nxyz_conv = vx_nxyz_conv3[start_idx_conv3 : start_idx_conv3+row_counts_conv3[idx], :][:, 1:].view(1, -1 ,3).contiguous()
                vx_feat_conv = vx_feat_conv3[start_idx_conv3 : start_idx_conv3+row_counts_conv3[idx], :].view(1, 32, -1).contiguous()
                  #voxel聚合点的特征
                p1 = nearest_neighbor_interpolate(points_mean, vx_nxyz_conv, vx_feat_conv).view(1, -1, 32)
                # # conv4输出的特征
                vx_nxyz_conv = vx_nxyz_conv4[start_idx_conv4 : start_idx_conv4+row_counts_conv4[idx], :][:, 1:].view(1, -1 ,3).contiguous()
                vx_feat_conv = vx_feat_conv4[start_idx_conv4 : start_idx_conv4+row_counts_conv4[idx], :].view(1, 32, -1).contiguous()
                 #voxel聚合点的特征
                p2 = nearest_neighbor_interpolate(points_mean, vx_nxyz_conv, vx_feat_conv).view(1, -1, 32)

                # 三个特征拼接
                pointwise_1 = self.point_fc(torch.cat([p0, p1, p2], dim=-1))
                # 分类和回归
                point_cls_1 = self.point_cls(pointwise_1) #torch.Size([1, N1, 1])
                # print("point_cls_1:{}".format(point_cls_1.shape))
                if idx==0:
                    pred_hm_list = point_cls_1
                else:
                    pred_hm_list = torch.cat([pred_hm_list, point_cls_1], dim=1) #torch.Size([1, N+N1, 1])

                # 获取gt_hm
                # hanet ################
                gt_boxes = batch_dict['gt_boxes'][idx,:,:][:,:7].view(1,-1,7)
                point_indices = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_mean, gt_boxes
                ) # (nboxes, npoints) point_indices: the point belone which box
                points_hm = torch.zeros_like(point_indices, dtype=torch.float)
                num_box = gt_boxes.size(1)
                point_indices = point_indices.reshape(-1)
                for i in range(num_box):
                    mask = point_indices == i
                    points_hm[:, mask] = self.gauss_fun(points_mean[0,mask,:], gt_boxes[0,i,:])
                # targets_dict['gt_hm'] = points_hm
                if idx==0:
                    gt_hm_list = points_hm
                else:
                    gt_hm_list = torch.cat([gt_hm_list, points_hm], dim=1)

                start_idx += row_counts1[idx]
                # start_idx_match += row_counts_match[idx]
                start_idx_conv2 += row_counts_conv2[idx]
                start_idx_conv3 += row_counts_conv3[idx]
                start_idx_conv4 += row_counts_conv4[idx]
                # debug输出
                if int(batch_dict["debug_mode"][0]):
                    import os
                    import numpy as np
                    gt_boxes = batch_dict['gt_boxes'][idx,:,:][:,:7].view(1,-1,7)
                    current_working_directory = os.getcwd()
                    parent_dir = os.path.dirname(current_working_directory)
                    np.save(parent_dir + '/output_debug/final_points_mean.npy', points_mean[0].cpu().numpy())
                    np.save(parent_dir + '/output_debug/final_gt_boxes.npy', gt_boxes[0].cpu().numpy())
                    np.save(parent_dir + '/output_debug/final_mask.npy', points_hm_1.cpu().numpy())
            batch_dict['pred_hm'] = pred_hm_list
            batch_dict['points_hm'] = gt_hm_list
            # # exit()

        # training: baseline method ##############################################
        elif self.training:
            batch_dict['pred_hm'] = torch.zeros(1, 1)
            batch_dict['points_hm'] = torch.zeros(1, 1)
            
        # 得到VoxelBackBone8x的输出特征
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        # 将稀疏的tensor转化为密集tensor,[bacth_size, 128, 2, 200, 176]
        # 结合batch，spatial_shape、indice和feature将特征还原到密集tensor中对应位置
        spatial_features = encoded_spconv_tensor.dense()
        # batch_size，128，2，200，176
        N, C, D, H, W = spatial_features.shape
        #print("1111111111111111111111111",spatial_features.shape)
        """
        将密集的3D tensor reshape为2D鸟瞰图特征    
        将两个深度方向内的voxel特征拼接成一个 shape : (batch_size, 256, 200, 176)
        z轴方向上没有物体会堆叠在一起，这样做可以增大Z轴的感受野，
        同时加快网络的速度，减小后期检测头的设计难度
        """
        spatial_features = spatial_features.view(N, C * D, H, W)
        spatial_features = spatial_features[:, :256, :, :]
        # 将特征和采样尺度加入batch_dict
        # print("1111111111111111111111111",spatial_features.shape)
        batch_dict['spatial_features'] = spatial_features
        # 特征图的下采样倍数 8倍
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        
        return batch_dict

def nearest_neighbor_interpolate(unknown, known, known_feats):
    """
    :param pts: (n, 4) tensor of the bxyz positions of the unknown features
    :param ctr: (m, 4) tensor of the bxyz positions of the known features
    :param ctr_feats: (m, C) tensor of features to be propigated
    :return:
        new_features: (n, C) tensor of the features of the unknown features
    """
    dist, idx = pointnet2_utils.three_nn(unknown, known)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

    return interpolated_feats