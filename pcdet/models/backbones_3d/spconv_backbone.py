from functools import partial

import torch.nn as nn
import torch

from ...utils.spconv_utils import replace_feature, spconv
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

class VoxelBackBone8x_explicit(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)#partial函数(其实是个class)从原函数中派生出固定某些参数的新函数,使函数所需的参数减少

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] -> [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] -> [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

        # pointnet++ ############################################################
        self.point_cls = nn.Linear(128+5, 1, bias=False)

        # self.point_cls1 = nn.Linear(128+5, 64, bias=False)
        # self.point_cls2 = nn.Linear(64, 32, bias=False)
        # self.point_cls3 = nn.Linear(32, 1, bias=False)

        # self.SA_modules ############################################################
        input_channels = 5
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        # self.FP_modules ##############################################################
        self.FP_modules = nn.ModuleList()
        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )
        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

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
    
    def break_up_voxel_features(self, pc):
        xyz = pc[:, :3].view(1, -1 ,3).contiguous()
        features = (pc[:, 3:].contiguous() if pc.size(-1) > 3 else None)
        return xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        # print("\n*****VoxelBackBone8x")
        # voxel_features, voxel_coords  shape (Batch * 16000, 4)
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        # training: our method ###########################################################################
        if int(batch_dict["mode"][0])==1:
            # 获取一个batch中各个data的数量
            batch_indices1 = voxel_coords[:, 0].int()
            # 使用bincount()函数计算每个batch中的行数
            row_counts1 = torch.bincount(batch_indices1)

            if 'match_points' in batch_dict: # 如果有匹配点
                match_points = batch_dict['match_points']
                # 获取一个batch中各个data的数量
                batch_indices_match = match_points[:, 0].int()
                # 使用bincount()函数计算每个batch中的行数
                row_counts_match = torch.bincount(batch_indices_match)
                start_idx_match = 0

            start_idx = 0

            # 遍历batch中的第idx个data
            for idx in range(row_counts1.shape[0]):
                points_mean, features = self.break_up_voxel_features(voxel_features[start_idx : start_idx+row_counts1[idx], :])
                res_feature = voxel_features[start_idx : start_idx+row_counts1[idx], :].view(1, -1, 5).contiguous()
                # pointnet++丰富特征表达
                xyz = points_mean.view(1, -1, 3) # [1, N, 3]=[1, 3446, 3]
                features = features.view(1, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None

                l_xyz, l_features = [xyz], [features]
                for i in range(len(self.SA_modules)):
                    li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
                    l_xyz.append(li_xyz)
                    l_features.append(li_features)

                for i in range(-1, -(len(self.FP_modules) + 1), -1):
                    l_features[i - 1] = self.FP_modules[i](
                        l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
                    )  # (B, C, N)
                point_features = l_features[0].permute(0, 2, 1).contiguous()  # (B, N, C)

                # 分类和回归
                # 只有pointnet2生成的特征
                # point_cls_1 = self.point_cls(point_features)#torch.Size([1, N1, 1])
                # pointnet2生成的特征和resnet生成的特征
                point_cls_1 = self.point_cls(torch.cat((point_features, res_feature), dim=-1))#torch.Size([1, 128+5, 1])
                # # pointnet2生成的特征和resnet生成的特征-过三层mlp
                # point_cls_1 = self.point_cls1(torch.cat((point_features, res_feature), dim=-1))#torch.Size([1, 128+5, 1])
                # point_cls_1 = self.point_cls2(point_cls_1)#torch.Size([1, 128+5, 1])
                # point_cls_1 = self.point_cls3(point_cls_1)#torch.Size([1, 128+5, 1])

                if idx==0:
                    pred_hm_list = point_cls_1
                else:
                    pred_hm_list = torch.cat([pred_hm_list, point_cls_1], dim=1) #torch.Size([1, N+N1, 1])
                
                start_idx += row_counts1[idx]

                # 获取gt_hm
                # denoise ##############################################################
                if self.training:
                    match_points_batch = match_points[start_idx_match:start_idx_match+row_counts_match[idx], 1:4]
                    points_hm_1 = self.generate_valid_points_mask(points_mean, match_points_batch)
                    points_hm_1 = points_hm_1.float()
                    if idx==0:
                        gt_hm_list = points_hm_1
                    else:
                        gt_hm_list = torch.cat([gt_hm_list, points_hm_1], dim=1)

                    start_idx_match += row_counts_match[idx]

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
            if self.training:
                batch_dict['points_hm'] = gt_hm_list

        # train & test 筛选voxel ##############################################################
        # softmax_pred = F.softmax(pred_hm_list, dim=1)
        # positive_mask = softmax_pred > 0.5
        positive_mask = pred_hm_list > 0.5
        valid_voxel_features = voxel_features[positive_mask.squeeze()]
        valid_voxel_coords = voxel_coords[positive_mask.squeeze()]
        batch_dict['voxel_features'], batch_dict['voxel_coords'] = valid_voxel_features, valid_voxel_coords
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']


        ##########################################################################################################
        # 根据voxel坐标，并将每个voxel放置voxel_coor对应的位置，建立成稀疏tensor
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            # [41,1600,1408] ZYX 每个voxel的长宽高为0.05，0.05，0.1 点云的范围为[0, -40, -3, 70.4, 40, 1]
            spatial_shape=self.sparse_shape,
            # 4
            batch_size=batch_size
        )
        """
        稀疏卷积的计算中，feature，channel，shape，index这几个内容都是分开存放的，
        在后面用out.dense才把这三个内容组合到一起了，变为密集型的张量
        spconv卷积的输入也是一样，输入和输出更像是一个  字典或者说元组
        注意卷积中pad与no_pad的区别
        """

        # # 进行submanifold convolution
        # [batch_size, 4, [41, 1600, 1408]] --> [batch_size, 16, [41, 1600, 1408]]
        x = self.conv_input(input_sp_tensor)
        # [batch_size, 16, [41, 1600, 1408]] --> [batch_size, 16, [41, 1600, 1408]]
        #print(x)
        x_conv1 = self.conv1(x)
        # [batch_size, 16, [41, 1600, 1408]] --> [batch_size, 32, [21, 800, 704]]
        #print(x_conv1)
        x_conv2 = self.conv2(x_conv1)
        # [batch_size, 32, [21, 800, 704]] --> [batch_size, 64, [11, 400, 352]]
        #print(x_conv2)
        x_conv3 = self.conv3(x_conv2)
        # [batch_size, 64, [11, 400, 352]] --> [batch_size, 64, [5, 200, 176]]
        #print(x_conv3)
        x_conv4 = self.conv4(x_conv3)
        #print(x_conv4)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        # [batch_size, 64, [5, 200, 176]] --> [batch_size, 128, [2, 200, 176]]
        out = self.conv_out(x_conv4)
        #print(out)
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)#partial函数(其实是个class)从原函数中派生出固定某些参数的新函数,使函数所需的参数减少

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] -> [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] -> [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        # print("\n*****VoxelBackBone8x")
        # voxel_features, voxel_coords  shape (Batch * 16000, 4)
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        # 根据voxel坐标，并将每个voxel放置voxel_coor对应的位置，建立成稀疏tensor
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            # [41,1600,1408] ZYX 每个voxel的长宽高为0.05，0.05，0.1 点云的范围为[0, -40, -3, 70.4, 40, 1]
            spatial_shape=self.sparse_shape,
            # 4
            batch_size=batch_size
        )
        """
        稀疏卷积的计算中，feature，channel，shape，index这几个内容都是分开存放的，
        在后面用out.dense才把这三个内容组合到一起了，变为密集型的张量
        spconv卷积的输入也是一样，输入和输出更像是一个  字典或者说元组
        注意卷积中pad与no_pad的区别
        """

        # # 进行submanifold convolution
        # [batch_size, 4, [41, 1600, 1408]] --> [batch_size, 16, [41, 1600, 1408]]
        x = self.conv_input(input_sp_tensor)
        # [batch_size, 16, [41, 1600, 1408]] --> [batch_size, 16, [41, 1600, 1408]]
        #print(x)
        x_conv1 = self.conv1(x)
        # [batch_size, 16, [41, 1600, 1408]] --> [batch_size, 32, [21, 800, 704]]
        #print(x_conv1)
        x_conv2 = self.conv2(x_conv1)
        # [batch_size, 32, [21, 800, 704]] --> [batch_size, 64, [11, 400, 352]]
        #print(x_conv2)
        x_conv3 = self.conv3(x_conv2)
        # [batch_size, 64, [11, 400, 352]] --> [batch_size, 64, [5, 200, 176]]
        #print(x_conv3)
        x_conv4 = self.conv4(x_conv3)
        #print(x_conv4)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        # [batch_size, 64, [5, 200, 176]] --> [batch_size, 128, [2, 200, 176]]
        out = self.conv_out(x_conv4)
        #print(out)
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]#[::-1]取从后向前（相反）的元素

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict

class VoxelImageFusionBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.fusion_layer = model_cfg.FUSION_LAYER
        self.fusion_method = model_cfg.get("FUSION_METHOD", "one_layer")

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

    def fusion(self, fuse_func, batch_dict, voxel_feat, layer_name):
        if self.fusion_method in ["one_layer"]:
            voxel_feat, batch_dict = fuse_func(batch_dict, encoded_voxel=voxel_feat,
                                               layer_name=layer_name)
        elif self.fusion_method in ["layer_by_layer"]:
            img_name = layer_name if "layer0" not in layer_name else "layer1"
            image_feat = batch_dict[img_name+"_feat2d"]
            voxel_feat, batch_dict = fuse_func(batch_dict, encoded_voxel=voxel_feat, 
                                               encoded_feat2d=image_feat,
                                               layer_name=layer_name)
        else:
            raise NotImplementedError
        
        return voxel_feat, batch_dict

    def forward(self, batch_dict, fuse_func=None):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        #print(voxel_coords)
        if voxel_features.shape[0] == 0:
            voxel_features = torch.randn(2,4)
            voxel_coords = [[0,27,331,1199],[0,24,331,1199]]
        elif voxel_features.shape[0] == 1:
            voxel_features = torch,cat([voxel_features,torch.randn(1,4)],dim=0)
            voxel_coords = torch,cat([voxel_coords,[0,24,331,1199]],dim=0)
        #print(voxel_features)
        #print("1111111111111111",voxel_features.shape)
        #print("2222222222222222",voxel_coords.shape)
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)
        if "x_conv0" in self.fusion_layer:
            x, batch_dict = self.fusion(fuse_func, batch_dict, x, "layer0")
        x_conv1 = self.conv1(x)
        if "x_conv1" in self.fusion_layer:
            x_conv1, batch_dict = self.fusion(fuse_func, batch_dict, x_conv1, "layer1")
        x_conv2 = self.conv2(x_conv1)
        if "x_conv2" in self.fusion_layer:
            x_conv2, batch_dict = self.fusion(fuse_func, batch_dict, x_conv2, "layer2")
        x_conv3 = self.conv3(x_conv2)
        if "x_conv3" in self.fusion_layer:
            x_conv3, batch_dict = self.fusion(fuse_func, batch_dict, x_conv3, "layer3")
        x_conv4 = self.conv4(x_conv3)
        if "x_conv4" in self.fusion_layer:
            x_conv4, batch_dict = self.fusion(fuse_func, batch_dict, x_conv4, "layer4")

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict