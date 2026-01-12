import os
os.environ['CUDA_VISIBLE_DEVICE']='1'

import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack

class DataSegProcessor(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg[0]
        # self.SA_modules ############################################################
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

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 2, 1)

        # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # self.SA_modules = self.SA_modules.to(device)  
        # self.FP_modules = self.FP_modules.to(device)  
        # self.conv1 = self.conv1.to(device)  
        # self.bn1 = self.bn1.to(device)  
        # self.drop1 = self.drop1.to(device)  
        # self.conv2 = self.conv2.to(device)  

        
        self.SA_modules = self.SA_modules.cuda()
        self.FP_modules = self.FP_modules.cuda()
        self.conv1 = self.conv1.cuda()
        self.bn1 = self.bn1.cuda()
        self.drop1 = self.drop1.cuda()
        self.conv2 = self.conv2.cuda()
        
        # #打印模型的结构
        # print('###打印模型self.conv1的结构####')
        # print(self.conv1)
        # print('\n')
        # print('###打印模型self.conv1加载参数前的初始值####')
        # print(list(self.conv1.parameters()))
        # print('\n')
        #############################################
        #2.加载部分预训练数据
        pretrained_dict = torch.load('CasA_0/output/dual_radar_models/CasA-V_arbe_seg/default-pointnet2_seg_0522/ckpt/checkpoint_epoch_80.pth')
        pretrained_dict = pretrained_dict['model_state']

        # for k,v in pretrained_dict.items():
        #     print(k)
        
        #自己的模型参数变量
        for module_name, module in [('SA_modules',self.SA_modules), ('FP_modules', self.FP_modules), ('conv1', self.conv1), ('bn1',self.bn1), ('drop1',self.drop1), ('conv2',self.conv2)]:

            model_dict = module.state_dict()
            
            for k, v in pretrained_dict.items():
                k_part = k.split(".")
                # name_1 = k_part[1]
                # name_2 = ".".join(k_part[2:])
                if len(k_part) < 2:
                    continue
                if (k_part[0] in 'backbone_3d') and (k_part[1] in module_name) and (".".join(k_part[2:]) in model_dict):
                    model_dict[".".join(k_part[2:])] = v
                else:
                    continue
            #参数更新
            model_dict.update(model_dict)
            # 加载我们真正需要的state_dict
            module.load_state_dict(model_dict)
            #3.冻结部分层
            #将满足条件的参数的 requires_grad 属性设置为False
            for name, value in module.named_parameters():
                value.requires_grad = False

        # # debug
        # print('###打印模型self.conv1加载参数后的参数值####')
        # print(list(self.conv1.parameters()))
        # print('\n')


    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def generate_valid_points_mask(self, target_points, match_points_1, th_distance=0.5):
        '''
            按阈值进行距离筛选, 超参数为0.5-->小于0.5m的点被选中
        '''
        target_points = target_points
        # 计算A和B中每个点对间的距离
        distances = torch.cdist(target_points, match_points_1)
        # 找到每个点最小距离和索引
        min_distances, indices = distances.min(dim=1)
        # 构建匹配mask
        mask_valid = min_distances < th_distance
        # mask_noise = ~mask_valid
        # mask_two = torch.stack([mask_valid.unsqueeze(0), mask_noise.unsqueeze(0)], dim=2)
        return mask_valid.unsqueeze(0)

    def forward(self, batch_dict):
        pass
        # """
        # Args:
        #     batch_dict:
        #         batch_size: int
        #         vfe_features: (num_voxels, C)
        #         points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        # Returns:
        #     batch_dict:
        #         encoded_spconv_tensor: sparse tensor
        #         point_features: (N, C)
        # """
        # batch_size = 1
        # pc = batch_dict['points']
        # pc = torch.from_numpy(pc)
        # pc = pc.cuda()
        # xyz = pc[:, 0:3].contiguous()
        # features = (pc[:, 3:].contiguous() if pc.size(-1) > 3 else None)

        # xyz = xyz.view(batch_size, -1, 3)
        # features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None
        # # features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1) if features is not None else None

        # l_xyz, l_features = [xyz], [features]
        # for i in range(len(self.SA_modules)):
        #     li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
        #     l_xyz.append(li_xyz)
        #     l_features.append(li_features)

        # for i in range(-1, -(len(self.FP_modules) + 1), -1):
        #     l_features[i - 1] = self.FP_modules[i](
        #         l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
        #     )  # (B, C, N)

        # point_features = l_features[0].permute(0, 2, 1).contiguous()  # (B, N, C)
        # # 预测结果#############################################
        # x = self.drop1(F.relu(self.bn1(self.conv1(l_features[0].contiguous())), inplace=True))
        # x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        # # 通过去噪网络滤除噪声######################################################
        # # 使用 torch.argmax 获取预测的标签
        # x_labels = torch.argmax(x[0], dim=1)
        # x_labels = x_labels > 0.5
        # # 更新 batch_dict['points']+batch_dict['arbe_num']
        # # 进行二次match
        # points_hm = self.generate_valid_points_mask(pc[:, 0:3], pc[x_labels][:, 0:3], 1.0)
        # batch_dict['points'] = pc[points_hm[0]].cpu().detach().numpy()
        # batch_dict['arbe_num'] = batch_dict['points'].shape[0]
        
        # if batch_dict['debug_mode'] == 1:
        #     import os
        #     current_working_directory = os.getcwd()
        #     parent_dir = os.path.dirname(current_working_directory)
        #     np.save(parent_dir + '/output_debug/filter_points.npy', batch_dict['points'])

        # gc.collect()
        # torch.cuda.empty_cache()
        # print("data segmentation...")
        # return batch_dict