from .detector3d_template import Detector3DTemplate_seg_detect
import torch.nn.functional as F


class VoxelRCNN_seg_detect(Detector3DTemplate_seg_detect):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def visual_save(self, batch_dict, pred_dicts, path='vis/'):
        import numpy as np

        # 假设你有一个形状为[N, 3]的PyTorch张量，代表N个3D点
        points_raw_radar = batch_dict['raw_points'][:, 1:4]
        points_valid_radar = batch_dict['points'][:, 1:4]
        points_lidar = batch_dict['points_lidar'][:, 1:4]
        # 3d box
       # 假设你有多个3D包围框
        gt_boxes = batch_dict['gt_boxes'].cpu().numpy()
        pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
        # 转换为numpy数组
        points_raw_radar_np = points_raw_radar.cpu().numpy()
        points_valid_radar_np = points_valid_radar.cpu().numpy()
        points_lidar_np = points_lidar.cpu().numpy()

        vis_list = {
            'points_raw_radar': points_raw_radar_np,
            'points_valid_radar': points_valid_radar_np,
            'points_lidar': points_lidar_np,
            'pred_boxes': pred_boxes,
            'gt_boxes': gt_boxes[0, :, :7]
        }
        # np.save('vis_list.npy', vis_list)
        np.save(path+ batch_dict['frame_id'][0] +'_pre.npy', vis_list)
        return vis_list

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:

            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts, = self.post_processing(batch_dict)

            # 测试用
            path = 'vis/iros/'
            vis_list = self.visual_save(batch_dict, pred_dicts, path)

            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss =  loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict
