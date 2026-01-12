export CUDA_VISIBLE_DEVICES=3
Model_name=CasA-V_arbe_seg_detection #voxel_rcnn_arbe_seg_detection_explicit #voxel_rcnn_arbe_seg_detection #CasA-V_arbe #CasA-V_arbe_seg_detection #voxel_rcnn_arbe
LogName=../logs/${Model_name}.log
PYFile=cfgs/tjradar_models/${Model_name}.yaml

python3 train.py \
    --cfg_file=${PYFile} \
    --batch_size=4 \
    --epochs=60 \
    --extra_tag='default_tiny_casA_V_arbe_seg_detection' \
    --batch_size=4 \
    --cuda_device=3