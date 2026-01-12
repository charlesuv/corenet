export CUDA_VISIBLE_DEVICES=2
Model_name=CasA-V_arbe_seg_detection #CasA-V_arbe_seg_detection #voxel_rcnn_arbe #CasA-V_arbe #
PYFile=cfgs/tjradar_models/${Model_name}.yaml
Extra_tag=default_casA_V_arbe_seg_detection
CKPT_DIR= your_ckpt_path

#CKPT_file=../output/dual_radar_models/${Model_name}/${Extra_tag}/ckpt/checkpoint_epoch_80.pth

python3 test.py \
    --cfg_file=${PYFile} \
    --ckpt_dir=${CKPT_DIR} \
    --batch_size=4 \
    --ckpt=${CKPT_file}  \
    --eval_all \
    --extra_tag=${Extra_tag} \
    --cuda_device=3 \
    --workers=4