export CUDA_VISIBLE_DEVICES=3
Model_name=CasA-V_arbe_seg_detection #CasA-V_arbe #
PYFile=cfgs/tjradar_models/${Model_name}.yaml
Extra_tag=default_casA_V_arbe_seg_detection_test
# CKPT_file=../output/dual_radar_models/${Model_name}/${Extra_tag}/ckpt/checkpoint_epoch_80.pth
CKPT_file=../output/tjradar_models/CasA-V_arbe_seg_detection/default_casA_V_arbe_seg_detection/ckpt/checkpoint_epoch_80.pth

python3 test.py \
    --cfg_file=${PYFile} \
    --batch_size=4 \
    --ckpt=${CKPT_file}  \
    --extra_tag=${Extra_tag} \
    --cuda_device=1