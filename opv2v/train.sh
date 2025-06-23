# Training (NOTE: fix a typo here) => result very bad
# CUDA_VISIBLE_DEVICES=0,1 \
#     python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     --use_env opencood/tools/train_camera.py \
#     --hypes_yaml opencood/hypes_yaml/opcamera/corpbevt.yaml


# Ref: https://github.com/DerrickXuNu/CoBEVT/issues/3

# # 1) Trained the fax.yaml for 90 epochs => DONE
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     python -m torch.distributed.launch \
#     --nproc_per_node=8 \
#     --use_env opencood/tools/train_camera.py \
#     --hypes_yaml opencood/hypes_yaml/opcamera/fax.yaml

# 1.5) Move pretrained model and config to finetune folder
# NOTE: use this timestep as a example
timestep=2025_06_22_17_40_34

# mkdir -p opencood/logs/corpbevt_${timestep}
# cp opencood/logs/fax_${timestep}/net_epoch86.pth opencood/logs/corpbevt_${timestep}/net_epoch1.pth
# cp opencood/hypes_yaml/opcamera/corpbevt.yaml opencood/logs/corpbevt_${timestep}/config.yaml


# # 2) Finetune with corpbevt.yaml fror 80 epochs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env opencood/tools/train_camera.py \
    --hypes_yaml opencood/logs/corpbevt_${timestep}/config.yaml \
    --model_dir opencood/logs/corpbevt_${timestep}

# 3) change yaml so we test on testset, not valset
# TODO: after this reproduce, try reproducing static results