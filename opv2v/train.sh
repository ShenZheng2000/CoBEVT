# Training (NOTE: fix a typo here)
# CUDA_VISIBLE_DEVICES=0,1 \
#     python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     --use_env opencood/tools/train_camera.py \
#     --hypes_yaml opencood/hypes_yaml/opcamera/corpbevt.yaml

# TODO: use 8 GPUs, train corpbevt2.yaml, trying to reproduce the dynamic results in the paper (Table 1)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env opencood/tools/train_camera.py \
    --hypes_yaml opencood/hypes_yaml/opcamera/corpbevt2.yaml

# after that, reproduce static results in the paper (Table 1)
# meanwhile, try inference pretrained model on argoverse v2 run segmetns. 