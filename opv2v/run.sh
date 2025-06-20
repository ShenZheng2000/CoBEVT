# Inference
PRE="opencood/logs/CoBEVT_Models-20240806T195322Z-001/CoBEVT_Models/cobevt"
PRE_STATIC="opencood/logs/CoBEVT_Models-20240806T195322Z-001/CoBEVT_Models/cobevt_static"

python opencood/tools/inference_camera.py --model_dir $PRE
python opencood/tools/inference_camera.py --model_dir $PRE_STATIC --model_type static
python opencood/tools/merge_dynamic_static.py --dynamic_path $PRE --static_path $PRE_STATIC --output_path merge_results


# Training (NOTE: fix a typo here)
python opencood/tools/train_camera.py --hypes_yaml opencood/hypes_yaml/opcamera/corpbevt.yaml

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env opencood/tools/train_camera.py --hypes_yaml opencood/hypes_yaml/opcamera/corpbevt.yaml