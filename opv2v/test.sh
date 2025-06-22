# # for pretrained model inference
run_inference() {
    BASE="$1"
    PRE="$BASE/cobevt"
    PRE_STATIC="$BASE/cobevt_static"
    PRE_MERGE="$BASE/cobevt_merge"

    python opencood/tools/inference_camera.py --model_dir "$PRE"
    python opencood/tools/inference_camera.py --model_dir "$PRE_STATIC" --model_type static
    python opencood/tools/merge_dynamic_static.py --dynamic_path "$PRE" --static_path "$PRE_STATIC" --output_path "$PRE_MERGE"
}

# # Example usage:
# run_inference opencood/logs/CoBEVT_Models-20240806T195322Z-001/CoBEVT_Models

# TODO: write a new script to demo on 2 run segments (in argoverse v2)
#  → 5ccb359a-2986-466c-88b2-a16f51774a8f ↔ f7cf93d8-f7bd-3799-8500-fbe842a96f63 within 50.0m

# TODO: argoverse v2 does not have rear cameras, '
# should we try nuscenes instead (at least have single-agent source code)