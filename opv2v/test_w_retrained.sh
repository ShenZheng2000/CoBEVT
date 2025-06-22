
# # for retrained model inference 
# # NOTE: hand-change validate_dir to test/ in *.yaml in result folders
run_inference() {
    MODEL_DIR="$1"

    python opencood/tools/inference_camera.py --model_dir "$MODEL_DIR"
}

# run_inference opencood/logs/fax_2025_06_18_21_19_31
# run_inference opencood/logs/fax_2025_06_18_21_19_31_epoch_91
# run_inference opencood/logs/fax_2025_06_21_03_41_22
# run_inference opencood/logs/fax_2025_06_21_03_41_22_epoch_91