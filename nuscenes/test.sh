# NOTE: i hardcode ckpt_path in test_try,py. need to modify later
python scripts/test_try.py \
  +experiment=cvt_pyramid_axial_nuscenes_vehicle \
  data.dataset_dir=/home/shenzheng_google_com/Projects/Inf_Perception/Datasets/nuscenes \
  data.labels_dir=/home/shenzheng_google_com/Projects/Inf_Perception/Datasets/cvt_labels_nuscenes_v2