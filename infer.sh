#! /bin/bash
set -x

# common stories
echo "common stories br (vanilla)" 
CUDA_VISIBLE_DEVICES=5 python infer_checkpoints.py --dataset_path processed/common_stories_16_roc_6_wp_percent_br --prefix common_stories_16_roc_6_wp_percent_br_Vanilla_1_Cycles_0.0_WarmR_1e-05_lr_10
echo "common stories br (rstmix)" 
CUDA_VISIBLE_DEVICES=5 python infer_checkpoints.py --dataset_path processed/common_stories_16_roc_6_wp_percent_br --prefix common_stories_16_roc_6_wp_percent_br_RSTMix_1_Cycles_0.0_WarmR_1e-05_lr_10
echo "common stories br (posmix)" 
CUDA_VISIBLE_DEVICES=5 python infer_checkpoints.py --dataset_path processed/common_stories_16_roc_6_wp_percent_br --prefix common_stories_16_roc_6_wp_percent_br_POSMix_1_Cycles_0.0_WarmR_1e-05_lr_10

# faketruebr
# CUDA_VISIBLE_DEVICES=1 python infer_checkpoints.py --dataset_path data/faketruebr --prefix common_stories_16_roc_6_wp_percent_Vanilla_1_Cycles_0.0_WarmR_1e-05_lr_10
# CUDA_VISIBLE_DEVICES=1 python infer_checkpoints.py --dataset_path data/faketruebr --prefix common_stories_16_roc_6_wp_percent_RSTMix_1_Cycles_0.0_WarmR_1e-05_lr_10
# CUDA_VISIBLE_DEVICES=1 python infer_checkpoints.py --dataset_path data/faketruebr --prefix common_stories_16_roc_6_wp_percent_POSMix_1_Cycles_0.0_WarmR_1e-05_lr_10

# CUDA_VISIBLE_DEVICES=1 python infer_checkpoints.py --dataset_path data/faketruebr --prefix common_stories_16_roc_6_wp_percent_br_Vanilla_1_Cycles_0.0_WarmR_1e-05_lr_10
# CUDA_VISIBLE_DEVICES=1 python infer_checkpoints.py --dataset_path data/faketruebr --prefix common_stories_16_roc_6_wp_percent_br_RSTMix_1_Cycles_0.0_WarmR_1e-05_lr_10
# CUDA_VISIBLE_DEVICES=1 python infer_checkpoints.py --dataset_path data/faketruebr --prefix common_stories_16_roc_6_wp_percent_br_POSMix_1_Cycles_0.0_WarmR_1e-05_lr_10

