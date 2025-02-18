#!/bin/bash
set -e

log() {
    printf "\033[31m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

MODEL_FAMILY=llava_vid

# CLUSTER="jupiter"

CLUSTER="all"
# GPUS=4

# CLUSTER="80gb"
GPUS=8

# PRETRAINED_LIST=(
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_27__video_lr1e-7"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_27__video_lr5e-6"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_27__video_50k_mc_50oe_lr1e-7"
# )

# PRETRAINED_LIST=(
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_29__video_50k_mc_50oe_lr1e-5"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_29__video_50k_mc_50oe_lr1e-6"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_29__video_50k_mc_50oe_lr5e-6"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_29__video_50k_mc_50oe_lr5e-7"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_29__video_lr1e-5"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_29__video_lr1e-6"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_29__video_lr5e-7"

    # 2025_02_01
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_29__video_50k_mc_50oe_lr5e-6"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_29__video_50k_mc_50oe_lr1e-5"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_29__video_50k_mc_50oe_lr5e-7"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_29__video_50k_mc_50oe_lr1e-6"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_ct"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_rel_dist"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_rel_dir"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_temp"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_desc"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt5_mixed"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt5_grouped"

    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt3_mixed"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_grouped"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt3_grouped"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_ct_oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_rel_dist_oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_rel_dir_oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_temp_oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_desc_oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__video_50k_mc_50oe_lr1e-6"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_mixed"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__video_50k_mc_70oe_lr1e-6"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__video_50k_mc_100oe_lr1e-6"
# )


# # 2025_02_09
# PRETRAINED_LIST=(
#     # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_09__vid_1k_mt5_grouped"
#     # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_09__vid_1k_mt3_mixed"
#     # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_09__vid_1k_mt3_grouped"
#     # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_09__vid_1k_mt5_mixed"
#     # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_09__vid_1k_mt1_mixed"
#     "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_09__vid_1k_mt1_grouped"
# )

# 2025_02_10
PRETRAINED_LIST=(
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_10__vid_mt1_abs_dist_oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_10__vid_mt1_long_size_est_oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_10__vid_mt1_long_size_est_mc"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_10__vid_mt1_short_size_est_oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_10__vid_mt1_short_size_est_mc"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_09__vid_1k_mt1_mixed"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_10__vid_mt1_abs_dist"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_10__vid_mt1_long_size_est_v2_oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_10__vid_mt1_long_size_est_v2_mc"
)


# for PRETRAINED in "${PRETRAINED_LIST[@]}"; do
#     log "Submitting job to cluster $CLUSTER"
#     log "Pretrained model: $PRETRAINED"
#     echo ""

#     bash scripts/ai2/launch_eval.sh $PRETRAINED --clusters $CLUSTER --gpus $GPUS --model_family $MODEL_FAMILY


#     log "Job submitted."
# done


CHECKPOINT_DIR="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision"
CHECKPOINTS=(
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_ct
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_rel_dist
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_rel_dir
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_temp
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_desc
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_ct_oe
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_rel_dist_oe
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_rel_dir_oe
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_temp_oe
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_desc_oe
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_10__vid_mt1_abs_dist
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_10__vid_mt1_abs_dist_oe
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_10__vid_mt1_long_size_est_mc
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_10__vid_mt1_long_size_est_oe
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_10__vid_mt1_short_size_est_mc
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_10__vid_mt1_short_size_est_oe
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_10__vid_mt1_long_size_est_v2_mc
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_10__vid_mt1_long_size_est_v2_oe

    # # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_11__vid_ai2_SAT_10k
    # # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_11__vid_ai2_SAT_50k
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_11__vid_mt1_house_size_est_mc
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_11__vid_mt1_house_size_est_oe
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_11__vid_mt1_n_rooms_mc
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_11__vid_mt1_n_rooms_oe
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_11__vid_mt1_temporal_order_2_mc
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_11__vid_mt1_temporal_order_2_oe
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_11__vid_mt1_temporal_order_3_mc
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_11__vid_mt1_temporal_order_3_oe
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_11__vid_mt1_temporal_order_4_mc
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_11__vid_mt1_temporal_order_4_oe
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_11__vid_mt1_temporal_order_5_mc
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_11__vid_mt1_temporal_order_5_oe

    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_12__vid_mt1_mix1
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_12__vid_mt3_mix1
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_12__vid_mt1_mix1_50k

    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_13__vid_mt1_vsi_mix1
    # ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_13__vid_mt1_vsi_mix1_50k
    ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_13__vid_mt1_mix_abs_dist_rel_dir
)

FRAMES=64
FRAMES=32

log "Submitting jobs to cluster $CLUSTER"
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    PRETRAINED="$CHECKPOINT_DIR/$CHECKPOINT"
    echo "Pretrained: $PRETRAINED"
    bash scripts/ai2/launch_eval.sh $PRETRAINED --clusters $CLUSTER --gpus $GPUS --model_family $MODEL_FAMILY --frames $FRAMES &
done


# bash scripts/ai2/launch_eval.sh /data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_11__vid_mt1_temporal_order_5_oe --clusters all --gpus 8 --model_family llava_vid_no_decord
