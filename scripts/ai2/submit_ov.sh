#!/bin/bash
set -e

log() {
    printf "\033[31m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

# CLUSTER="jupiter"
CLUSTER="all"
# GPUS=8
GPUS=4

# PRETRAINED="lmms-lab/llava-onevision-qwen2-7b-ov"
# PRETRAINED="lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/llava-onevision-qwen2-7b-ov"
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-onevision-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9"
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9_half_lr_20250108_180208"  # Half LR,  MC Data
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9_half_lr_oe_20250108_221111"  # Half LR, OE Data
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9_half_lr_mc+oe_20250108_222050"  # Half LR, MC+OE Data
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_64F_half_lr_mc_20250109_192405"  # video MC "test"
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_64F__half_lr_mc_20250110_172251"  # video MC 64F
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_32F_half_lr_mc_20250114_230834"  # video MC 32F


# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9_32F_half_lr_mc_20250116_032249/checkpoint-1000"  # spoc+ovdata ckpt 1k
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc+ov-ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9_32F_ovdataV2_20250117_185829/checkpoint-11000"  # spoc+ovdata V2
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_20__spoc_ov_v3"  # spoc+ovdata V3


# LR Sweep
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_21__sweep_lr5e-6_vlr5e-6"
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_21__sweep_lr5e-6_vlr1e-7"
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_21__sweep_lr5e-6_vlr1e-6"
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_21__sweep_lr5e-6_vlr1e-6_gbs128"
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_21__sweep_lr1e-7_vlr1e-6"
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_21__sweep_lr1e-6_vlr1e-6"


# PRETRAINED_LIST=(
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_26__spoc_50k_mc"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_26__spoc_50k_mc_lr1e-7"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_26__spoc_50k_100mc_20oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_26__spoc_50k_100mc_50oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_26__spoc_50k_100mc_50oe_lr1e-7"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_27__spoc_50k_100mc_50oe_lr5e-8"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_27__spoc_50k_100mc_80oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_27__spoc_50k_100mc_80oe_lr1e-7"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_27__spoc_50k_100mc_100oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_27__spoc_50k_100mc_100oe_lr1e-7"
# )

# 2025_01_29
# PRETRAINED_LIST=(
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_27__spoc_50k_100mc_50oe_lr1e-8"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_27__spoc_50k_100mc_50oe_lr5e-9"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_27__spoc_ov20_100mc_50oe_lr1e-7"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_27__spoc_ov20_100mc_50oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_26__spoc_ov_v4"

    # 2025_01_31
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_30__mt5_mixed"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_30__mt3_grouped"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_30__mt3_mixed"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_30__mt5_grouped"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_30__mt1_grouped"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_30__mt1_mixed"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_29__spoc_ov20_100mc_50oe_lr1e-5"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_29__spoc8k_ov30_100mc_80oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_29__spoc8k_ov20_100mc_80oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_29__spoc_ov20_100mc_50oe_lr1e-6"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_29__spoc_ov20_100mc_50oe_lr5e-7"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_29__spoc8k_ov40_100mc_80oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_29__spoc8k_ov10_100mc_80oe"
# )


# # 2025_02_01
# PRETRAINED_LIST=(
#     # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_01__mt1_ct"
#     # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_01__mt1_rel_dist"
#     # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_01__mt1_rel_dir"
#     # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_01__mt1_temp"
#     # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_01__mt1_desc"

#     "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_01__mt1_ct_oe"
#     "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_01__mt1_rel_dist_oe"
#     "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_01__mt1_rel_dir_oe"
#     "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_01__mt1_temp_oe"
#     "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_01__mt1_desc_oe"
# )

# 2025_02_09
PRETRAINED_LIST=(
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_09__ov_1k_mt5_mixed"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_09__ov_1k_mt5_grouped"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_09__ov_1k_mt3_mixed"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_09__ov_1k_mt3_grouped"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_09__ov_1k_mt1_mixed"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_09__ov_1k_mt1_grouped"

    # 2025_02_10
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_10__ov_mt1_abs_dist_oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_10__ov_mt1_abs_dist_mc"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_10__ov_mt1_long_size_est_oe"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_10__ov_mt1_long_size_est_mc"
    "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_10__ov_mt1_long_size_est_v2_oe"
    "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_10__ov_mt1_long_size_est_v2_mc"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_10__ov_mt1_short_size_est_mc"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_10__ov_mt1_short_size_est_oe"
)

# for PRETRAINED in "${PRETRAINED_LIST[@]}"; do
#     log "Submitting job to cluster $CLUSTER"
#     log "Pretrained model: $PRETRAINED"
#     echo ""

#     bash scripts/ai2/launch_eval.sh $PRETRAINED --clusters $CLUSTER --gpus $GPUS

#     log "Job submitted."
# done

# TASKS=("house_size_est" "n_rooms" "temporal_order_2" "temporal_order_3" "temporal_order_4" "temporal_order_5")

CHECKPOINT_DIR="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision"
CHECKPOINTS=(
    # ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_11__ov_ai2_SAT_10k
    # ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_11__ov_ai2_SAT_50k
    ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_11__ov_mt1_house_size_est_mc
    ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_11__ov_mt1_house_size_est_oe
    ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_11__ov_mt1_n_rooms_mc
    ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_11__ov_mt1_n_rooms_oe
    # ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_11__ov_mt1_temporal_order_2_mc
    ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_11__ov_mt1_temporal_order_2_oe
    ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_11__ov_mt1_temporal_order_3_mc
    # ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_11__ov_mt1_temporal_order_3_oe
    # ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_11__ov_mt1_temporal_order_4_mc
    ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_11__ov_mt1_temporal_order_4_oe
    # ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_11__ov_mt1_temporal_order_5_mc
    ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_02_11__ov_mt1_temporal_order_5_oe
)

log "Submitting jobs to cluster $CLUSTER"
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    PRETRAINED="$CHECKPOINT_DIR/$CHECKPOINT"
    echo "Pretrained: $PRETRAINED"
    bash scripts/ai2/launch_eval.sh $PRETRAINED --clusters $CLUSTER --gpus $GPUS &
done
