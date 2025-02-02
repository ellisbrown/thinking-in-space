#!/bin/bash
set -e

log() {
    printf "\033[31m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

MODEL_FAMILY=llava_vid

# CLUSTERS="jupiter"
CLUSTERS="all"
GPUS=8
# GPUS=4

# PRETRAINED_LIST=(
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_27__video_lr1e-7"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_27__video_lr5e-6"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_27__video_50k_mc_50oe_lr1e-7"
# )

PRETRAINED_LIST=(
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
    "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_02_01__vid_mt1_grouped"
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
)

for PRETRAINED in "${PRETRAINED_LIST[@]}"; do
    log "Submitting job to cluster $CLUSTERS"
    log "Pretrained model: $PRETRAINED"
    echo ""

    bash scripts/ai2/launch_eval.sh $PRETRAINED --clusters $CLUSTERS --gpus $GPUS --model_family $MODEL_FAMILY


    log "Job submitted."
done
