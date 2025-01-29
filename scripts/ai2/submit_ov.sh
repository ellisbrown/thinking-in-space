#!/bin/bash
set -e

log() {
    printf "\033[31m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

CLUSTER="jupiter"

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
PRETRAINED_LIST=(
    "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_27__spoc_50k_100mc_50oe_lr1e-8"
    "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_27__spoc_50k_100mc_50oe_lr5e-9"
    "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_27__spoc_ov20_100mc_50oe_lr1e-7"
    "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_27__spoc_ov20_100mc_50oe"
    "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_26__spoc_ov_v4"
)

for PRETRAINED in "${PRETRAINED_LIST[@]}"; do
    log "Submitting job to cluster $CLUSTER"
    log "Pretrained model: $PRETRAINED"
    echo ""

    bash scripts/ai2/launch_eval.sh $PRETRAINED --clusters $CLUSTER

    log "Job submitted."
done
