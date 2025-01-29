#!/bin/bash
set -e

log() {
    printf "\033[31m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

MODEL_FAMILY=llava_vid

# CLUSTERS="jupiter"
CLUSTERS="all"
# GPUS=8
GPUS=4

PRETRAINED_LIST=(
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_27__video_lr1e-7"
    # "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_27__video_lr5e-6"
    "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_27__video_50k_mc_50oe_lr1e-7"
)


for PRETRAINED in "${PRETRAINED_LIST[@]}"; do
    log "Submitting job to cluster $CLUSTERS"
    log "Pretrained model: $PRETRAINED"
    echo ""

    bash scripts/ai2/launch_eval.sh $PRETRAINED --clusters $CLUSTERS --gpus $GPUS --model_family $MODEL_FAMILY


    log "Job submitted."
done
