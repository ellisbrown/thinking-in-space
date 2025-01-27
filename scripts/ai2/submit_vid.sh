#!/bin/bash

log() {
    printf "\033[31m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

CLUSTER="jupiter"
MODEL_FAMILY=llava_vid

PRETRAINED_LIST=(
    "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_27__video_lr1e-7"
    "/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-64F_vidS2R_2025_01_27__video_lr5e-6"
)


for PRETRAINED in "${PRETRAINED_LIST[@]}"; do
    log "Submitting job to cluster $CLUSTER"
    log "Pretrained model: $PRETRAINED"
    echo ""

    bash scripts/ai2/launch_eval.sh $PRETRAINED --cluster $CLUSTER --model_family $MODEL_FAMILY

    log "Job submitted."
done
