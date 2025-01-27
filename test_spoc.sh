#! /bin/bash

set -e


# Function to print logs with timestamp
log() {
    printf "\033[31m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}


if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    log "CUDA_VISIBLE_DEVICES is not set. Using all GPUs."
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    log "CUDA_VISIBLE_DEVICES is set to $CUDA_VISIBLE_DEVICES."
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    gpu_count=${#devices[@]}
fi
log "Number of GPUs: $gpu_count"


BATCH_SIZE=1
GPU_MEM=$( nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0  | awk '{print $1 / 1024}' )
# ERROR: LLaVA does not support batched generation
# if [ "$GPU_MEM" -gt 48 ]; then
#     BATCH_SIZE=4
# else
#     BATCH_SIZE=1
# fi
log "GPU Memory: $GPU_MEM GB."
log "Batch size set to: $BATCH_SIZE"


export CUDA_LAUNCH_BLOCKING=1

benchmark=vsibench
output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")
num_processes=$gpu_count
num_frames=32
launcher=accelerate


# PRETRAINED="lmms-lab/llava-onevision-qwen2-7b-ov"
# PRETRAINED="lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/llava-onevision-qwen2-7b-ov"
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-onevision-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9"
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9_half_lr_20250108_180208"  # Half LR,  MC Data
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9_half_lr_oe_20250108_221111"  # Half LR, OE Data
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9_half_lr_mc+oe_20250108_222050"  # Half LR, MC+OE Data
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_64F_half_lr_mc_20250109_192405"  # video MC "test"
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_64F__half_lr_mc_20250110_172251"  # video MC 64F
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-video-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_32F_half_lr_mc_20250114_230834"  # video MC 32F


# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9_32F_half_lr_mc_20250116_032249/checkpoint-1000"  # spoc+ovdata ckpt 1k
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc+ov-ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9_32F_ovdataV2_20250117_185829/checkpoint-11000"  # spoc+ovdata V2
# PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_20__spoc_ov_v3"  # spoc+ovdata V3


# LR Sweep
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_21__sweep_lr5e-6_vlr5e-6"
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_21__sweep_lr5e-6_vlr1e-7"
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_21__sweep_lr5e-6_vlr1e-6"
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_21__sweep_lr5e-6_vlr1e-6_gbs128"
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_21__sweep_lr1e-7_vlr1e-6"
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/ft-llava-ov-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-32F_vidS2R_2025_01_21__sweep_lr1e-6_vlr1e-6"


model_family="llava_onevision"
model="llava_one_vision_qwen2_7b_ov_${num_frames}f"
model_args="pretrained=${PRETRAINED},conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=$num_frames"



# PORT=29500
PORT=$(( ( RANDOM % 64512 ) + 1024 ))


export LMMS_EVAL_LAUNCHER="accelerate"
evaluate_script="accelerate launch \
    --num_processes=$num_processes \
    --main_process_port=$PORT \
    "

evaluate_script="$evaluate_script -m lmms_eval \
    --model $model_family \
    --model_args $model_args \
    --tasks $benchmark \
    --batch_size $BATCH_SIZE \
    --log_samples \
    --log_samples_suffix $model \
    --output_path $output_path/$benchmark"

log "Running command:"
echo $evaluate_script
eval $evaluate_script

log "Done."
