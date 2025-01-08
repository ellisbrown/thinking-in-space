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


export CUDA_LAUNCH_BLOCKING=1

benchmark=vsibench

benchmark=vsibench
output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")
num_processes=$gpu_count
num_frames=32
launcher=accelerate


# PRETRAINED="lmms-lab/llava-onevision-qwen2-7b-ov"
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/llava-onevision-qwen2-7b-ov"
PRETRAINED="/data/weka/ellisb/LLaVA-NeXT/checkpoints/onevision/spoc-ft-llava-onevision-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9"

model_family="llava_onevision"
model="llava_one_vision_qwen2_7b_ov_${num_frames}f"
model_args="pretrained=${PRETRAINED},conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=$num_frames"



export LMMS_EVAL_LAUNCHER="accelerate"
evaluate_script="accelerate launch \
    --num_processes=$num_processes \
    "

evaluate_script="$evaluate_script -m lmms_eval \
    --model $model_family \
    --model_args $model_args \
    --tasks $benchmark \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $model \
    --output_path $output_path/$benchmark \
    "

log "Running command:"
echo $evaluate_script
eval $evaluate_script

log "Done."
