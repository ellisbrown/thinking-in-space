#!/bin/bash
# eval_ov.sh

set -e

CKPT_PATH=$1
NUM_FRAMES=${2:-32}  # Default to 32 frames

log() {
    # printf "\033[31m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
    # no color
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

[ -z "$CKPT_PATH" ] && { log "Error: Checkpoint path required"; exit 1; }

# Set up environment
ENV="$PROJ_ROOT/env"
# log "Activating conda environment $ENV"
# conda activate "$ENV"
ACCELERATE=$ENV/bin/accelerate
echo "Using accelerate: $ACCELERATE"

# Build evaluation command
BENCHMARK=vsibench
OUTPUT_PATH=logs/$(TZ="America/New_York" date "+%Y%m%d")


# MODEL_FAMILY=llava_onevision
if [ "$MODEL_FAMILY" = "llava_onevision" ]; then
    MODEL_NAME="llava_one_vision_qwen2_7b_ov_${NUM_FRAMES}f"
    MODEL_ARGS="pretrained=$CKPT_PATH,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=$NUM_FRAMES"
elif [ "$MODEL_FAMILY" = "llava_vid" ]; then
    MODEL_NAME="llava_video_7b_qwen2_${NUM_FRAMES}f"
    MODEL_ARGS="pretrained=$CKPT_PATH,video_decode_backend=decord,conv_template=qwen_1_5,max_frames_num=$NUM_FRAMES"
else
    log "Invalid model family"
    exit 1
fi

ckpt_suffix=$(basename $CKPT_PATH)

PORT=$(( ( RANDOM % 64512 ) + 1024 ))

export LMMS_EVAL_LAUNCHER="accelerate"
CMD="$ACCELERATE launch \
    --num_processes=$N_GPUS \
    --main_process_port=$PORT \
    -m lmms_eval \
    --model $MODEL_FAMILY \
    --model_args $MODEL_ARGS \
    --tasks $BENCHMARK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $ckpt_suffix \
    --output_path $OUTPUT_PATH/$BENCHMARK"

log "Running evaluation command:"
echo "$CMD"
eval "$CMD"

log "Evaluation completed."


# collect results

PYTHON=$ENV/bin/python

CMD="$PYTHON $PROJ_ROOT/collect_vsibench.py"

log "Running collect command:"
echo "$CMD"
eval "$CMD"

log "Collection completed."

log "All done."