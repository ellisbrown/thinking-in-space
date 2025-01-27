#!/bin/bash

# Default values
GPUS=8
SHARED_MEMORY="250GiB"
CLUSTER="jupiter"
NUM_FRAMES=32

log() {
    printf "\033[34m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

print_help() {
    echo "Usage: $0 [options] <checkpoint-path>"
    echo ""
    echo "Options:"
    echo "  --gpus <number>       Number of GPUs (default: 8)"
    echo "  --shared_mem <size>   Shared memory size (default: 250GiB)"
    echo "  --cluster <name>      Cluster name (default: jupiter)"
    echo "  --frames <num>        Number of frames (default: 32)"
    echo "  --help                Show help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2 ;;
        --shared_mem)
            SHARED_MEMORY="$2"
            shift 2 ;;
        --cluster)
            CLUSTER="$2"
            shift 2 ;;
        --frames)
            NUM_FRAMES="$2"
            shift 2 ;;
        --help)
            print_help
            exit 0 ;;
        *)
            CKPT_PATH="$1"
            shift ;;
    esac
done

[ -z "$CKPT_PATH" ] && { log "Error: Checkpoint path required"; print_help; exit 1; }

# Cluster mapping
case $CLUSTER in
    saturn) cluster_fullname="ai2/saturn-cirrascale" ;;
    jupiter) cluster_fullname="ai2/jupiter-cirrascale-2" ;;
    ceres) cluster_fullname="ai2/ceres-cirrascale" ;;
    neptune) cluster_fullname="ai2/neptune-cirrascale" ;;
    *) log "Invalid cluster"; exit 1 ;;
esac

# Build description
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
YAML_PATH="$DIR/beaker_eval_base.yaml"
DESCRIPTION="eval_$(basename $CKPT_PATH | tr '/' '_')"
EXTENDED_DESC="${CLUSTER}_1x${GPUS}_${DESCRIPTION}"

# Export environment variables
export GPUS
export SHARED_MEMORY
export CLUSTER=$cluster_fullname
export DESCRIPTION
export EXTENDED_DESCRIPTION=$EXTENDED_DESC
export CKPT_PATH
export NUM_FRAMES

# Submit job
CMD="beaker experiment create $YAML_PATH"
log "Submitting evaluation job:"
echo "Checkpoint: $CKPT_PATH"
echo "GPUs: $GPUS | Cluster: $CLUSTER | Frames: $NUM_FRAMES"
eval "$CMD"