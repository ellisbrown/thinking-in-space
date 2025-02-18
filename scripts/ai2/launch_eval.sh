#!/bin/bash

# Default values
GPUS=8
SHARED_MEMORY="250GiB"
CLUSTERS="all"
NUM_FRAMES=32
MODEL_FAMILY=llava_onevision

log() {
    printf "\033[34m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

print_help() {
    echo "Usage: $0 [options] <checkpoint-path>"
    echo ""
    echo "Options:"
    echo "  --gpus <number>       Number of GPUs (default: 8)"
    echo "  --shared_mem <size>   Shared memory size (default: 250GiB)"
    echo "  --clusters <name>     Cluster name or combination (default: all)"
    echo "                        Examples:"
    echo "                          jupiter"
    echo "                          jupiter+saturn"
    echo "                          all"
    echo "  --frames <num>        Number of frames (default: 32)"
    echo "  --model_family <name> Model family (default: llava_onevision)"
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
        --clusters)
            CLUSTERS="$2"
            shift 2 ;;
        --frames)
            NUM_FRAMES="$2"
            shift 2 ;;
        --model_family)
            MODEL_FAMILY="$2"
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

# ------------------------------------------------------------------------------
# A simple function that maps a short cluster name to its full name.
# Returns an empty string if the short name is unknown.
# ------------------------------------------------------------------------------
get_cluster_fullname() {
    case "$1" in
        saturn)   echo "ai2/saturn-cirrascale" ;;
        jupiter)  echo "ai2/jupiter-cirrascale-2" ;;
        ceres)    echo "ai2/ceres-cirrascale" ;;
        neptune)  echo "ai2/neptune-cirrascale" ;;
        *)        echo "" ;;
    esac
}

# ------------------------------------------------------------------------------
# parse_clusters():
# Accepts a single cluster name, plus-separated cluster names (e.g. "jupiter+saturn"),
# or the special keyword "all" to get *all* known clusters.
# Returns an array of cluster_fullnames via stdout.
# ------------------------------------------------------------------------------
parse_clusters() {
    local input="$1"
    local cluster_fullname
    local cluster_array=()

    if [[ "$input" == "all" ]]; then
        # Add all known clusters. Adjust as needed.
        cluster_array+=($(get_cluster_fullname "jupiter"))
        cluster_array+=($(get_cluster_fullname "ceres"))
        cluster_array+=($(get_cluster_fullname "saturn"))
        cluster_array+=($(get_cluster_fullname "neptune"))
    elif [[ "$input" == "80gb" ]]; then
        # Add all 80GB clusters (A100 + H100)
        cluster_array+=($(get_cluster_fullname "jupiter"))
        cluster_array+=($(get_cluster_fullname "ceres"))
        cluster_array+=($(get_cluster_fullname "saturn"))
    else
        # Split on "+" sign for multiple clusters
        IFS='+' read -ra splitted <<< "$input"
        for c in "${splitted[@]}"; do
            cluster_fullname="$(get_cluster_fullname "$c")"
            if [[ -z "$cluster_fullname" ]]; then
                log "Invalid cluster: $c"
                exit 1
            fi
            cluster_array+=("$cluster_fullname")
        done
    fi

    # Echo them space-separated so we can capture in an array later
    echo "${cluster_array[@]}"
}

# ------------------------------------------------------------------------------
# Parse the user-provided cluster(s) into an array
# ------------------------------------------------------------------------------
read_clusters=($(parse_clusters "$CLUSTERS"))
[[ ${#read_clusters[@]} -eq 0 ]] && {
    log "No valid cluster(s) specified."
    exit 1
}

clusters_yaml=""
for c in "${read_clusters[@]}"; do
    clusters_yaml+="'${c}',"
done

CLUSTERS_YAML=""
for c in "${READ_CLUSTERS[@]}"; do
    CLUSTERS_YAML+="'${c}',"
done

# Build description
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
YAML_PATH="$DIR/beaker_eval_base.yaml"
DESCRIPTION="eval_$(basename $CKPT_PATH | tr '/' '_')"
EXTENDED_DESC="${CLUSTERS}_1x${GPUS}_${DESCRIPTION}"

# Export environment variables
export GPUS
export SHARED_MEMORY
export CLUSTERS=$clusters_yaml
export DESCRIPTION
export EXTENDED_DESCRIPTION=$EXTENDED_DESC
export CKPT_PATH
export NUM_FRAMES
export MODEL_FAMILY

# Submit job
CMD="beaker experiment create $YAML_PATH"
log "Submitting evaluation job:"
echo "Checkpoint: $CKPT_PATH"
echo "GPUs: $GPUS | Clusters: $CLUSTERS | Frames: $NUM_FRAMES"
eval "$CMD"