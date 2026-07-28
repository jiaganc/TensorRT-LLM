#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimi K3 serving benchmark on 16 NVIDIA Blackwell GPUs.
#
# Brings up trtllm-serve, then drives the in-repo benchmark client
# (tensorrt_llm/serve/scripts/benchmark_serving.py) with synthetic random
# 8K-input/1K-output requests.
# Per concurrency C: one warmup pass (C requests, discarded), then one
# measured pass (10*C requests) saved as JSON.
#
# Submit the job with the container, repository, and configuration paths:
#
#   sbatch examples/kimi_k3/run_serving_benchmark_kimi_k3.sbatch \
#       --image /path/to/tensorrt-llm-container.sqsh \
#       --repo /path/to/TensorRT-LLM \
#       --config /path/to/TensorRT-LLM/examples/kimi_k3/serving_benchmark_llm_options.yaml
#
# The checkpoint path is fixed to the shared Kimi K3 model below. --config
# must contain the complete LLM configuration, including parallelism and
# capacity settings compatible with the requested input/output lengths.
#
# Optional arguments: --repo PATH, --isl N (default 8192), --osl N (default
# 1024), and --concurrencies "64 256". --repo defaults to REPO or the submit
# directory and must contain a built TensorRT-LLM checkout. The job creates
# one named container per node from --image and installs the checkout editable
# in each container.
#
# Results and server/client logs go to kimi-k3-serving-benchmark-<job-id>/
# in the submission directory.
#
# The --comment below requests an idle-GPU-reaper exemption on clusters that
# cancel jobs whose GPUs sit idle: server initialization and weight loading
# take 10-15 minutes before timed benchmark traffic starts. A command-line
# --comment overrides this directive; repeat it verbatim if you pass one.
#
#SBATCH --job-name=kimi-k3-serving-benchmark
#SBATCH --partition=batch
#SBATCH --account=${account}
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --output=kimi-k3-serving-benchmark-%j.log
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"45","reason":"benchmarking","description":"Kimi K3 16-GPU server initialization and weight loading takes 10-15 minutes before timed benchmark traffic"}}'

set -euo pipefail

usage() {
    echo "Usage: sbatch $0 --image FILE --config FILE [--repo PATH] [--isl N] [--osl N] [--concurrencies \"64 256\"]"
}

MODEL_DIR="/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model/moonshotai_kimi-k3/hf"
MODEL="$MODEL_DIR/hf-9f62e4e_orig"
CONTAINER_IMAGE=""
CONFIG_FILE=""
REPO_ARG=""
ISL=8192
OSL=1024
CONCURRENCIES="64 256"
PORT=8000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            [[ $# -ge 2 ]] || { echo "error: --image requires a value" >&2; usage >&2; exit 2; }
            CONTAINER_IMAGE=$2
            shift 2
            ;;
        --config)
            [[ $# -ge 2 ]] || { echo "error: --config requires a value" >&2; usage >&2; exit 2; }
            CONFIG_FILE=$2
            shift 2
            ;;
        --repo)
            [[ $# -ge 2 ]] || { echo "error: --repo requires a value" >&2; usage >&2; exit 2; }
            REPO_ARG=$2
            shift 2
            ;;
        --isl)
            [[ $# -ge 2 ]] || { echo "error: --isl requires a value" >&2; usage >&2; exit 2; }
            ISL=$2
            shift 2
            ;;
        --osl)
            [[ $# -ge 2 ]] || { echo "error: --osl requires a value" >&2; usage >&2; exit 2; }
            OSL=$2
            shift 2
            ;;
        --concurrencies)
            [[ $# -ge 2 ]] || { echo "error: --concurrencies requires a value" >&2; usage >&2; exit 2; }
            CONCURRENCIES=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n "$CONTAINER_IMAGE" ]] || { echo "error: --image is required" >&2; usage >&2; exit 2; }
[[ -n "$CONFIG_FILE" ]] || { echo "error: --config is required" >&2; usage >&2; exit 2; }
[[ -e "$MODEL" ]] || { echo "error: model path does not exist: $MODEL" >&2; exit 2; }
[[ -f "$CONTAINER_IMAGE" ]] || { echo "error: image file does not exist: $CONTAINER_IMAGE" >&2; exit 2; }
[[ -f "$CONFIG_FILE" ]] || { echo "error: config file does not exist: $CONFIG_FILE" >&2; exit 2; }
[[ "$ISL" =~ ^[0-9]+$ && "$OSL" =~ ^[0-9]+$ ]] || { echo "error: --isl/--osl must be integers" >&2; exit 2; }
for concurrency in $CONCURRENCIES; do
    [[ "$concurrency" =~ ^[0-9]+$ ]] || { echo "error: bad concurrency: $concurrency" >&2; exit 2; }
done

REPO=${REPO_ARG:-${REPO:-$SLURM_SUBMIT_DIR}}
[[ -d "$REPO" ]] || { echo "error: repo path does not exist: $REPO" >&2; exit 2; }
REPO=$(realpath "$REPO")
[[ -f "$REPO/setup.py" ]] || { echo "error: repo path is not a TensorRT-LLM checkout: $REPO" >&2; exit 2; }
CONFIG_FILE=$(realpath "$CONFIG_FILE")

WORK_DIR=$SLURM_SUBMIT_DIR/kimi-k3-serving-benchmark-$SLURM_JOB_ID
mkdir -p "$WORK_DIR"

# Shared warm caches, persistent across runs: without a pre-populated
# FLASHINFER_CUBIN_DIR, ranks on egress-less compute nodes can hang fetching
# cubins during the TRTLLM-Gen FMHA warmup.
CACHE_DIR=${CACHE_DIR:-$SLURM_SUBMIT_DIR/.kimi_k3_cache}
mkdir -p "$CACHE_DIR"

MOUNTS="$REPO:$REPO:rw,$MODEL:$MODEL:ro,$CONFIG_FILE:$CONFIG_FILE:ro,$WORK_DIR:$WORK_DIR:rw,$CACHE_DIR:$CACHE_DIR:rw"
if [ -n "${EXTRA_MOUNTS:-}" ]; then
    MOUNTS+=",$EXTRA_MOUNTS"
fi

CONTAINER_NAME="kimi-k3-serving-benchmark-$SLURM_JOB_ID"
srun --nodes="$SLURM_NNODES" \
    --ntasks="$SLURM_NNODES" \
    --ntasks-per-node=1 \
    --kill-on-bad-exit=1 \
    --container-image="$CONTAINER_IMAGE" \
    --container-name="$CONTAINER_NAME" \
    --container-mounts="$MOUNTS" \
    --container-workdir="$REPO" \
    bash -c '
        set -euo pipefail
        python3 -m pip install -e "$1"
    ' bash "$REPO"

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR == 1')

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Server: all 16 ranks; rank 0 (this node) binds the HTTP endpoint.
# --kill-on-bad-exit=1: any failed rank kills the whole step immediately.
srun --mpi=pmix --kill-on-bad-exit=1 \
    --container-name="$CONTAINER_NAME" \
    --container-mounts="$MOUNTS" \
    --container-workdir="$REPO" \
    --output="$WORK_DIR/serve.log" \
    bash -c "
        set -x
        ulimit -n 65536

        # Node-local Triton cache: ~/.triton on shared NFS races across
        # ranks during autotune JIT (stale file handles).
        export TRITON_CACHE_DIR=/tmp/triton-cache-rank\${SLURM_PROCID:-0}
        mkdir -p \"\$TRITON_CACHE_DIR\"
        # Shared warm caches (see the CACHE_DIR comment above).
        export FLASHINFER_WORKSPACE_BASE=$CACHE_DIR/fi_workspace
        export FLASHINFER_CUBIN_DIR=$CACHE_DIR/fi_cubins
        export HF_HOME=\${HF_HOME:-$CACHE_DIR/hf_home}
        mkdir -p \"\$FLASHINFER_WORKSPACE_BASE\" \"\$FLASHINFER_CUBIN_DIR\" \"\$HF_HOME\"

        # Exercise CommunicationFactory AUTO selection rather than inheriting
        # a caller-side forced communication strategy.
        unset TRTLLM_FORCE_COMM_METHOD

        exec trtllm-llmapi-launch trtllm-serve \
            \"$MODEL\" \
            --backend pytorch \
            --host 0.0.0.0 \
            --port $PORT \
            --config '$CONFIG_FILE'
    " &
SERVER_PID=$!

# Wait up to 25 min for health; on timeout or server death, cancel the job
# so the allocation never sits on idle GPUs.
deadline=$((SECONDS + 1500))
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null || (( SECONDS >= deadline )); then
        echo "error: server not healthy within 25 min (see $WORK_DIR/serve.log); cancelling job" >&2
        scancel "$SLURM_JOB_ID"
        sleep 120
        exit 1
    fi
    sleep 15
done
echo "Server is healthy; starting benchmark sweep (see $WORK_DIR/client.log)."

# Client on the head node, alongside the server. The random dataset is
# generated client-side (--random-ids --tokenize-on-client) so the server
# sees exactly ISL input tokens; --ignore-eos fixes output length at OSL.
srun --overlap --kill-on-bad-exit=1 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    --container-name="$CONTAINER_NAME" \
    --container-mounts="$MOUNTS" \
    --container-workdir="$REPO" \
    --output="$WORK_DIR/client.log" \
    bash -c "
        set -x

        run_client() {
            python3 -m tensorrt_llm.serve.scripts.benchmark_serving \
                --model \"$MODEL\" \
                --host 127.0.0.1 \
                --port $PORT \
                --dataset-name random \
                --random-ids \
                --tokenize-on-client \
                --random-input-len $ISL \
                --random-output-len $OSL \
                --ignore-eos \
                --trust-remote-code \
                \"\$@\"
        }

        for concurrency in $CONCURRENCIES; do
            echo \"=== Warmup: concurrency \$concurrency ===\"
            run_client --num-prompts \$concurrency --max-concurrency \$concurrency

            echo \"=== Measured: concurrency \$concurrency ===\"
            run_client \
                --num-prompts \$((concurrency * 10)) \
                --max-concurrency \$concurrency \
                --save-result \
                --result-dir '$WORK_DIR' \
                --result-filename \"concurrency_\${concurrency}.json\"
        done
    "

echo "Benchmark complete; results in $WORK_DIR"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
