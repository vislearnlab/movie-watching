#!/usr/bin/env bash
# Iterative annotation cycle. Each pass: prompts + corrections → SAM3 on every shot whose
# tasks changed (annotate_videos.py signatures) → merge → evaluate. Then it waits for
# inputs/corrections/ to change and runs the next pass. Run it inside tmux:
#
#   tmux new-window -t annotations -n cycle "bash pipeline/cycle.sh"
#
# Env overrides: GPUS="2 3", WORKERS_PER_GPU=2 (3 can run out of memory on long shots),
# ONCE=1 (one pass, then exit), PY=<python>.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANN="$(dirname "$HERE")"
PY="${PY:-/Scratch/tsepuri/miniconda3/envs/moviewatching/bin/python}"
GPUS=(${GPUS:-2 3})
WPG="${WORKERS_PER_GPU:-2}"
OUT="$ANN/output/current"
LOGS="$OUT/logs"
# Long crowded shots grow to ~18 GB per worker; this limits fragmentation between shots.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS=ignore
# torch/OpenMP default to one thread per core (128) in every worker, which oversubscribes the machine.
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
mkdir -p "$LOGS"
cd "$ANN"

while true; do
    pass=$(( $(grep -c "start" "$LOGS/cycle.log" 2>/dev/null || true) + 2 ))   # pass 1 was the Oct 21 run
    echo "=== pass $pass start $(date) on GPUs ${GPUS[*]} × $WPG ===" | tee -a "$LOGS/cycle.log"
    "$PY" pipeline/prepare_prompts.py > "$LOGS/prepare_prompts.log"
    tail -2 "$LOGS/prepare_prompts.log"
    stamp_file="$LOGS/.corrections_seen"; touch "$stamp_file"

    find "$OUT" -name "*.lock" -delete          # no worker is running, so any lock is stale
    pids=(); w=0
    for gpu in "${GPUS[@]}"; do
        for _ in $(seq "$WPG"); do
            "$PY" pipeline/annotate_videos.py run --gpu "$gpu" > "$LOGS/worker${w}.log" 2>&1 &
            pids+=($!); w=$((w + 1))
        done
    done
    failed=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "worker $i FAILED, see $LOGS/worker${i}.log"; failed=1; }
    done
    [ "$failed" -eq 0 ] || { echo "=== pass $pass FAILED $(date) ===" | tee -a "$LOGS/cycle.log"; exit 1; }

    "$PY" pipeline/annotate_videos.py merge > "$LOGS/merge.log"
    "$PY" pipeline/evaluate.py > "$LOGS/evaluate.log"
    echo "=== pass $pass done $(date) ===" | tee -a "$LOGS/cycle.log"
    [ -z "${ONCE:-}" ] || exit 0

    echo "waiting for changes in inputs/corrections/ …"
    until [ -n "$(find inputs/corrections -name '*.csv' -newer "$stamp_file")" ]; do sleep 60; done
done
