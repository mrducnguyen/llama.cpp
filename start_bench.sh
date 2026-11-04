#!/usr/bin/env bash
#
# Benchmark all discovered GGUF models from HuggingFace cache.
#
# Usage: ./start_bench.sh [output_file]
#   output_file: where to save results (default: bench_results.txt)
#
# Discovers all .gguf models in the HuggingFace cache, runs llama-bench
# on each, and writes results to a file. Models that fail to load (e.g.,
# CUDA OOM) are marked with [FAIL] and the error is captured.

set -euo pipefail

cd "$(dirname "$0")"

BENCH_TOOL="./build/bin/llama-bench"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface/hub}"
RESULTS_FILE="${1:-bench_results.txt}"

# fork perf env vars, inherited by all model instances
export GGML_CUDA_REGISTER_HOST=1
export GGML_SCHED_PREFETCH_EXPERTS=1

if [[ ! -x "$BENCH_TOOL" ]]; then
    echo "Error: llama-bench not found at $BENCH_TOOL"
    echo "Build first: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
    exit 1
fi

echo "=== llama.cpp Model Benchmark Suite ==="
echo "HuggingFace cache: $HF_CACHE"
echo "Results will be saved to: $RESULTS_FILE"
echo

# Discover all .gguf model files (exclude draft/MTP models and vocab files)
# Use -L to follow symlinks
mapfile -t MODELS < <(find -L "$HF_CACHE" -name "*.gguf" -type f \
    ! -path "*/MTP/*" \
    ! -path "*/draft*" \
    2>/dev/null | sort)

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "No models found in $HF_CACHE"
    exit 0
fi

echo "Found ${#MODELS[@]} model(s):"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo

# Clear previous results
echo "Model Benchmark Results" > "$RESULTS_FILE"
echo "Generated: $(date)" >> "$RESULTS_FILE"
echo "Benchmark tool: $BENCH_TOOL" >> "$RESULTS_FILE"
echo "Environment: GGML_CUDA_REGISTER_HOST=$GGML_CUDA_REGISTER_HOST, GGML_SCHED_PREFETCH_EXPERTS=$GGML_SCHED_PREFETCH_EXPERTS" >> "$RESULTS_FILE"
echo "==========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

TOTAL=${#MODELS[@]}
PASSED=0
FAILED=0

for model in "${MODELS[@]}"; do
    # Use the last path component as the display name
    model_name=$(basename "$model")

    echo "--------------------------------------"
    echo "Benchmarking: $model_name"
    echo "--------------------------------------"
    echo

    echo "## $model_name" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    set +e
    output=$("$BENCH_TOOL" -m "$model" -ngl 99 -p 128 -n 128 -b 8 2>&1)
    status=$?
    set -e

    # Check for CUDA OOM
    if echo "$output" | grep -q "out of memory"; then
        echo "[CUDA OOM] Model too large for GPU"
        echo "" >> "$RESULTS_FILE"
        echo "**[FAIL] CUDA OOM - Model too large for GPU**" >> "$RESULTS_FILE"
        echo "```" >> "$RESULTS_FILE"
        echo "$output" >> "$RESULTS_FILE"
        echo "```" >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
        FAILED=$((FAILED + 1))
    # Check for other errors
    elif [[ $status -ne 0 ]]; then
        echo "[FAIL] Error running benchmark"
        echo "" >> "$RESULTS_FILE"
        echo "**[FAIL] Error (exit code: $status)**" >> "$RESULTS_FILE"
        echo "```" >> "$RESULTS_FILE"
        echo "$output" >> "$RESULTS_FILE"
        echo "```" >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
        FAILED=$((FAILED + 1))
    # Success
    else
        echo "[PASS] Benchmark complete"
        echo "" >> "$RESULTS_FILE"
        echo "```" >> "$RESULTS_FILE"
        echo "$output" >> "$RESULTS_FILE"
        echo "```" >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
        PASSED=$((PASSED + 1))
    fi
done

echo
echo "=========================================="
echo "Benchmark complete!"
echo "Results saved to: $RESULTS_FILE"
echo "=========================================="
echo "Summary: $PASSED passed, $FAILED failed out of $TOTAL models"
