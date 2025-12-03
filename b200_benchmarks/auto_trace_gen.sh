
#!/bin/bash
set -e  # Exit on error
# ============================================================================
# Configuration & Arguments
# ============================================================================
# Model configuration
MODEL_NAME="${MODEL_NAME:-deepseek-ai/deepseek-coder-1.3b-instruct}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# Benchmarking configuration - format: "input_len,output_len,concurrency,num_prompts"
TEST_SPECS="${TEST_SPECS:-1024,128,4,100 512,256,8,100 1024,512,16,100}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-3600}"
# Server configuration
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"
# Output configuration
RESULT_DIR="${RESULT_DIR:-${HOME}/logs_sglang}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="${RESULT_DIR}/${TIMESTAMP}"
TRACES_DIR="${OUTPUT_DIR}/traces_sglang"
BENCHMARK_DIR="${OUTPUT_DIR}/benchmarks_sglang"
# Docker configuration
DOCKER_IMAGE="${DOCKER_IMAGE:-lmsysorg/sglang:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-mkhalidm_sglang}"
HUGGINGFACE_CACHE="${HUGGINGFACE_CACHE:-/data/huggingface-cache}"
# Sanity check configuration
SANITY_CHECK_PROMPT="${SANITY_CHECK_PROMPT:-Write a hello world program in Python}"
SANITY_CHECK_MAX_TOKENS="${SANITY_CHECK_MAX_TOKENS:-100}"
# Logging
LOG_TZ="${LOG_TZ:-Europe/Helsinki}"
STREAM_LOGS="${STREAM_LOGS:-1}"
TRACE="${TRACE:-0}"
#Profiling
#https://rocm.blogs.amd.com/software-tools-optimization/kernel-analysis-deep/README.html
export SGLANG_TORCH_PROFILER_DIR=/workspace/profile/

[[ "$TRACE" == "1" ]] && set -x
# ============================================================================
# Setup
# ============================================================================
mkdir -p "$OUTPUT_DIR" "$TRACES_DIR" "$BENCHMARK_DIR"
# Save configuration
cat > "${OUTPUT_DIR}/config.txt" << EOF
Benchmark Configuration
=======================
Date: $(date)
Model: $MODEL_NAME
Tensor Parallel Size: $TP_SIZE
Max Model Length: $MAX_MODEL_LEN
Test Specs: $TEST_SPECS
GPU Memory Utilization: $GPU_MEMORY_UTIL
Output Directory: $OUTPUT_DIR
EOF
echo "Configuration saved to ${OUTPUT_DIR}/config.txt"
# Export HF token
export HF_TOKEN=$(cat ~/huggingface 2>/dev/null || echo "")
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
# ============================================================================
# Helper Functions
# ============================================================================
ts() { TZ="${LOG_TZ}" date +%Y%m%d-%H%M%S; }
STAMP="$(ts)"
SERVER_LOG="${OUTPUT_DIR}/sglang_server_${STAMP}.log"
STATUS_LOG="${OUTPUT_DIR}/status_${STAMP}.log"
LOGGER_PID=""
stop_log_stream() {
    if [[ -n "${LOGGER_PID}" ]] && kill -0 "${LOGGER_PID}" 2>/dev/null; then
        kill "${LOGGER_PID}" >/dev/null 2>&1 || true
        wait "${LOGGER_PID}" 2>/dev/null || true
        LOGGER_PID=""
    fi
}
cleanup() {
    echo "Cleaning up..."
    stop_log_stream
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
}
trap cleanup EXIT
# ============================================================================
# Start SGLang Server
# ============================================================================
echo "Starting SGLang server with model: $MODEL_NAME"
docker rm -f "$CONTAINER_NAME" 
docker run -d \
  --gpus=all \
  --name $CONTAINER_NAME \
  --ipc=host \
  --network=host \
  --privileged \
  --security-opt seccomp=unconfined \
  --ulimit core=0:0 \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
  -e TOKENIZERS_PARALLELISM=true \
  -e TORCH_CUDA_ARCH_LIST="10.0" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN \
  -e HF_HOME=/root/.cache/huggingface \
  -e SGLANG_TORCH_PROFILER_DIR=/workspace/profile \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -v "${HUGGINGFACE_CACHE}/hub:/root/.cache/huggingface/hub" \
  -v "${HOME}:/workspace" \
  -v "${BENCHMARK_DIR}:/workspace/benchmarks" \
  -v "${TRACES_DIR}:/workspace/profile" \
  -w /workspace \
  $DOCKER_IMAGE \
  python3 -m sglang.launch_server \
    --model-path $MODEL_NAME \
    --host 0.0.0.0 \
    --port $PORT \
    --tp-size $TP_SIZE \
    --trust-remote-code \
    --enable-metrics \
    --mem-fraction-static $GPU_MEMORY_UTIL \
    --max-total-tokens $MAX_MODEL_LEN \
    --chunked-prefill-size 8192 \
    --max-running-requests 64 \
    --cuda-graph-max-bs 64 \
    --disable-radix-cache \
    --attention-backend flashinfer \

# Stream logs to file
: > "$SERVER_LOG"; : > "$STATUS_LOG"
if [[ "${STREAM_LOGS}" == "1" ]]; then
    if command -v stdbuf >/dev/null 2>&1; then
        ( TZ="${LOG_TZ}" stdbuf -oL -eL docker logs -f --since=0s "$CONTAINER_NAME" 2>&1 \
          | awk '{ print strftime("[%Y-%m-%dT%H:%M:%S%z]"), $0; fflush(); }' \
          | tee -a "$SERVER_LOG" ) &
    else
        ( TZ="${LOG_TZ}" docker logs -f --since=0s "$CONTAINER_NAME" 2>&1 \
          | awk '{ print strftime("[%Y-%m-%dT%H:%M:%S%z]"), $0; fflush(); }' \
          | tee -a "$SERVER_LOG" ) &
    fi
    LOGGER_PID=$!
fi
echo "Server logs being written to: $SERVER_LOG"
# Fail fast if container died
sleep 2
state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo 'missing')"
if [[ "$state" != "running" ]]; then
    echo "ERROR: Server container exited early. Last 200 lines:" | tee -a "$STATUS_LOG"
    docker logs --tail 200 "$CONTAINER_NAME" 2>&1 || true
    exit 1
fi
# Wait for server to be ready
deadline=$((SECONDS+1200))
echo "Waiting for server health at http://${HOST}:${PORT}/health ..."
until code="$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" || true)"; [[ "$code" == "200" ]]; do
    state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo 'missing')"
    if [[ "$state" != "running" ]]; then
        echo "Server container stopped before becoming healthy. Last 200 lines:" | tee -a "$STATUS_LOG"
        docker logs --tail 200 "$CONTAINER_NAME" 2>&1 || true
        stop_log_stream
        exit 1
    fi
    if (( SECONDS > deadline )); then
        echo "Server did not become healthy in time. Last 200 log lines:" | tee -a "$STATUS_LOG"
        docker logs --tail 200 "$CONTAINER_NAME" 2>&1 || true
        stop_log_stream
        docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
        exit 1
    fi
    sleep 2
done
echo "Health OK."
# ============================================================================
# Sanity Check
# ============================================================================
echo "Running sanity check..."
echo "Testing completion endpoint..."
timeout 30 bash -c 'until curl -s http://'${HOST}':'${PORT}'/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"'${MODEL_NAME}'\",\"prompt\":\"test\",\"max_tokens\":1}" | grep -q "choices"; do sleep 2; done'
SANITY_OUTPUT=$(curl -s "http://${HOST}:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_NAME\",
    \"prompt\": \"$SANITY_CHECK_PROMPT\",
    \"max_tokens\": $SANITY_CHECK_MAX_TOKENS,
    \"temperature\": 0.7
  }")
echo "$SANITY_OUTPUT" > "${OUTPUT_DIR}/sanity_check.json"
# Validate output
if echo "$SANITY_OUTPUT" | grep -q '"choices"'; then
    if command -v jq >/dev/null 2>&1; then
        GENERATED_TEXT=$(echo "$SANITY_OUTPUT" | jq -r '.choices[0].text' 2>/dev/null || echo "Unable to extract text")
    else
        GENERATED_TEXT=$(echo "$SANITY_OUTPUT" | grep -o '"text":"[^"]*"' | head -1 | sed 's/"text":"\(.*\)"/\1/' || echo "Unable to extract text")
    fi
    echo "✓ Sanity check passed!"
    echo "Generated text: $GENERATED_TEXT"
else
    echo "✗ Sanity check failed!"
    echo "Response: $SANITY_OUTPUT"
    exit 1
fi
# ============================================================================
# Run Benchmarks
# ============================================================================
echo "Starting benchmarking..."
SCENARIO_ID=0
# Parse test specs: "input_len,output_len,concurrency,num_prompts"
for spec in $TEST_SPECS; do
    IFS=',' read -r input_len output_len conc num_prompts <<< "$spec"
    
    SCENARIO_ID=$((SCENARIO_ID+1))
    SCENARIO_NAME="isl${input_len}_osl${output_len}_c${conc}_np${num_prompts}"
    
    echo ""
    echo "=========================================="
    echo "Running Scenario $SCENARIO_ID: $SCENARIO_NAME"
    echo "  Input Length: $input_len"
    echo "  Output Length: $output_len"
    echo "  Concurrency: $conc"
    echo "  Num Prompts: $num_prompts"
    echo "=========================================="
    
    
    # Warmup 
    echo -e "\n[INFO] Starting warmup..." 
    set +e
    docker exec $CONTAINER_NAME \
        python3 -m sglang.bench_serving \
        --host $HOST \
        --port $PORT \
        --backend sglang \
        --tokenizer $MODEL_NAME \
        --dataset-name random \
        --random-input-len $input_len \
        --random-output-len $output_len \
        --num-prompts $conc \
        --request-rate inf \
        --max-concurrency $conc \
        2>&1 | tee "$BENCH_STDOUT_LOG"
    bench_rc=${PIPESTATUS[0]}
    set -e
    
    
    RESULT_JSON="${SCENARIO_NAME}_${STAMP}.json"
    BENCH_STDOUT_LOG="${TRACES_DIR}/${SCENARIO_NAME}.bench.log"
    
    # Main benchmark
    echo -e "\n[INFO] Starting main benchmark..."
    #docker exec $CONTAINER_NAME curl http://localhost:$PORT/start_profile
    set +e
    docker exec $CONTAINER_NAME \
        python3 -m sglang.bench_serving \
        --host $HOST \
        --port $PORT \
        --backend sglang \
        --tokenizer $MODEL_NAME \
        --dataset-name random \
        --random-input-len $input_len \
        --random-output-len $output_len \
        --num-prompts $num_prompts \
        --request-rate inf \
        --max-concurrency $conc \
        --profile \
        --output-file "/workspace/benchmarks/${RESULT_JSON}" \
        2>&1 | tee "$BENCH_STDOUT_LOG"
    bench_rc=${PIPESTATUS[0]}
    set -e
    #docker exec $CONTAINER_NAME curl http://localhost:$PORT/stop_profile
    
    echo "Benchmark exit code: ${bench_rc}" | tee -a "$STATUS_LOG"
    
    if [ $bench_rc -eq 0 ]; then
        echo "✓ Completed scenario: $SCENARIO_NAME"
    else
        echo "✗ Failed scenario: $SCENARIO_NAME (exit code: $bench_rc)" | tee -a "$STATUS_LOG"
    fi
    
    sleep 5  # Cool down between scenarios
done
echo ""
echo "=========================================="
echo "Benchmarking complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
# ============================================================================
# Stop Server
# ============================================================================
echo "Stopping SGLang server..." | tee -a "$STATUS_LOG"
stop_log_stream
docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
echo "Server stopped." | tee -a "$STATUS_LOG"
# ============================================================================
# Summary
# ============================================================================
echo ""
echo "Output locations:"
echo "  Config: ${OUTPUT_DIR}/config.txt"
echo "  Server log: ${SERVER_LOG}"
echo "  Status log: ${STATUS_LOG}"
echo "  Benchmark results: ${BENCHMARK_DIR}/"
echo "  Traces/logs: ${TRACES_DIR}/"
# Check if any JSON files are missing
missing_count=0
for spec in $TEST_SPECS; do
    IFS=',' read -r input_len output_len conc num_prompts <<< "$spec"
    SCENARIO_NAME="isl${input_len}_osl${output_len}_c${conc}_np${num_prompts}"
    RESULT_JSON="${SCENARIO_NAME}_${STAMP}.json"
    if [[ ! -f "${BENCHMARK_DIR}/${RESULT_JSON}" ]]; then
        echo "WARN: Benchmark JSON not found at ${BENCHMARK_DIR}/${RESULT_JSON}" | tee -a "$STATUS_LOG"
        missing_count=$((missing_count+1))
    fi
done
if [ $missing_count -eq 0 ]; then
    echo "✓ All benchmark results saved successfully"
else
    echo "✗ Missing $missing_count benchmark result file(s)"
fi
