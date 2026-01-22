#!/bin/bash
# Start the LangCoach Mini Program API
# 统一的 API 服务，端口 8600
#
# 服务架构:
# - LLM: Ollama + GLM-4-9B (hf.co/unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL)
# - TTS: Edge-TTS (Microsoft Azure) - 快速模式，无需预加载
# - STT: unsloth/whisper-large-v3 + 4bit quantization
#
# 提供以下功能:
# - 对话管理 (/api/chat/*)
# - 语音识别 (/api/transcribe)
# - 语音合成 (/api/synthesize)
# - 词典查询 (/api/dictionary)
# - 用户认证 (/api/auth/*)
#
# Usage:
#   ./run_miniprogram_api.sh

set -e

cd /workspace/LangCoach

# Use base conda environment
source /workspace/miniconda3/bin/activate lm

# Default settings
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8600}

# 默认预加载模型，避免首次请求超时
PRELOAD=${PRELOAD_MODELS:-true}

# Clean up port
echo "Cleaning up port $PORT..."
for pid in $(lsof -t -i:$PORT 2>/dev/null); do
    echo "Killing process $pid on port $PORT"
    kill -9 $pid 2>/dev/null || true
done
sleep 2

fuser -k $PORT/tcp 2>/dev/null || true
sleep 1

echo "=========================================="
echo "  LangCoach Mini Program API"
echo "=========================================="
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Preload models: $PRELOAD"
echo "  Python: $(which python)"
echo ""
echo "  Services:"
echo "    LLM: Ollama + GLM-4-9B"
echo "    TTS: Edge-TTS (fast mode)"
echo "    STT: Whisper-large-v3 (4bit)"
echo ""
echo "  Endpoints:"
echo "    Health:     GET  /health"
echo "    Scenarios:  GET  /api/scenarios"
echo "    Chat Start: POST /api/chat/start"
echo "    Chat Msg:   POST /api/chat/message"
echo "    Transcribe: POST /api/transcribe"
echo "    Synthesize: POST /api/synthesize"
echo "    Dictionary: GET  /api/dictionary?word=xxx"
echo "=========================================="

# Set environment variables
export API_HOST=$HOST
export API_PORT=$PORT
export PRELOAD_MODELS=$PRELOAD

# Run the API server
exec uvicorn src.api.miniprogram_api:app --host $HOST --port $PORT
