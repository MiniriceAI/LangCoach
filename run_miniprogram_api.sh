#!/bin/bash
# Start the LangCoach Mini Program API
# 统一的 API 服务，端口 8600
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
source /workspace/miniconda3/bin/activate base

# Default settings
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8600}
PRELOAD=${PRELOAD_MODELS:-false}

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
