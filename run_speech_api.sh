#!/bin/bash
# Start the Speech API server for LangCoach
# This provides REST endpoints for TTS and STT services

set -e

cd /workspace/LangCoach

# Use base conda environment (has PyTorch with CUDA)
source /workspace/miniconda3/bin/activate base

# Default settings
HOST=${SPEECH_API_HOST:-0.0.0.0}
PORT=${SPEECH_API_PORT:-8301}
PRELOAD=${PRELOAD_MODELS:-true}  # 默认预加载模型，避免首次请求超时

# Aggressively kill any existing process on the port
echo "Cleaning up port $PORT..."
for pid in $(lsof -t -i:$PORT 2>/dev/null); do
    echo "Killing process $pid on port $PORT"
    kill -9 $pid 2>/dev/null || true
done
sleep 2

# Double check with fuser
fuser -k $PORT/tcp 2>/dev/null || true
sleep 1

echo "Starting LangCoach Speech API..."
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Preload models: $PRELOAD"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# Set environment variables
export SPEECH_API_HOST=$HOST
export SPEECH_API_PORT=$PORT
export PRELOAD_MODELS=$PRELOAD

# Run the API server directly (not as module to avoid double import)
exec uvicorn src.api.speech_api:app --host $HOST --port $PORT
