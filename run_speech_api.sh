#!/bin/bash
# Start the Speech API server for LangCoach
# This provides REST endpoints for TTS and STT services

cd /workspace/LangCoach

# Default settings
HOST=${SPEECH_API_HOST:-0.0.0.0}
PORT=${SPEECH_API_PORT:-8301}
PRELOAD=${PRELOAD_MODELS:-false}

echo "Starting LangCoach Speech API..."
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Preload models: $PRELOAD"

# Set environment variables
export SPEECH_API_HOST=$HOST
export SPEECH_API_PORT=$PORT
export PRELOAD_MODELS=$PRELOAD

# Run the API server
python -m uvicorn src.api.speech_api:app --host $HOST --port $PORT
