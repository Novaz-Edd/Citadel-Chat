#!/bin/bash
# ─────────────────────────────────────────────
# Citadel-Chat Startup Script
# Starts Ollama → Pulls models → Starts FastAPI
# ─────────────────────────────────────────────
set -e

echo "🏰 Citadel-Chat: Starting up..."

# 0. Setup persistent data directory
DATA_DIR=${DATA_DIR:-/app}
mkdir -p "$DATA_DIR/citadel_vault" "$DATA_DIR/citadel_memory"

# Store Ollama models on persistent disk if DATA_DIR is set
if [ "$DATA_DIR" != "/app" ]; then
    export OLLAMA_MODELS="$DATA_DIR/ollama_models"
    mkdir -p "$OLLAMA_MODELS"
fi

# 1. Memory optimization — unload models after each request
# so embedding + chat models don't compete for RAM
export OLLAMA_KEEP_ALIVE=0
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1

# 2. Start Ollama server in background
echo "📦 Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# 3. Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to initialize..."
MAX_WAIT=60
WAITED=0
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "❌ Ollama failed to start within ${MAX_WAIT}s"
        exit 1
    fi
done
echo "✅ Ollama is ready!"

# 4. Pull required models (skip if already present)
MODEL_NAME=${MODEL_NAME:-"qwen2.5:1.5b"}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-"nomic-embed-text"}

echo "📥 Pulling model: $MODEL_NAME ..."
ollama pull "$MODEL_NAME"

echo "📥 Pulling embedding model: $EMBEDDING_MODEL ..."
ollama pull "$EMBEDDING_MODEL"

echo "✅ All models ready!"

# 5. Start FastAPI application
PORT=${PORT:-8000}
echo "🚀 Starting Citadel-Chat on port $PORT..."
exec python main.py
